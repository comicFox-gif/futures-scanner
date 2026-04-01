"""
Strategy 2 — Support & Resistance Bounce
-----------------------------------------
Timeframes:
  4H  → detect key S/R levels (swing highs/lows with touch count)
  15m → confirm bounce/rejection at those levels

Stage 1 WARNING:
  Price approaching a key level (within 1.5x ATR)
  Level has 2+ touches (confirmed significant)

Stage 2 CONFIRMED:
  Price inside touch zone (within 0.5x ATR)
  Bounce pattern confirmed: hammer / shooting star / engulfing candle
  Volume above average
  RSI not extreme
  Fake breakout filters pass

Quality score 1-5 stars:
  + Level strength (touch count)
  + Volume ratio
  + Pattern clarity (wick ratio)
  + RSI sweet spot
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    find_key_levels,
    price_near_level,
    price_approaching_level,
    is_hammer,
    is_shooting_star,
    is_bullish_engulfing,
    is_bearish_engulfing,
    candle_body_ratio,
    volume_building,
)

logger = logging.getLogger("futures_bot.sr_bounce")


class SRBounceStrategy:
    NAME = "S/R Bounce"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        flt = cfg.get("filters", {})
        sr  = cfg.get("sr_bounce", {})

        # Indicators
        self.ema_fast         = s["ema_fast"]
        self.ema_mid          = s["ema_mid"]
        self.ema_slow         = s["ema_slow"]
        self.ema_trend        = s["ema_trend"]
        self.macd_fast        = s["macd_fast"]
        self.macd_slow        = s["macd_slow"]
        self.macd_signal      = s["macd_signal"]
        self.rsi_period       = s["rsi_period"]
        self.atr_period       = s["atr_period"]
        self.volume_sma_period = s["volume_sma_period"]

        # S/R level detection settings
        self.sr_n_left        = sr.get("swing_lookback_left", 5)
        self.sr_n_right       = sr.get("swing_lookback_right", 2)
        self.sr_cluster_pct   = sr.get("cluster_pct", 0.003)
        self.sr_max_levels    = sr.get("max_levels", 6)
        self.sr_min_touches   = sr.get("min_touches", 2)
        self.touch_atr_mult   = sr.get("touch_atr_multiplier", 0.5)
        self.approach_atr_mult = sr.get("approach_atr_multiplier", 1.5)

        # Risk levels
        self.atr_sl_mult      = sig.get("atr_sl_multiplier", 1.5)
        self.tp1_rr           = sig["tp1_rr"]
        self.tp2_rr           = sig["tp2_rr"]
        self.tp3_rr           = sig["tp3_rr"]

        # Filters
        self.volume_filter_mult      = s["volume_filter_multiplier"]
        self.volume_building_candles = flt.get("volume_building_candles", 2)
        self.rsi_long_min    = s["rsi_long_min"]
        self.rsi_long_max    = s["rsi_long_max"]
        self.rsi_short_min   = s["rsi_short_min"]
        self.rsi_short_max   = s["rsi_short_max"]

    # ------------------------------------------------------------------
    # Indicator enrichment
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    # ------------------------------------------------------------------
    # Quality score 1-5
    # ------------------------------------------------------------------

    def _quality_score(self, level_strength: int, vol_ratio: float,
                       wick_ratio: float, rsi: float) -> int:
        score = 1  # base — passed all filters
        if level_strength >= 4:
            score += 1
        elif level_strength >= 2:
            score += 0.5
        if vol_ratio >= 2.0:
            score += 1
        elif vol_ratio >= 1.5:
            score += 0.5
        if wick_ratio >= 0.6:
            score += 1
        elif wick_ratio >= 0.4:
            score += 0.5
        if 40 <= rsi <= 60:
            score += 1
        elif 35 <= rsi <= 65:
            score += 0.5
        return min(5, round(score))

    # ------------------------------------------------------------------
    # Bounce pattern check
    # ------------------------------------------------------------------

    def _bullish_bounce(self, entry_df: pd.DataFrame) -> tuple[bool, float]:
        """
        Returns (pattern_found, wick_ratio).
        Checks hammer OR bullish engulfing on last confirmed candle.
        """
        row = entry_df.iloc[-2]
        hammer    = is_hammer(row)
        engulfing = is_bullish_engulfing(entry_df, -2)
        if not (hammer or engulfing):
            return False, 0.0
        # Wick ratio = lower wick / total range (quality indicator)
        total_rng = row["high"] - row["low"]
        lo_wick   = min(row["open"], row["close"]) - row["low"]
        wick_ratio = lo_wick / total_rng if total_rng > 0 else 0
        return True, wick_ratio

    def _bearish_bounce(self, entry_df: pd.DataFrame) -> tuple[bool, float]:
        """
        Returns (pattern_found, wick_ratio).
        Checks shooting star OR bearish engulfing.
        """
        row = entry_df.iloc[-2]
        star      = is_shooting_star(row)
        engulfing = is_bearish_engulfing(entry_df, -2)
        if not (star or engulfing):
            return False, 0.0
        total_rng  = row["high"] - row["low"]
        up_wick    = row["high"] - max(row["open"], row["close"])
        wick_ratio = up_wick / total_rng if total_rng > 0 else 0
        return True, wick_ratio

    # ------------------------------------------------------------------
    # Main signal generator
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        htf_df: pd.DataFrame,   # 4H — for S/R level detection
        entry_df: pd.DataFrame, # 15m — for bounce confirmation
    ) -> dict:
        """
        Returns signal dict:
          {stage, direction, symbol, entry, sl, tp1, tp2, tp3,
           rsi, vol_ratio, quality, level_price, level_touches, reason}
        or None if no signal.
        """
        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])
        vol_ratio = row["volume"] / row["volume_sma"] if row["volume_sma"] > 0 else 0

        # Detect key levels from 4H
        levels = find_key_levels(
            htf_df,
            n_left=self.sr_n_left,
            n_right=self.sr_n_right,
            cluster_pct=self.sr_cluster_pct,
            max_levels=self.sr_max_levels,
        )

        if not levels:
            return None

        # Filter: only levels with minimum touches
        strong_levels = [l for l in levels if l["touches"] >= self.sr_min_touches]

        # Separate supports below price and resistances above price
        supports     = [l for l in strong_levels if l["type"] == "support"    and l["price"] < price]
        resistances  = [l for l in strong_levels if l["type"] == "resistance" and l["price"] > price]

        # Find nearest level of each type
        nearest_sup  = max(supports,    key=lambda x: x["price"]) if supports    else None
        nearest_res  = min(resistances, key=lambda x: x["price"]) if resistances else None

        # ---- LONG: bounce from support ----
        if nearest_sup:
            lv_price   = nearest_sup["price"]
            lv_touches = nearest_sup["touches"]
            lv_strength = nearest_sup["strength"]

            approaching = price_approaching_level(price, lv_price, atr, self.approach_atr_mult)
            touching    = price_near_level(price, lv_price, atr, self.touch_atr_mult)

            if touching:
                # Check bounce pattern
                bounce_ok, wick_ratio = self._bullish_bounce(entry_df)
                rsi_ok  = self.rsi_long_min <= rsi <= self.rsi_long_max
                vol_ok  = vol_ratio >= self.volume_filter_mult
                vol_bld = volume_building(entry_df, self.volume_building_candles)

                if bounce_ok and rsi_ok and vol_ok and vol_bld:
                    quality = self._quality_score(lv_strength, vol_ratio, wick_ratio, rsi)
                    sl_dist = atr * self.atr_sl_mult
                    return {
                        "stage": 2, "direction": "long", "symbol": symbol,
                        "entry": price,
                        "sl":    price - sl_dist,
                        "tp1":   price + sl_dist * self.tp1_rr,
                        "tp2":   price + sl_dist * self.tp2_rr,
                        "tp3":   price + sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio,
                        "quality": quality,
                        "level_price": lv_price, "level_touches": lv_touches,
                        "atr": atr,
                        "reason": (
                            f"Support bounce @ {lv_price:.4f} "
                            f"({lv_touches} touches) | "
                            f"{'Hammer' if is_hammer(row) else 'Engulfing'} | "
                            f"Vol {vol_ratio:.1f}x"
                        ),
                    }

            elif approaching:
                # Stage 1 warning — price getting close
                if lv_touches >= self.sr_min_touches:
                    sl_dist = atr * self.atr_sl_mult
                    return {
                        "stage": 1, "direction": "long", "symbol": symbol,
                        "entry": price,
                        "sl":    lv_price - sl_dist,
                        "tp1":   lv_price + sl_dist * self.tp1_rr,
                        "tp2":   lv_price + sl_dist * self.tp2_rr,
                        "tp3":   lv_price + sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio,
                        "quality": lv_touches,
                        "level_price": lv_price, "level_touches": lv_touches,
                        "atr": atr,
                        "reason": (
                            f"Approaching support {lv_price:.4f} "
                            f"({lv_touches} touches) | "
                            f"Watch for bounce"
                        ),
                    }

        # ---- SHORT: rejection from resistance ----
        if nearest_res:
            lv_price    = nearest_res["price"]
            lv_touches  = nearest_res["touches"]
            lv_strength = nearest_res["strength"]

            approaching = price_approaching_level(price, lv_price, atr, self.approach_atr_mult)
            touching    = price_near_level(price, lv_price, atr, self.touch_atr_mult)

            if touching:
                bounce_ok, wick_ratio = self._bearish_bounce(entry_df)
                rsi_ok  = self.rsi_short_min <= rsi <= self.rsi_short_max
                vol_ok  = vol_ratio >= self.volume_filter_mult
                vol_bld = volume_building(entry_df, self.volume_building_candles)

                if bounce_ok and rsi_ok and vol_ok and vol_bld:
                    quality = self._quality_score(lv_strength, vol_ratio, wick_ratio, rsi)
                    sl_dist = atr * self.atr_sl_mult
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    price + sl_dist,
                        "tp1":   price - sl_dist * self.tp1_rr,
                        "tp2":   price - sl_dist * self.tp2_rr,
                        "tp3":   price - sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio,
                        "quality": quality,
                        "level_price": lv_price, "level_touches": lv_touches,
                        "atr": atr,
                        "reason": (
                            f"Resistance rejection @ {lv_price:.4f} "
                            f"({lv_touches} touches) | "
                            f"{'Shooting Star' if is_shooting_star(row) else 'Engulfing'} | "
                            f"Vol {vol_ratio:.1f}x"
                        ),
                    }

            elif approaching:
                if lv_touches >= self.sr_min_touches:
                    sl_dist = atr * self.atr_sl_mult
                    return {
                        "stage": 1, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    lv_price + sl_dist,
                        "tp1":   lv_price - sl_dist * self.tp1_rr,
                        "tp2":   lv_price - sl_dist * self.tp2_rr,
                        "tp3":   lv_price - sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio,
                        "quality": lv_touches,
                        "level_price": lv_price, "level_touches": lv_touches,
                        "atr": atr,
                        "reason": (
                            f"Approaching resistance {lv_price:.4f} "
                            f"({lv_touches} touches) | "
                            f"Watch for rejection"
                        ),
                    }

        return None
