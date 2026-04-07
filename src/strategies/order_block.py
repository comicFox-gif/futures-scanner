"""
Strategy — ICT Order Block
----------------------------
Order blocks are institutional footprints: the last candle before a large
impulse move. When price returns to that zone, institutions refill positions.

Bullish OB: last bearish candle before a strong up-move → demand zone → long
Bearish OB: last bullish candle before a strong down-move → supply zone → short

Timeframes:
  4H  → detect order blocks (significant institutional zones)
  15m → entry confirmation (bounce pattern inside OB zone)

Stage 1 WARNING:
  Price is approaching an OB zone (within 1.5x ATR)

Stage 2 CONFIRMED:
  Price is inside the OB zone
  Bounce candle confirmed (hammer / engulfing / strong close)
  RSI in valid range
  OB is fresh / un-mitigated (price hasn't closed through it before)

Quality score (1–5 stars):
  + OB recency (more recent = stronger institutional memory)
  + Body of the entry candle
  + RSI position
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    find_order_blocks,
    is_hammer,
    is_shooting_star,
    is_bullish_engulfing,
    is_bearish_engulfing,
    candle_body_ratio,
)

logger = logging.getLogger("futures_bot.order_block")


class OrderBlockStrategy:
    NAME = "Order Block"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        ob  = cfg.get("order_block", {})

        self.ema_fast          = s["ema_fast"]
        self.ema_mid           = s["ema_mid"]
        self.ema_slow          = s["ema_slow"]
        self.ema_trend         = s["ema_trend"]
        self.macd_fast         = s["macd_fast"]
        self.macd_slow         = s["macd_slow"]
        self.macd_signal_p     = s["macd_signal"]
        self.rsi_period        = s["rsi_period"]
        self.atr_period        = s["atr_period"]
        self.volume_sma_period = s["volume_sma_period"]
        self.rsi_long_min      = s["rsi_long_min"]
        self.rsi_long_max      = s["rsi_long_max"]
        self.rsi_short_min     = s["rsi_short_min"]
        self.rsi_short_max     = s["rsi_short_max"]

        # OB detection settings
        self.ob_lookback       = ob.get("lookback_candles", 80)
        self.ob_impulse_can    = ob.get("impulse_candles", 3)
        self.ob_impulse_atr    = ob.get("impulse_atr_mult", 2.0)
        self.approach_atr_mult = ob.get("approach_atr_mult", 1.5)
        self.touch_atr_mult    = ob.get("touch_atr_mult", 0.6)

        # Levels
        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.5)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal_p,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, formed_idx: int, total_len: int, body: float, rsi: float, vol_ratio: float = 0.0) -> int:
        # 5 binary conditions — need all 5 for a confirmed signal
        score = 1  # C1: base — price in OB zone + candle pattern confirmed
        recency = (total_len - formed_idx) / total_len
        if recency < 0.50:          score += 1  # C2: fresh OB (formed in last 50% of lookback)
        if body >= 0.45:            score += 1  # C3: strong entry candle body
        if 35 <= rsi <= 65:         score += 1  # C4: RSI in valid range
        if vol_ratio >= 1.3:        score += 1  # C5: volume confirms institutional interest
        return score

    def _bullish_pattern(self, entry_df: pd.DataFrame) -> bool:
        row = entry_df.iloc[-2]
        return is_hammer(row) or is_bullish_engulfing(entry_df, -2)

    def _bearish_pattern(self, entry_df: pd.DataFrame) -> bool:
        row = entry_df.iloc[-2]
        return is_shooting_star(row) or is_bearish_engulfing(entry_df, -2)

    def generate_signal(
        self,
        symbol: str,
        htf_df: pd.DataFrame,   # 4H — OB detection
        entry_df: pd.DataFrame, # 15m — entry confirmation
    ) -> dict | None:
        row       = entry_df.iloc[-2]
        price     = float(row["close"])
        atr       = float(row["atr"])
        rsi       = float(row["rsi"])
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0.0
        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        obs = find_order_blocks(
            htf_df,
            lookback=self.ob_lookback,
            impulse_candles=self.ob_impulse_can,
            impulse_atr_mult=self.ob_impulse_atr,
        )
        if not obs:
            return None

        body = candle_body_ratio(entry_df)

        # ── Bullish OBs: look for longs ───────────────────────────────────
        bullish_obs = [o for o in obs if o["type"] == "bullish"]
        for ob in reversed(bullish_obs):  # most recent first
            ob_high = ob["high"]
            ob_low  = ob["low"]
            ob_mid  = ob["mid"]

            in_zone     = ob_low <= price <= ob_high
            approaching = price > ob_high and price - ob_high < atr * self.approach_atr_mult

            if in_zone:
                rsi_ok  = self.rsi_long_min <= rsi <= self.rsi_long_max
                pattern = self._bullish_pattern(entry_df)
                if rsi_ok and pattern:
                    quality = self._quality(ob["formed_idx"], len(htf_df), body, rsi, vol_ratio)
                    sl_dist = max(atr * self.atr_sl_mult, price - ob_low + atr * 0.3)
                    return {
                        "stage": 2, "direction": "long", "symbol": symbol,
                        "entry": price,
                        "sl":    price - sl_dist,
                        "tp1":   price + sl_dist * self.tp1_rr,
                        "tp2":   price + sl_dist * self.tp2_rr,
                        "tp3":   price + sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bullish OB {ob_low:.4f}–{ob_high:.4f} | "
                            f"{'Hammer' if is_hammer(row) else 'Engulfing'} | "
                            f"RSI {rsi:.1f}"
                        ),
                    }

            elif approaching:
                sl_dist = atr * self.atr_sl_mult
                return {
                    "stage": 1, "direction": "long", "symbol": symbol,
                    "entry": price,
                    "sl":    ob_low - atr * 0.3,
                    "tp1":   price + sl_dist * self.tp1_rr,
                    "tp2":   price + sl_dist * self.tp2_rr,
                    "tp3":   price + sl_dist * self.tp3_rr,
                    "rsi": rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                    "reason": f"Approaching bullish OB {ob_low:.4f}–{ob_high:.4f} | Watch for entry",
                }

        # ── Bearish OBs: look for shorts ──────────────────────────────────
        bearish_obs = [o for o in obs if o["type"] == "bearish"]
        for ob in reversed(bearish_obs):
            ob_high = ob["high"]
            ob_low  = ob["low"]

            in_zone     = ob_low <= price <= ob_high
            approaching = price < ob_low and ob_low - price < atr * self.approach_atr_mult

            if in_zone:
                rsi_ok  = self.rsi_short_min <= rsi <= self.rsi_short_max
                pattern = self._bearish_pattern(entry_df)
                if rsi_ok and pattern:
                    quality = self._quality(ob["formed_idx"], len(htf_df), body, rsi, vol_ratio)
                    sl_dist = max(atr * self.atr_sl_mult, ob_high - price + atr * 0.3)
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    price + sl_dist,
                        "tp1":   price - sl_dist * self.tp1_rr,
                        "tp2":   price - sl_dist * self.tp2_rr,
                        "tp3":   price - sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bearish OB {ob_low:.4f}–{ob_high:.4f} | "
                            f"{'Shooting Star' if is_shooting_star(row) else 'Engulfing'} | "
                            f"RSI {rsi:.1f}"
                        ),
                    }

            elif approaching:
                sl_dist = atr * self.atr_sl_mult
                return {
                    "stage": 1, "direction": "short", "symbol": symbol,
                    "entry": price,
                    "sl":    ob_high + atr * 0.3,
                    "tp1":   price - sl_dist * self.tp1_rr,
                    "tp2":   price - sl_dist * self.tp2_rr,
                    "tp3":   price - sl_dist * self.tp3_rr,
                    "rsi": rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                    "reason": f"Approaching bearish OB {ob_low:.4f}–{ob_high:.4f} | Watch for entry",
                }

        return None
