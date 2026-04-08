"""
Strategy — RSI Divergence
--------------------------
RSI divergence is one of the most reliable reversal signals in technical analysis.
When price and RSI disagree about momentum, a trend reversal is likely.

Regular Bullish Divergence (bearish → bullish reversal):
  Price: makes a lower low (new recent low)
  RSI:   makes a higher low (RSI doesn't confirm the new low)
  → Selling momentum is drying up. Smart money accumulating. Go long.

Regular Bearish Divergence (bullish → bearish reversal):
  Price: makes a higher high (new recent high)
  RSI:   makes a lower high (RSI fails to confirm the new high)
  → Buying momentum is exhausted. Distribution happening. Go short.

Timeframes:
  1H  → divergence detection (needs enough price history to find two swing points)
  15m → entry confirmation (MACD cross or reversal candle pattern)

Stage 1 WARNING:
  Divergence detected on 1H, but no 15m confirmation yet

Stage 2 CONFIRMED:
  Divergence present + 15m MACD histogram turned in divergence direction
  OR reversal candle pattern on 15m (hammer/shooting star/engulfing)

Quality score (1–5 stars):
  + RSI gap between the two divergence points (larger = stronger)
  + Candle pattern quality
  + RSI not already in extreme territory (room to run)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    rsi_bullish_divergence,
    rsi_bearish_divergence,
    macd_histogram_turning_positive,
    macd_histogram_turning_negative,
    is_hammer,
    is_shooting_star,
    is_bullish_engulfing,
    is_bearish_engulfing,
    candle_body_ratio,
    detect_liquidity_sweep,
)

logger = logging.getLogger("futures_bot.rsi_divergence")


class RSIDivergenceStrategy:
    NAME = "RSI Divergence"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        rd  = cfg.get("rsi_divergence", {})

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

        self.div_lookback  = rd.get("lookback_candles", 50)
        self.min_rsi_diff  = rd.get("min_rsi_difference", 3.0)
        # RSI range for valid divergence signals
        self.bull_rsi_max  = rd.get("bull_div_rsi_max", 55)   # don't long if RSI already high
        self.bear_rsi_min  = rd.get("bear_div_rsi_min", 45)   # don't short if RSI already low

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

    def _quality(self, curr_rsi: float, prior_rsi: float, body: float, direction: str, vol_ratio: float = 0.0) -> int:
        # 5 binary conditions — need all 5 for a confirmed signal
        score = 1  # C1: base — divergence confirmed on HTF
        rsi_gap = abs(curr_rsi - prior_rsi)
        if rsi_gap >= 8:            score += 1  # C2: clear RSI divergence (8+ point gap)
        if body >= 0.45:            score += 1  # C3: strong reversal candle body
        # C4: RSI at extreme (ideal reversal zone)
        if direction == "long" and curr_rsi < 45:   score += 1
        elif direction == "short" and curr_rsi > 55: score += 1
        if vol_ratio >= 1.3:        score += 1  # C5: volume spike on reversal candle
        return score

    def generate_signal(
        self,
        symbol: str,
        itf_df: pd.DataFrame,   # 1H — divergence detection
        entry_df: pd.DataFrame, # 15m — entry confirmation
    ) -> dict | None:
        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])
        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        # ADX gate: divergence in ranging markets is unreliable noise
        adx = float(row.get("adx", 25))
        if not pd.isna(adx) and adx < 18:
            return None

        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0.0
        body = candle_body_ratio(entry_df)
        sl_dist = atr * self.atr_sl_mult

        # HTF trend alignment from itf_df (used as trend frame)
        htf_row   = itf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        htf_bull  = not pd.isna(ema50_htf) and htf_price > ema50_htf
        htf_bear  = not pd.isna(ema50_htf) and htf_price < ema50_htf

        sweep = detect_liquidity_sweep(entry_df)

        # ── Bullish divergence → LONG ─────────────────────────────────────
        bull_div, curr_rsi, prior_rsi, div_reason = rsi_bullish_divergence(
            itf_df, self.div_lookback, self.min_rsi_diff
        )
        if bull_div and curr_rsi <= self.bull_rsi_max and htf_bull:
            # Stage 2: MACD confirms AND reversal pattern on 15m (both required)
            macd_turn   = macd_histogram_turning_positive(entry_df)
            rev_pattern = is_hammer(row) or is_bullish_engulfing(entry_df, -2)

            if macd_turn and rev_pattern:
                if sweep == "buy_side":
                    logger.debug(f"[RSI DIV] {symbol} LONG blocked — buy-side liquidity sweep")
                    return None
                quality = self._quality(curr_rsi, prior_rsi, body, "long", vol_ratio)
                return {
                    "stage": 2, "direction": "long", "symbol": symbol,
                    "entry": price,
                    "sl":    price - sl_dist,
                    "tp1":   price + sl_dist * self.tp1_rr,
                    "tp2":   price + sl_dist * self.tp2_rr,
                    "tp3":   price + sl_dist * self.tp3_rr,
                    "rsi": curr_rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                    "reason": f"{div_reason} | MACD + reversal candle confirmed",
                }

            # Stage 1: divergence only, no confirmation yet
            return {
                "stage": 1, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": curr_rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                "reason": f"{div_reason} | Awaiting 15m confirmation",
            }

        # ── Bearish divergence → SHORT ────────────────────────────────────
        bear_div, curr_rsi, prior_rsi, div_reason = rsi_bearish_divergence(
            itf_df, self.div_lookback, self.min_rsi_diff
        )
        if bear_div and curr_rsi >= self.bear_rsi_min and htf_bear:
            macd_turn   = macd_histogram_turning_negative(entry_df)
            rev_pattern = is_shooting_star(row) or is_bearish_engulfing(entry_df, -2)

            if macd_turn and rev_pattern:
                if sweep == "sell_side":
                    logger.debug(f"[RSI DIV] {symbol} SHORT blocked — sell-side liquidity sweep")
                    return None
                quality = self._quality(curr_rsi, prior_rsi, body, "short", vol_ratio)
                return {
                    "stage": 2, "direction": "short", "symbol": symbol,
                    "entry": price,
                    "sl":    price + sl_dist,
                    "tp1":   price - sl_dist * self.tp1_rr,
                    "tp2":   price - sl_dist * self.tp2_rr,
                    "tp3":   price - sl_dist * self.tp3_rr,
                    "rsi": curr_rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                    "reason": f"{div_reason} | MACD + reversal candle confirmed",
                }

            return {
                "stage": 1, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    price + sl_dist,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": curr_rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                "reason": f"{div_reason} | Awaiting 15m confirmation",
            }

        return None
