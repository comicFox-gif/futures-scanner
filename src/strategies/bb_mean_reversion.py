"""
Forex Strategy — Bollinger Band Mean Reversion
------------------------------------------------
Forex pairs spend ~70% of their time in ranging/mean-reverting conditions.
When price touches the outer Bollinger Band and shows a reversal candle,
it has historically high probability of reverting to the mean (BB midline).

Logic:
  Touch  : Price wicks into or closes at/beyond the outer BB (±2 std)
  Reject : Current candle closes back inside the band (rejection)
  Filter : RSI confirms overbought/oversold (RSI > 68 short, < 32 long)
  Filter : No strong directional trend (ADX < 28 — mean reversion works best in ranges)
  Session: London/NY only

Quality (5 binary conditions):
  C1: Price touched outer BB (wick or close at/beyond band)
  C2: Current candle closes back inside band (rejection confirmed)
  C3: RSI extreme (long: RSI < 32, short: RSI > 68)
  C4: ADX < 28 (ranging, not trending — mean reversion conditions)
  C5: Reversal candle pattern (hammer/shooting star/engulfing)
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    is_hammer,
    is_shooting_star,
    is_bullish_engulfing,
    is_bearish_engulfing,
)

logger = logging.getLogger("forex_bot.bb_mean_reversion")

SESSION_START = 6
SESSION_END   = 21


class BBMeanReversionStrategy:
    NAME = "BB Mean Reversion"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        bb  = cfg.get("bb_mean_reversion", {})

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

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.0)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        self.rsi_ob       = bb.get("rsi_overbought", 68)
        self.rsi_os       = bb.get("rsi_oversold", 32)
        self.adx_max      = bb.get("adx_max", 28)      # only trade in ranging conditions
        self.session_filter = bb.get("session_filter", True)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, touched_band: bool, rejected: bool, rsi_extreme: bool,
                 ranging: bool, reversal_candle: bool) -> int:
        score = 0
        if touched_band:    score += 1  # C1
        if rejected:        score += 1  # C2
        if rsi_extreme:     score += 1  # C3
        if ranging:         score += 1  # C4
        if reversal_candle: score += 1  # C5
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        if len(entry_df) < 25:
            return None

        if self.session_filter:
            now_h = datetime.now(timezone.utc).hour
            if not (SESSION_START <= now_h < SESSION_END):
                return None

        row   = entry_df.iloc[-2]
        prev  = entry_df.iloc[-3]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        for col in ("bb_upper", "bb_lower", "bb_mid"):
            if col not in entry_df.columns or pd.isna(row.get(col)):
                return None

        bb_upper = float(row["bb_upper"])
        bb_lower = float(row["bb_lower"])
        bb_mid   = float(row["bb_mid"])
        adx      = float(row.get("adx", 30))

        prev_high  = float(prev["high"])
        prev_low   = float(prev["low"])
        prev_close = float(prev["close"])

        sl_dist = atr * self.atr_sl_mult
        ranging = not pd.isna(adx) and adx < self.adx_max

        # LONG: previous candle touched/broke lower BB, current closes back inside
        touched_lower  = prev_low <= bb_lower or prev_close <= bb_lower
        rejected_lower = price > bb_lower and prev_close <= bb_lower
        rsi_os         = rsi < self.rsi_os
        bounce_candle  = is_hammer(row) or is_bullish_engulfing(entry_df, -2)

        if touched_lower and rejected_lower:
            quality = self._quality(touched_lower, rejected_lower, rsi_os, ranging, bounce_candle)
            if quality < 5:
                return None
            # TP targets: BB midline (TP1/TP2) and upper band (TP3 proxy)
            tp_mid = bb_mid
            return {
                "stage": 2, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                "reason": (
                    f"BB Lower bounce ↑ | RSI={rsi:.0f} | ADX={adx:.0f} | "
                    f"BB_mid={tp_mid:.5f}"
                ),
            }

        # SHORT: previous touched/broke upper BB, current closes back inside
        touched_upper  = prev_high >= bb_upper or prev_close >= bb_upper
        rejected_upper = price < bb_upper and prev_close >= bb_upper
        rsi_ob         = rsi > self.rsi_ob
        reject_candle  = is_shooting_star(row) or is_bearish_engulfing(entry_df, -2)

        if touched_upper and rejected_upper:
            quality = self._quality(touched_upper, rejected_upper, rsi_ob, ranging, reject_candle)
            if quality < 5:
                return None
            tp_mid = bb_mid
            return {
                "stage": 2, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    price + sl_dist,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                "reason": (
                    f"BB Upper rejection ↓ | RSI={rsi:.0f} | ADX={adx:.0f} | "
                    f"BB_mid={tp_mid:.5f}"
                ),
            }

        return None
