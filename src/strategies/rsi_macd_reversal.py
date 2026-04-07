"""
Strategy 6 — RSI + MACD Reversal
----------------------------------
Direct port of the Pine Script indicator. Pure reversal strategy — no EMA filter.

Logic:
  LONG  → RSI below oversold (30) AND MACD bullish crossover
  SHORT → RSI above overbought (70) AND MACD bearish crossover

This is the opposite of the EMA Trend strategy (which is trend-following).
This strategy catches bottoms and tops — high-probability reversal entries.

Stage 1 WARNING:
  RSI approaching extreme zone (< 35 long / > 65 short)
  MACD histogram turning in signal direction (cross coming)

Stage 2 CONFIRMED:
  RSI in extreme zone (< 30 long / > 70 short)
  MACD line has crossed the signal line (confirmed momentum shift)

No EMA trend filter — fires in any market direction.
Quality score based on how extreme RSI is and MACD histogram strength.
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    is_macd_bullish_cross,
    is_macd_bearish_cross,
    macd_histogram_turning_positive,
    macd_histogram_turning_negative,
)

logger = logging.getLogger("futures_bot.rsi_macd_reversal")


class RSIMACDReversalStrategy:
    NAME = "RSI+MACD Reversal"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        rm  = cfg.get("rsi_macd_reversal", {})

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

        # RSI thresholds (configurable, defaults match Pine Script)
        self.rsi_oversold      = rm.get("rsi_oversold", 30)
        self.rsi_overbought    = rm.get("rsi_overbought", 70)
        self.rsi_warn_long     = rm.get("rsi_warn_long", 35)   # approaching oversold
        self.rsi_warn_short    = rm.get("rsi_warn_short", 65)  # approaching overbought

        # Signal levels
        self.atr_sl_mult       = sig.get("atr_sl_multiplier", 1.5)
        self.tp1_rr            = sig["tp1_rr"]
        self.tp2_rr            = sig["tp2_rr"]
        self.tp3_rr            = sig["tp3_rr"]

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal_p,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality_score(self, rsi: float, direction: str, macd_hist: float, atr: float, vol_ratio: float = 0.0) -> int:
        # 5 binary conditions — need all 5 for a confirmed signal
        score = 1  # C1: base — RSI at extreme (≤30/≥70) + MACD cross confirmed
        # C2: RSI in stronger extreme zone (more conviction)
        if direction == "long" and rsi < 25:    score += 1
        elif direction == "short" and rsi > 75: score += 1
        # C3: MACD histogram has meaningful strength (not just a noise tick)
        hist_str = abs(macd_hist) / atr if atr > 0 else 0
        if hist_str >= 0.05:                    score += 1
        # C4: volume confirms the reversal move
        if vol_ratio >= 1.3:                    score += 1
        # C5: RSI is not still moving deeper (turning up/down already)
        # Proxy: RSI < 28 for longs means it's at a meaningful low
        if direction == "long" and rsi < 28:    score += 1
        elif direction == "short" and rsi > 72: score += 1
        return score

    def _build(self, stage: int, direction: str, symbol: str,
               price: float, atr: float, rsi: float, reason: str, quality: int = 3,
               vol_ratio: float = 0.0) -> dict:
        sl_dist = atr * self.atr_sl_mult
        if direction == "long":
            return {
                "stage": stage, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": reason,
            }
        return {
            "stage": stage, "direction": "short", "symbol": symbol,
            "entry": price,
            "sl":    price + sl_dist,
            "tp1":   price - sl_dist * self.tp1_rr,
            "tp2":   price - sl_dist * self.tp2_rr,
            "tp3":   price - sl_dist * self.tp3_rr,
            "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
            "reason": reason,
        }

    def generate_signal(self, symbol: str, entry_df: pd.DataFrame) -> dict | None:
        """
        Only needs the entry timeframe (15m).
        No trend filter — fires in any market condition.
        """
        row       = entry_df.iloc[-2]
        price     = float(row["close"])
        atr       = float(row["atr"])
        rsi       = float(row["rsi"])
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0.0

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        macd_hist = float(row["macd_hist"]) if not pd.isna(row.get("macd_hist", float("nan"))) else 0.0

        # ── LONG: RSI oversold + MACD bullish cross ─────────────────────
        if rsi <= self.rsi_oversold and is_macd_bullish_cross(entry_df):
            quality = self._quality_score(rsi, "long", macd_hist, atr, vol_ratio)
            return self._build(
                2, "long", symbol, price, atr, rsi,
                f"RSI {rsi:.1f} oversold + MACD bullish cross | Reversal entry",
                quality, vol_ratio,
            )

        # LONG WARNING: RSI approaching oversold + MACD turning up
        if rsi <= self.rsi_warn_long and macd_histogram_turning_positive(entry_df):
            return self._build(
                1, "long", symbol, price, atr, rsi,
                f"RSI {rsi:.1f} approaching oversold | MACD turning up | Watch for cross",
                2,
            )

        # ── SHORT: RSI overbought + MACD bearish cross ──────────────────
        if rsi >= self.rsi_overbought and is_macd_bearish_cross(entry_df):
            quality = self._quality_score(rsi, "short", macd_hist, atr, vol_ratio)
            return self._build(
                2, "short", symbol, price, atr, rsi,
                f"RSI {rsi:.1f} overbought + MACD bearish cross | Reversal entry",
                quality, vol_ratio,
            )

        # SHORT WARNING: RSI approaching overbought + MACD turning down
        if rsi >= self.rsi_warn_short and macd_histogram_turning_negative(entry_df):
            return self._build(
                1, "short", symbol, price, atr, rsi,
                f"RSI {rsi:.1f} approaching overbought | MACD turning down | Watch for cross",
                2,
            )

        return None
