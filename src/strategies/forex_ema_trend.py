"""
Forex Strategy 1 — EMA Trend Inception
----------------------------------------
Multi-timeframe EMA momentum strategy adapted for forex markets.

Timeframes:
  4H  → macro trend (EMA alignment + EMA200 filter)
  1H  → intermediate confirmation
  15m → entry trigger

Stage 1 WARNING:
  4H trend aligned (EMA 9>21>50, price above/below EMA200)
  1H trend confirms direction (EMA 9>21, price > EMA50)
  15m MACD histogram turning in trend direction
  RSI in valid range

Stage 2 CONFIRMED:
  All Stage 1 conditions met +
  15m MACD crossover confirmed
  Candle body ratio filter (decisive candle, no doji)
  Wick against trade direction filter (no rejection wick)
  Consecutive closes in trend direction (follow-through)
  MACD histogram strength (not a noise cross)
  Active session filter (6am–9pm UTC)

Quality score (1–5 stars):
  + Candle body quality
  + RSI position (sweet spot 45-65)
  + MACD histogram strength relative to ATR
"""

from __future__ import annotations
import logging
from datetime import datetime
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    is_macd_bullish_cross,
    is_macd_bearish_cross,
    macd_histogram_turning_positive,
    macd_histogram_turning_negative,
    candle_body_ratio,
    upper_wick_ratio,
    lower_wick_ratio,
    consecutive_bullish_closes,
    consecutive_bearish_closes,
    macd_histogram_strong,
)

logger = logging.getLogger("forex_bot.ema_trend")

SESSION_START_UTC = 6   # London pre-open
SESSION_END_UTC   = 21  # NY close


class ForexEmaTrendStrategy:
    NAME = "FX EMA Trend"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        flt = cfg.get("filters", {})

        self.ema_fast          = s["ema_fast"]
        self.ema_mid           = s["ema_mid"]
        self.ema_slow          = s["ema_slow"]
        self.ema_trend         = s["ema_trend"]
        self.macd_fast         = s["macd_fast"]
        self.macd_slow         = s["macd_slow"]
        self.macd_signal_p     = s["macd_signal"]
        self.rsi_period        = s["rsi_period"]
        self.rsi_long_min      = s["rsi_long_min"]
        self.rsi_long_max      = s["rsi_long_max"]
        self.rsi_short_min     = s["rsi_short_min"]
        self.rsi_short_max     = s["rsi_short_max"]
        self.atr_period        = s["atr_period"]
        self.atr_sl_mult       = sig.get("atr_sl_multiplier", 1.5)
        self.volume_sma_period = s["volume_sma_period"]
        self.tp1_rr            = sig["tp1_rr"]
        self.tp2_rr            = sig["tp2_rr"]
        self.tp3_rr            = sig["tp3_rr"]

        # Fake-breakout filters
        self.min_body_ratio     = flt.get("min_body_ratio", 0.40)
        self.max_wick_against   = flt.get("max_wick_against_trade", 0.40)
        self.consecutive_closes = flt.get("consecutive_closes", 1)
        self.macd_hist_atr_mult = flt.get("macd_hist_atr_multiplier", 0.02)
        self.session_filter     = flt.get("session_filter", True)

    # ------------------------------------------------------------------
    # Indicator enrichment
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal_p,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    # ------------------------------------------------------------------
    # Session guard
    # ------------------------------------------------------------------

    def _in_session(self) -> bool:
        if not self.session_filter:
            return True
        h = datetime.utcnow().hour
        return SESSION_START_UTC <= h < SESSION_END_UTC

    # ------------------------------------------------------------------
    # Trend filters
    # ------------------------------------------------------------------

    def _htf_bullish(self, htf_df: pd.DataFrame) -> bool:
        """4H: EMA9 > EMA21 > EMA50, close > EMA200."""
        row = htf_df.iloc[-2]
        return (
            row[f"ema_{self.ema_fast}"] > row[f"ema_{self.ema_mid}"]
            and row[f"ema_{self.ema_mid}"] > row[f"ema_{self.ema_slow}"]
            and row["close"] > row[f"ema_{self.ema_trend}"]
        )

    def _htf_bearish(self, htf_df: pd.DataFrame) -> bool:
        """4H: EMA9 < EMA21 < EMA50, close < EMA200."""
        row = htf_df.iloc[-2]
        return (
            row[f"ema_{self.ema_fast}"] < row[f"ema_{self.ema_mid}"]
            and row[f"ema_{self.ema_mid}"] < row[f"ema_{self.ema_slow}"]
            and row["close"] < row[f"ema_{self.ema_trend}"]
        )

    def _itf_bullish(self, itf_df: pd.DataFrame) -> bool:
        """1H: EMA9 > EMA21, close > EMA50."""
        row = itf_df.iloc[-2]
        return (
            row[f"ema_{self.ema_fast}"] > row[f"ema_{self.ema_mid}"]
            and row["close"] > row[f"ema_{self.ema_slow}"]
        )

    def _itf_bearish(self, itf_df: pd.DataFrame) -> bool:
        """1H: EMA9 < EMA21, close < EMA50."""
        row = itf_df.iloc[-2]
        return (
            row[f"ema_{self.ema_fast}"] < row[f"ema_{self.ema_mid}"]
            and row["close"] < row[f"ema_{self.ema_slow}"]
        )

    # ------------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------------

    def _quality_score(self, body: float, rsi: float, macd_hist: float, atr: float) -> int:
        score = 0
        if body >= 0.45:              score += 1  # C1: decisive candle body
        if 40 <= rsi <= 65:           score += 1  # C2: RSI in trend zone
        hist_str = abs(macd_hist) / atr if atr > 0 else 0
        if hist_str >= 0.05:          score += 1  # C3: MACD histogram has strength
        score += 1                                # C4: base — both HTF+ITF aligned (gate passed)
        score += 1                                # C5: MACD cross confirmed (gate passed)
        return min(5, score)

    # ------------------------------------------------------------------
    # Entry filters (fake breakout prevention)
    # ------------------------------------------------------------------

    def _long_filters_pass(self, entry_df: pd.DataFrame, atr: float) -> tuple[bool, float]:
        """Returns (passed, body_ratio). Passes entry_df (DataFrame) to all filter functions."""
        body = candle_body_ratio(entry_df)
        if body < self.min_body_ratio:
            return False, body
        if upper_wick_ratio(entry_df) > self.max_wick_against:
            return False, body
        if not consecutive_bullish_closes(entry_df, self.consecutive_closes):
            return False, body
        if not macd_histogram_strong(entry_df, self.macd_hist_atr_mult):
            return False, body
        return True, body

    def _short_filters_pass(self, entry_df: pd.DataFrame, atr: float) -> tuple[bool, float]:
        """Returns (passed, body_ratio)."""
        body = candle_body_ratio(entry_df)
        if body < self.min_body_ratio:
            return False, body
        if lower_wick_ratio(entry_df) > self.max_wick_against:
            return False, body
        if not consecutive_bearish_closes(entry_df, self.consecutive_closes):
            return False, body
        if not macd_histogram_strong(entry_df, self.macd_hist_atr_mult):
            return False, body
        return True, body

    # ------------------------------------------------------------------
    # Signal levels builder
    # ------------------------------------------------------------------

    def _build(self, stage: int, direction: str, pair: str,
               price: float, atr: float, rsi: float, reason: str,
               quality: int = 2) -> dict:
        sl_dist = atr * self.atr_sl_mult
        if direction == "long":
            return {
                "stage": stage, "direction": "long", "symbol": pair,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": 0,
                "quality": quality, "atr": atr,
                "reason": reason,
            }
        return {
            "stage": stage, "direction": "short", "symbol": pair,
            "entry": price,
            "sl":    price + sl_dist,
            "tp1":   price - sl_dist * self.tp1_rr,
            "tp2":   price - sl_dist * self.tp2_rr,
            "tp3":   price - sl_dist * self.tp3_rr,
            "rsi": rsi, "vol_ratio": 0,
            "quality": quality, "atr": atr,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Main signal generator
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        pair: str,
        htf_df: pd.DataFrame,   # 4H — macro trend
        itf_df: pd.DataFrame,   # 1H — intermediate trend
        entry_df: pd.DataFrame, # 15m — entry trigger
    ) -> dict | None:
        if not self._in_session():
            return None

        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        macd_hist = float(row["macd_hist"]) if not pd.isna(row.get("macd_hist", float("nan"))) else 0.0

        # ── LONG ─────────────────────────────────────────────────────────
        if self._htf_bullish(htf_df) and self._itf_bullish(itf_df):
            rsi_ok = self.rsi_long_min <= rsi <= self.rsi_long_max

            if is_macd_bullish_cross(entry_df) and rsi_ok:
                passed, body = self._long_filters_pass(entry_df, atr)
                if passed:
                    quality = self._quality_score(body, rsi, macd_hist, atr)
                    return self._build(
                        2, "long", pair, price, atr, rsi,
                        f"4H+1H bull | MACD cross | RSI {rsi:.1f} | Body {body:.0%}",
                        quality,
                    )

            if macd_histogram_turning_positive(entry_df) and rsi_ok:
                return self._build(
                    1, "long", pair, price, atr, rsi,
                    f"4H+1H bull | MACD turning up | RSI {rsi:.1f} | Watch for cross",
                )

        # ── SHORT ────────────────────────────────────────────────────────
        if self._htf_bearish(htf_df) and self._itf_bearish(itf_df):
            rsi_ok = self.rsi_short_min <= rsi <= self.rsi_short_max

            if is_macd_bearish_cross(entry_df) and rsi_ok:
                passed, body = self._short_filters_pass(entry_df, atr)
                if passed:
                    quality = self._quality_score(body, rsi, macd_hist, atr)
                    return self._build(
                        2, "short", pair, price, atr, rsi,
                        f"4H+1H bear | MACD cross | RSI {rsi:.1f} | Body {body:.0%}",
                        quality,
                    )

            if macd_histogram_turning_negative(entry_df) and rsi_ok:
                return self._build(
                    1, "short", pair, price, atr, rsi,
                    f"4H+1H bear | MACD turning down | RSI {rsi:.1f} | Watch for cross",
                )

        return None
