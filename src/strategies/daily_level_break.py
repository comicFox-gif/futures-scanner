"""
Forex Strategy — Previous Day High/Low Break + Retest
-------------------------------------------------------
Previous day's high and low are the most watched levels by institutional
traders worldwide. A clean break above PDH (or below PDL) followed by
a retest and hold is one of the cleanest forex setups.

Logic:
  Level   : Previous day's high (PDH) or low (PDL) — reset at 00:00 UTC
  Break   : Price closes above PDH (long) or below PDL (short)
  Retest  : Price pulls back to within 0.3x ATR of the broken level
  Hold    : Current candle closes back in the break direction (reclaim)
  Session : London/NY session only (6:00–21:00 UTC)

Quality (5 binary conditions):
  C1: Level is clean PDH/PDL (genuine prior-day extreme)
  C2: Break candle had volume >= 1.3x avg (real breakout)
  C3: Retest held the level (didn't close back through)
  C4: Bounce candle pattern on retest
  C5: RSI confirms (long 40-65, short 35-60)
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    is_hammer,
    is_shooting_star,
    is_bullish_engulfing,
    is_bearish_engulfing,
)

logger = logging.getLogger("forex_bot.daily_level_break")

SESSION_START = 6
SESSION_END   = 21


class DailyLevelBreakStrategy:
    NAME = "Daily Level Break"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        dl  = cfg.get("daily_level_break", {})

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

        self.retest_atr_mult = dl.get("retest_atr_mult", 0.3)
        self.vol_mult        = dl.get("volume_multiplier", 1.3)
        self.session_filter  = dl.get("session_filter", True)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _get_prev_day_levels(self, df: pd.DataFrame) -> tuple[float, float] | tuple[None, None]:
        """Get previous calendar day's high and low from a 1H/15m DataFrame."""
        if not hasattr(df.index, "date"):
            return None, None
        today     = df.index[-1].date()
        prev_day  = today - timedelta(days=1)
        prev_mask = [d == prev_day for d in df.index.date]
        prev_data = df[prev_mask]
        if len(prev_data) < 4:
            return None, None
        return float(prev_data["high"].max()), float(prev_data["low"].min())

    def _quality(self, clean_level: bool, vol_break: bool, held: bool,
                 bounce: bool, rsi: float, direction: str) -> int:
        score = 0
        if clean_level:  score += 1  # C1
        if vol_break:    score += 1  # C2
        if held:         score += 1  # C3
        if bounce:       score += 1  # C4
        if direction == "long"  and 40 <= rsi <= 65: score += 1  # C5
        elif direction == "short" and 35 <= rsi <= 60: score += 1
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        if len(entry_df) < 30:
            return None

        # Session filter
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

        pdh, pdl = self._get_prev_day_levels(entry_df)
        if pdh is None:
            return None

        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0
        sl_dist   = atr * self.atr_sl_mult
        zone      = atr * self.retest_atr_mult
        prev_close = float(prev["close"])

        # LONG: PDH broken, now retesting from above
        broke_above  = prev_close > pdh
        retesting    = abs(price - pdh) <= zone
        held_above   = price > pdh
        bounce_long  = is_hammer(row) or is_bullish_engulfing(entry_df, -2)
        vol_ok       = vol_ratio >= self.vol_mult

        if broke_above and retesting and held_above:
            quality = self._quality(True, vol_ok, held_above, bounce_long, rsi, "long")
            if quality < 5:
                return None
            return {
                "stage": 2, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"PDH Break+Retest ↑ | PDH={pdh:.5f} | "
                    f"Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # SHORT: PDL broken, now retesting from below
        broke_below  = prev_close < pdl
        retesting_s  = abs(price - pdl) <= zone
        held_below   = price < pdl
        bounce_short = is_shooting_star(row) or is_bearish_engulfing(entry_df, -2)

        if broke_below and retesting_s and held_below:
            quality = self._quality(True, vol_ok, held_below, bounce_short, rsi, "short")
            if quality < 5:
                return None
            return {
                "stage": 2, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    price + sl_dist,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"PDL Break+Retest ↓ | PDL={pdl:.5f} | "
                    f"Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
