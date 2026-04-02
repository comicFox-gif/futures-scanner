"""
Strategy — Trendline Break & Bounce
--------------------------------------
Trendlines connect a series of swing highs (downtrend) or swing lows (uptrend).
They represent dynamic support/resistance that the market respects repeatedly.

Two signal types:
  BOUNCE   — price pulls back to a valid trendline and bounces off it
             (trend-continuation signal)
  BREAKOUT — price breaks cleanly through a trendline
             (momentum / reversal signal — e.g. break above downtrend line)

Timeframes:
  1H  → trendline detection (connect swing highs/lows)
  15m → entry confirmation (close relative to trendline)

Trendline validity rules:
  - At least 2 confirmed swing touch points
  - Ascending (for uptrend) or descending (for downtrend)
  - Extrapolated value is within 3 ATR of current price (still relevant)
  - Not too steep (slope < 2 ATR per candle — avoids parabolic lines)

Stage 1 WARNING:
  Price within 1.0 ATR of a valid trendline

Stage 2 CONFIRMED:
  Bounce: price touched line (within 0.5 ATR) and close moved away
  Break:  candle closed on the other side of the line by at least 0.3 ATR

Quality score (1–5 stars):
  + Number of trendline touches
  + Body ratio of confirmation candle
  + RSI position
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    find_swing_highs_idx,
    find_swing_lows_idx,
    candle_body_ratio,
    is_hammer,
    is_shooting_star,
    is_bullish_engulfing,
    is_bearish_engulfing,
)

logger = logging.getLogger("futures_bot.trendline")


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Fit a line through 2+ points. Returns (slope, intercept) or None."""
    if len(points) < 2:
        return None
    p1_idx, p1_price = points[-2]
    p2_idx, p2_price = points[-1]
    if p2_idx == p1_idx:
        return None
    slope     = (p2_price - p1_price) / (p2_idx - p1_idx)
    intercept = p1_price - slope * p1_idx
    return slope, intercept


def _line_value(line: tuple[float, float], idx: int) -> float:
    """Price at given candle index along the trendline."""
    slope, intercept = line
    return slope * idx + intercept


def _count_touches(
    df: pd.DataFrame,
    line: tuple[float, float],
    start_idx: int,
    atr: float,
    direction: str,   # "up" or "down"
    touch_mult: float = 0.5,
) -> int:
    """Count how many candles touched the trendline within touch_mult * ATR."""
    touches = 0
    for i in range(start_idx, len(df) - 1):
        lv = _line_value(line, i)
        row = df.iloc[i]
        if direction == "up" and abs(float(row["low"]) - lv) <= atr * touch_mult:
            touches += 1
        elif direction == "down" and abs(float(row["high"]) - lv) <= atr * touch_mult:
            touches += 1
    return touches


class TrendlineBreakStrategy:
    NAME = "Trendline"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        tl  = cfg.get("trendline", {})

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

        self.swing_left  = tl.get("swing_left", 5)
        self.swing_right = tl.get("swing_right", 3)
        self.warn_mult   = tl.get("warn_atr_mult", 1.0)
        self.touch_mult  = tl.get("touch_atr_mult", 0.5)
        self.break_mult  = tl.get("break_atr_mult", 0.3)
        self.max_slope   = tl.get("max_slope_atr_per_candle", 1.5)

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

    def _quality(self, touches: int, body: float, rsi: float) -> int:
        score = 1
        if touches >= 4:
            score += 1
        elif touches >= 3:
            score += 0.5
        if body >= 0.60:
            score += 1
        elif body >= 0.40:
            score += 0.5
        if 40 <= rsi <= 60:
            score += 1
        elif 35 <= rsi <= 65:
            score += 0.5
        return min(5, round(score))

    def generate_signal(
        self,
        symbol: str,
        itf_df: pd.DataFrame,   # 1H — trendline detection
        entry_df: pd.DataFrame, # 15m — entry confirmation
    ) -> dict | None:
        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])
        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        current_idx = len(itf_df) - 2   # index of last confirmed candle
        body        = candle_body_ratio(entry_df)

        # ── Uptrend line (connects ascending swing lows) ──────────────────
        swing_lows = find_swing_lows_idx(itf_df, self.swing_left, self.swing_right)
        if len(swing_lows) >= 2:
            # Find ascending sequence
            ascending = [p for i, p in enumerate(swing_lows) if
                         i == 0 or p[1] > swing_lows[i - 1][1]]
            if len(ascending) >= 2:
                line = _fit_line(ascending[-3:])  # up to 3 most recent
                if line:
                    slope, _ = line
                    lv_now   = _line_value(line, current_idx)
                    dist     = price - lv_now
                    slope_ok = abs(slope) < atr * self.max_slope

                    if slope > 0 and slope_ok and abs(lv_now - price) < atr * 3:
                        touches = _count_touches(itf_df, line, ascending[0][0],
                                                 atr, "up", self.touch_mult)

                        # Stage 2 BOUNCE: price touched line and closed above it
                        if dist >= 0 and dist < atr * self.touch_mult:
                            rsi_ok  = self.rsi_long_min <= rsi <= self.rsi_long_max
                            pattern = is_hammer(row) or is_bullish_engulfing(entry_df, -2)
                            if rsi_ok and pattern:
                                quality = self._quality(touches, body, rsi)
                                sl_dist = atr * self.atr_sl_mult
                                return {
                                    "stage": 2, "direction": "long", "symbol": symbol,
                                    "entry": price,
                                    "sl":    lv_now - atr * 0.5,
                                    "tp1":   price + sl_dist * self.tp1_rr,
                                    "tp2":   price + sl_dist * self.tp2_rr,
                                    "tp3":   price + sl_dist * self.tp3_rr,
                                    "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                                    "reason": (
                                        f"Uptrend bounce @ {lv_now:.4f} | "
                                        f"{touches} touches | RSI {rsi:.1f}"
                                    ),
                                }

                        # Stage 1 WARNING: price approaching uptrend line
                        elif 0 <= dist < atr * self.warn_mult:
                            sl_dist = atr * self.atr_sl_mult
                            return {
                                "stage": 1, "direction": "long", "symbol": symbol,
                                "entry": price,
                                "sl":    lv_now - atr * 0.5,
                                "tp1":   price + sl_dist * self.tp1_rr,
                                "tp2":   price + sl_dist * self.tp2_rr,
                                "tp3":   price + sl_dist * self.tp3_rr,
                                "rsi": rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                                "reason": (
                                    f"Approaching uptrend line @ {lv_now:.4f} | "
                                    f"{touches} touches | Watch for bounce"
                                ),
                            }

        # ── Downtrend line (connects descending swing highs) ──────────────
        swing_highs = find_swing_highs_idx(itf_df, self.swing_left, self.swing_right)
        if len(swing_highs) >= 2:
            descending = [p for i, p in enumerate(swing_highs) if
                          i == 0 or p[1] < swing_highs[i - 1][1]]
            if len(descending) >= 2:
                line = _fit_line(descending[-3:])
                if line:
                    slope, _ = line
                    lv_now   = _line_value(line, current_idx)
                    dist     = lv_now - price
                    slope_ok = abs(slope) < atr * self.max_slope

                    if slope < 0 and slope_ok and abs(lv_now - price) < atr * 3:
                        touches = _count_touches(itf_df, line, descending[0][0],
                                                 atr, "down", self.touch_mult)

                        # Stage 2 BREAKOUT: price closes above the downtrend line
                        if price > lv_now + atr * self.break_mult:
                            rsi_ok = self.rsi_long_min <= rsi <= self.rsi_long_max
                            if rsi_ok and body >= 0.35:
                                quality = self._quality(touches, body, rsi)
                                sl_dist = atr * self.atr_sl_mult
                                return {
                                    "stage": 2, "direction": "long", "symbol": symbol,
                                    "entry": price,
                                    "sl":    lv_now - atr * 0.3,
                                    "tp1":   price + sl_dist * self.tp1_rr,
                                    "tp2":   price + sl_dist * self.tp2_rr,
                                    "tp3":   price + sl_dist * self.tp3_rr,
                                    "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                                    "reason": (
                                        f"Downtrend break @ {lv_now:.4f} | "
                                        f"{touches} touches | RSI {rsi:.1f}"
                                    ),
                                }

                        # Stage 1 WARNING: price pushing against downtrend line
                        elif dist >= 0 and dist < atr * self.warn_mult:
                            sl_dist = atr * self.atr_sl_mult
                            return {
                                "stage": 1, "direction": "long", "symbol": symbol,
                                "entry": price,
                                "sl":    lv_now - atr * 0.3,
                                "tp1":   price + sl_dist * self.tp1_rr,
                                "tp2":   price + sl_dist * self.tp2_rr,
                                "tp3":   price + sl_dist * self.tp3_rr,
                                "rsi": rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                                "reason": (
                                    f"Testing downtrend resistance @ {lv_now:.4f} | "
                                    f"{touches} touches | Watch for break"
                                ),
                            }

                        # Stage 2 BOUNCE at downtrend line (short)
                        elif dist >= 0 and dist < atr * self.touch_mult:
                            rsi_ok  = self.rsi_short_min <= rsi <= self.rsi_short_max
                            pattern = is_shooting_star(row) or is_bearish_engulfing(entry_df, -2)
                            if rsi_ok and pattern:
                                quality = self._quality(touches, body, rsi)
                                sl_dist = atr * self.atr_sl_mult
                                return {
                                    "stage": 2, "direction": "short", "symbol": symbol,
                                    "entry": price,
                                    "sl":    lv_now + atr * 0.5,
                                    "tp1":   price - sl_dist * self.tp1_rr,
                                    "tp2":   price - sl_dist * self.tp2_rr,
                                    "tp3":   price - sl_dist * self.tp3_rr,
                                    "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                                    "reason": (
                                        f"Downtrend bounce @ {lv_now:.4f} | "
                                        f"{touches} touches | RSI {rsi:.1f}"
                                    ),
                                }

        return None
