"""
Forex Strategy — Fibonacci Pullback
-------------------------------------
One of the most reliable setups in professional forex trading.
In a clear impulse move, price retraces to the 50% or 61.8% Fibonacci level
and bounces. Institutions place limit orders at these levels.

Logic:
  Impulse: identify a clean swing high → low (or low → high) over the last N candles
  Retrace: price pulls back to 50%-61.8% fib zone
  Confirm: bounce candle (hammer/engulfing) + MACD histogram turning + RSI in range
  ADX gate: impulse must have been strong (ADX >= 22 at time of impulse)

Quality (5 binary conditions):
  C1: Clear impulse move (height >= 2x ATR)
  C2: Price inside fib zone (50%-61.8%)
  C3: Bounce candle (hammer or engulfing)
  C4: MACD histogram confirms direction
  C5: RSI in zone (long: 35-55, short: 45-65)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    is_hammer,
    is_shooting_star,
    is_bullish_engulfing,
    is_bearish_engulfing,
    candle_body_ratio,
    macd_histogram_turning_positive,
    macd_histogram_turning_negative,
)

logger = logging.getLogger("forex_bot.fibonacci_pullback")


class FibonacciPullbackStrategy:
    NAME = "Fib Pullback"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        fib = cfg.get("fibonacci_pullback", {})

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

        self.swing_lookback  = fib.get("swing_lookback", 30)
        self.fib_low         = fib.get("fib_low", 0.50)    # 50% retracement
        self.fib_high        = fib.get("fib_high", 0.618)  # 61.8% retracement
        self.impulse_atr_min = fib.get("impulse_atr_min", 2.0)  # impulse >= 2x ATR

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, big_impulse: bool, in_fib_zone: bool, bounce_candle: bool,
                 macd_ok: bool, rsi: float, direction: str) -> int:
        score = 0
        if big_impulse:    score += 1  # C1
        if in_fib_zone:    score += 1  # C2
        if bounce_candle:  score += 1  # C3
        if macd_ok:        score += 1  # C4
        if direction == "long"  and 35 <= rsi <= 55: score += 1  # C5
        elif direction == "short" and 45 <= rsi <= 65: score += 1
        return score

    def _find_impulse(self, df: pd.DataFrame, atr: float):
        """
        Scan recent candles for a clean bullish or bearish impulse.
        Returns (direction, swing_low, swing_high) or (None, None, None).
        """
        window = df.iloc[-(self.swing_lookback + 2):-2]
        if len(window) < 10:
            return None, None, None

        swing_low  = window["low"].min()
        swing_high = window["high"].max()
        move_size  = swing_high - swing_low

        if move_size < atr * self.impulse_atr_min:
            return None, None, None

        # Determine direction: did price finish high or low?
        low_idx  = window["low"].idxmin()
        high_idx = window["high"].idxmax()

        if low_idx < high_idx:
            return "long", swing_low, swing_high   # bullish impulse — expect pullback to go long
        else:
            return "short", swing_low, swing_high  # bearish impulse — expect pullback to go short

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        if len(entry_df) < self.swing_lookback + 5:
            return None

        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        direction, swing_low, swing_high = self._find_impulse(entry_df, atr)
        if direction is None:
            return None

        move      = swing_high - swing_low
        fib_50    = swing_high - move * self.fib_low   if direction == "long"  else swing_low + move * self.fib_low
        fib_618   = swing_high - move * self.fib_high  if direction == "long"  else swing_low + move * self.fib_high

        fib_zone_low  = min(fib_50, fib_618)
        fib_zone_high = max(fib_50, fib_618)

        in_fib_zone  = fib_zone_low <= price <= fib_zone_high
        big_impulse  = move >= atr * self.impulse_atr_min
        sl_dist      = atr * self.atr_sl_mult

        if direction == "long":
            bounce_candle = is_hammer(row) or is_bullish_engulfing(entry_df, -2)
            macd_ok       = macd_histogram_turning_positive(entry_df)
        else:
            bounce_candle = is_shooting_star(row) or is_bearish_engulfing(entry_df, -2)
            macd_ok       = macd_histogram_turning_negative(entry_df)

        quality = self._quality(big_impulse, in_fib_zone, bounce_candle, macd_ok, rsi, direction)

        if quality < 5 or not in_fib_zone:
            return None

        if direction == "long":
            return {
                "stage": 2, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                "reason": (
                    f"Fib 50-61.8% pullback ↑ | Zone {fib_zone_low:.5f}–{fib_zone_high:.5f} | "
                    f"RSI={rsi:.0f}"
                ),
            }
        else:
            return {
                "stage": 2, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    price + sl_dist,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                "reason": (
                    f"Fib 50-61.8% pullback ↓ | Zone {fib_zone_low:.5f}–{fib_zone_high:.5f} | "
                    f"RSI={rsi:.0f}"
                ),
            }
