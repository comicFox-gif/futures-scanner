"""
Technical indicators using pandas-ta.
All functions return the indicator series or a dict of series.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def compute_emas(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Add EMA columns for each period to df."""
    for p in periods:
        df[f"ema_{p}"] = ta.ema(df["close"], length=p)
    return df


def compute_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Add MACD columns: macd, macd_signal, macd_hist."""
    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    df["macd"] = macd[f"MACD_{fast}_{slow}_{signal}"]
    df["macd_signal"] = macd[f"MACDs_{fast}_{slow}_{signal}"]
    df["macd_hist"] = macd[f"MACDh_{fast}_{slow}_{signal}"]
    return df


def compute_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add RSI column."""
    df["rsi"] = ta.rsi(df["close"], length=period)
    return df


def compute_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add ATR column."""
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    return df


def compute_volume_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add volume SMA column."""
    df["volume_sma"] = ta.sma(df["volume"], length=period)
    return df


def compute_all_indicators(
    df: pd.DataFrame,
    ema_fast: int,
    ema_mid: int,
    ema_slow: int,
    ema_trend: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    rsi_period: int,
    atr_period: int,
    volume_sma_period: int,
) -> pd.DataFrame:
    """Compute all indicators and return enriched DataFrame."""
    df = compute_emas(df, [ema_fast, ema_mid, ema_slow, ema_trend])
    df = compute_macd(df, macd_fast, macd_slow, macd_signal)
    df = compute_rsi(df, rsi_period)
    df = compute_atr(df, atr_period)
    df = compute_volume_sma(df, volume_sma_period)
    df.dropna(inplace=True)
    return df


def is_macd_bullish_cross(df: pd.DataFrame) -> bool:
    """
    True if MACD line crossed above signal line on the last closed candle.
    We look at candle[-2] (last confirmed close) vs candle[-3] (previous).
    """
    if len(df) < 3:
        return False
    prev = df.iloc[-3]
    last = df.iloc[-2]
    return (prev["macd"] < prev["macd_signal"]) and (last["macd"] > last["macd_signal"])


def is_macd_bearish_cross(df: pd.DataFrame) -> bool:
    """True if MACD line crossed below signal line on the last closed candle."""
    if len(df) < 3:
        return False
    prev = df.iloc[-3]
    last = df.iloc[-2]
    return (prev["macd"] > prev["macd_signal"]) and (last["macd"] < last["macd_signal"])


def macd_histogram_turning_positive(df: pd.DataFrame) -> bool:
    """True if MACD histogram turned from negative to positive (extra momentum check)."""
    if len(df) < 3:
        return False
    return df.iloc[-3]["macd_hist"] < 0 and df.iloc[-2]["macd_hist"] > 0


def macd_histogram_turning_negative(df: pd.DataFrame) -> bool:
    """True if MACD histogram turned from positive to negative."""
    if len(df) < 3:
        return False
    return df.iloc[-3]["macd_hist"] > 0 and df.iloc[-2]["macd_hist"] < 0


def find_swing_levels(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Find recent significant swing high and swing low for S/R reference.
    Returns dict with 'support' and 'resistance' price levels.
    """
    recent = df.iloc[-lookback - 2 : -1]
    support = recent["low"].min()
    resistance = recent["high"].max()
    return {"support": support, "resistance": resistance}


# ===========================================================================
# Trend Inception Detectors
# ===========================================================================

def ema_recently_crossed_bullish(df: pd.DataFrame, fast_col: str, slow_col: str, max_lookback: int = 8) -> tuple[bool, int]:
    """
    Detects if fast EMA crossed above slow EMA within the last `max_lookback` candles.
    Returns (True, candles_ago) or (False, -1).
    We look at confirmed closed candles only (skip iloc[-1] which is live).
    """
    for i in range(2, max_lookback + 2):
        curr = df.iloc[-i]
        prev = df.iloc[-i - 1]
        if curr[fast_col] > curr[slow_col] and prev[fast_col] <= prev[slow_col]:
            return True, i - 2   # 0 = just crossed on last confirmed candle
    return False, -1


def ema_recently_crossed_bearish(df: pd.DataFrame, fast_col: str, slow_col: str, max_lookback: int = 8) -> tuple[bool, int]:
    """
    Detects if fast EMA crossed below slow EMA within the last `max_lookback` candles.
    """
    for i in range(2, max_lookback + 2):
        curr = df.iloc[-i]
        prev = df.iloc[-i - 1]
        if curr[fast_col] < curr[slow_col] and prev[fast_col] >= prev[slow_col]:
            return True, i - 2
    return False, -1


def ema_cross_imminent_bullish(df: pd.DataFrame, fast_col: str, slow_col: str, proximity_pct: float = 0.002) -> bool:
    """
    True if fast EMA is below slow EMA but within `proximity_pct` distance of crossing.
    E.g. 0.002 = within 0.2% — cross is imminent, get ready.
    """
    row = df.iloc[-2]
    fast = row[fast_col]
    slow = row[slow_col]
    if fast >= slow:
        return False   # already crossed
    gap_pct = (slow - fast) / slow
    return gap_pct <= proximity_pct


def ema_cross_imminent_bearish(df: pd.DataFrame, fast_col: str, slow_col: str, proximity_pct: float = 0.002) -> bool:
    """
    True if fast EMA is above slow EMA but within `proximity_pct` of crossing down.
    """
    row = df.iloc[-2]
    fast = row[fast_col]
    slow = row[slow_col]
    if fast <= slow:
        return False
    gap_pct = (fast - slow) / slow
    return gap_pct <= proximity_pct


def price_near_ema(df: pd.DataFrame, ema_col: str, atr_tolerance: float = 0.6) -> bool:
    """
    True if price (low for longs, high for shorts) came within `atr_tolerance` × ATR of the EMA.
    This detects a pullback touch of the EMA.
    """
    row = df.iloc[-2]
    distance = abs(row["close"] - row[ema_col])
    return distance <= row["atr"] * atr_tolerance


def price_bouncing_bullish(df: pd.DataFrame, ema_col: str) -> bool:
    """
    Price touched the EMA zone (low <= EMA) and closed above it = bullish bounce.
    Checks last 2 confirmed candles for the touch+close pattern.
    """
    for idx in [-2, -3]:
        row = df.iloc[idx]
        if row["low"] <= row[ema_col] * 1.002 and row["close"] > row[ema_col]:
            return True
    return False


def price_bouncing_bearish(df: pd.DataFrame, ema_col: str) -> bool:
    """
    Price touched the EMA zone (high >= EMA) and closed below it = bearish bounce.
    """
    for idx in [-2, -3]:
        row = df.iloc[idx]
        if row["high"] >= row[ema_col] * 0.998 and row["close"] < row[ema_col]:
            return True
    return False


# ===========================================================================
# Fake Breakout Filters
# ===========================================================================

def candle_body_ratio(df: pd.DataFrame, idx: int = -2) -> float:
    """
    Body size as a fraction of total candle range (0–1).
    High value = strong decisive candle.
    Low value = wicky/indecision candle → likely fake move.
    """
    row = df.iloc[idx]
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return 0.0
    return abs(row["close"] - row["open"]) / candle_range


def upper_wick_ratio(df: pd.DataFrame, idx: int = -2) -> float:
    """
    Upper wick as fraction of total range.
    High value on a bullish candle = price was rejected at highs → bearish sign.
    """
    row = df.iloc[idx]
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return 0.0
    return (row["high"] - max(row["open"], row["close"])) / candle_range


def lower_wick_ratio(df: pd.DataFrame, idx: int = -2) -> float:
    """
    Lower wick as fraction of total range.
    High value on a bearish candle = price was rejected at lows → bullish sign.
    """
    row = df.iloc[idx]
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return 0.0
    return (min(row["open"], row["close"]) - row["low"]) / candle_range


def consecutive_bullish_closes(df: pd.DataFrame, count: int = 2) -> bool:
    """
    Last `count` confirmed closed candles all closed bullish (close > open).
    Filters single-candle fakeouts — real moves show follow-through.
    """
    if len(df) < count + 2:
        return False
    for i in range(-2, -2 - count, -1):
        row = df.iloc[i]
        if row["close"] <= row["open"]:
            return False
    return True


def consecutive_bearish_closes(df: pd.DataFrame, count: int = 2) -> bool:
    """
    Last `count` confirmed closed candles all closed bearish (close < open).
    """
    if len(df) < count + 2:
        return False
    for i in range(-2, -2 - count, -1):
        row = df.iloc[i]
        if row["close"] >= row["open"]:
            return False
    return True


def macd_histogram_strong(df: pd.DataFrame, atr_mult: float = 0.08) -> bool:
    """
    MACD histogram magnitude is meaningful relative to ATR.
    Filters barely-crossing MACD signals that have no real momentum.
    """
    row = df.iloc[-2]
    return abs(row["macd_hist"]) >= row["atr"] * atr_mult


def volume_building(df: pd.DataFrame, lookback: int = 2) -> bool:
    """
    Volume has been consistently above average for last `lookback` candles.
    A single volume spike can be a stop hunt. Sustained volume = real interest.
    """
    if len(df) < lookback + 2:
        return False
    recent = df.iloc[-2 - lookback : -1]
    vol_sma = recent["volume_sma"].iloc[-1]
    if vol_sma == 0:
        return False
    return all(recent["volume"] > vol_sma * 0.85)
