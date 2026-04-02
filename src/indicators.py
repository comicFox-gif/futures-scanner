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
# S/R Level Detection
# ===========================================================================

def find_key_levels(df: pd.DataFrame, n_left: int = 5, n_right: int = 2,
                    cluster_pct: float = 0.003, max_levels: int = 6) -> list[dict]:
    """
    Detect key S/R levels from swing highs/lows.

    n_left  : candles to the left that must be lower/higher
    n_right : candles to the right for confirmation (small = catches recent levels)
    cluster_pct : group levels within this % of each other into one zone
    max_levels  : max levels to return (strongest only)

    Returns list of dicts sorted by strength:
      {price, touches, type: 'support'|'resistance', strength: 1-5}
    """
    levels: list[dict] = []
    # Use confirmed candles only (exclude last candle which may be live)
    data = df.iloc[:-1]
    n = len(data)

    for i in range(n_left, n - n_right):
        row = data.iloc[i]

        # Swing high → resistance
        left_highs  = data.iloc[i - n_left : i]["high"]
        right_highs = data.iloc[i + 1 : i + n_right + 1]["high"]
        if row["high"] >= left_highs.max() and row["high"] >= right_highs.max():
            levels.append({"price": row["high"], "type": "resistance", "touches": 1})

        # Swing low → support
        left_lows  = data.iloc[i - n_left : i]["low"]
        right_lows = data.iloc[i + 1 : i + n_right + 1]["low"]
        if row["low"] <= left_lows.min() and row["low"] <= right_lows.min():
            levels.append({"price": row["low"], "type": "support", "touches": 1})

    if not levels:
        return []

    # Cluster nearby levels (within cluster_pct of each other)
    levels.sort(key=lambda x: x["price"])
    clustered: list[dict] = []
    for lv in levels:
        merged = False
        for cl in clustered:
            if abs(lv["price"] - cl["price"]) / cl["price"] <= cluster_pct and lv["type"] == cl["type"]:
                # Merge: average price, add touch
                cl["price"] = (cl["price"] * cl["touches"] + lv["price"]) / (cl["touches"] + 1)
                cl["touches"] += 1
                merged = True
                break
        if not merged:
            clustered.append(dict(lv))

    # Assign strength score 1-5 based on touch count
    for cl in clustered:
        t = cl["touches"]
        cl["strength"] = 5 if t >= 5 else 4 if t >= 4 else 3 if t >= 3 else 2 if t >= 2 else 1

    # Sort by strength desc, return top N
    clustered.sort(key=lambda x: x["strength"], reverse=True)
    return clustered[:max_levels]


def price_near_level(current_price: float, level_price: float, atr: float,
                     tolerance_mult: float = 0.5) -> bool:
    """True if price is within tolerance_mult * ATR of the level."""
    return abs(current_price - level_price) <= atr * tolerance_mult


def price_approaching_level(current_price: float, level_price: float, atr: float,
                             approach_mult: float = 1.5) -> bool:
    """True if price is within approach_mult * ATR of the level (wider zone for warning)."""
    return abs(current_price - level_price) <= atr * approach_mult


# ===========================================================================
# Bounce Pattern Detection
# ===========================================================================

def is_hammer(row: pd.Series) -> bool:
    """
    Bullish hammer at support:
    - Lower wick >= 2x body
    - Upper wick <= 0.5x body
    - Body is at least 10% of range (not a doji)
    """
    body       = abs(row["close"] - row["open"])
    lo_wick    = min(row["open"], row["close"]) - row["low"]
    up_wick    = row["high"] - max(row["open"], row["close"])
    total_rng  = row["high"] - row["low"]
    if total_rng == 0:
        return False
    return (lo_wick >= body * 2 and up_wick <= body * 0.5 and body / total_rng >= 0.10)


def is_shooting_star(row: pd.Series) -> bool:
    """
    Bearish shooting star at resistance:
    - Upper wick >= 2x body
    - Lower wick <= 0.5x body
    - Body is at least 10% of range
    """
    body      = abs(row["close"] - row["open"])
    lo_wick   = min(row["open"], row["close"]) - row["low"]
    up_wick   = row["high"] - max(row["open"], row["close"])
    total_rng = row["high"] - row["low"]
    if total_rng == 0:
        return False
    return (up_wick >= body * 2 and lo_wick <= body * 0.5 and body / total_rng >= 0.10)


def is_bullish_engulfing(df: pd.DataFrame, idx: int = -2) -> bool:
    """
    Current candle (bullish) fully engulfs previous bearish candle.
    Strong reversal signal at support.
    """
    curr = df.iloc[idx]
    prev = df.iloc[idx - 1]
    return (
        curr["close"] > curr["open"]   # current bullish
        and prev["close"] < prev["open"]  # previous bearish
        and curr["open"] <= prev["close"]
        and curr["close"] >= prev["open"]
    )


def is_bearish_engulfing(df: pd.DataFrame, idx: int = -2) -> bool:
    """
    Current candle (bearish) fully engulfs previous bullish candle.
    Strong reversal signal at resistance.
    """
    curr = df.iloc[idx]
    prev = df.iloc[idx - 1]
    return (
        curr["close"] < curr["open"]
        and prev["close"] > prev["open"]
        and curr["open"] >= prev["close"]
        and curr["close"] <= prev["open"]
    )


# ===========================================================================
# Trend Inception Detectors
# ===========================================================================

def ema_recently_crossed_bullish(df: pd.DataFrame, fast_col: str, slow_col: str, max_lookback: int = 8) -> tuple[bool, int]:
    """
    Detects if fast EMA crossed above slow EMA within the last `max_lookback` candles.
    Returns (True, candles_ago) or (False, -1).
    We look at confirmed closed candles only (skip iloc[-1] which is live).
    """
    max_i = min(max_lookback + 2, len(df) - 1)
    for i in range(2, max_i):
        curr = df.iloc[-i]
        prev = df.iloc[-i - 1]
        if curr[fast_col] > curr[slow_col] and prev[fast_col] <= prev[slow_col]:
            return True, i - 2   # 0 = just crossed on last confirmed candle
    return False, -1


def ema_recently_crossed_bearish(df: pd.DataFrame, fast_col: str, slow_col: str, max_lookback: int = 8) -> tuple[bool, int]:
    """
    Detects if fast EMA crossed below slow EMA within the last `max_lookback` candles.
    """
    max_i = min(max_lookback + 2, len(df) - 1)
    for i in range(2, max_i):
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


# ===========================================================================
# Order Block Detection (ICT)
# ===========================================================================

def find_order_blocks(
    df: pd.DataFrame,
    lookback: int = 80,
    impulse_candles: int = 3,
    impulse_atr_mult: float = 2.0,
) -> list[dict]:
    """
    Detect ICT-style order blocks in the last `lookback` candles.

    Bullish OB: last bearish candle before a strong bullish impulse.
                The candle's range becomes a demand zone — price returning = long.
    Bearish OB: last bullish candle before a strong bearish impulse.
                The candle's range becomes a supply zone — price returning = short.

    Validity: an OB is invalidated if price closes through its opposite boundary
    after formation. Only un-mitigated (fresh) OBs are returned.

    Returns list of dicts: {type, high, low, mid, formed_idx}
    """
    if len(df) < lookback + impulse_candles + 5:
        return []

    atr = float(df["atr"].iloc[-2])
    if pd.isna(atr) or atr == 0:
        return []

    current_price = float(df.iloc[-2]["close"])
    obs: list[dict] = []

    start = max(0, len(df) - lookback - impulse_candles)
    end   = len(df) - impulse_candles - 2

    for i in range(start, end):
        candle  = df.iloc[i]
        body    = abs(float(candle["close"]) - float(candle["open"]))
        if body < atr * 0.1:          # skip doji candles
            continue

        impulse = df.iloc[i + 1 : i + impulse_candles + 1]
        subsequent = df.iloc[i + 1 :]

        # ── Bullish OB: bearish candle before strong up-move ─────────────
        if candle["close"] < candle["open"]:
            if impulse["high"].max() - float(candle["low"]) < impulse_atr_mult * atr:
                continue
            ob_high = float(candle["high"])
            ob_low  = float(candle["low"])
            # Invalidated if any close went below OB low
            if (subsequent["close"] < ob_low).any():
                continue
            if current_price >= ob_low * 0.998:
                obs.append({"type": "bullish", "high": ob_high, "low": ob_low,
                            "mid": (ob_high + ob_low) / 2, "formed_idx": i})

        # ── Bearish OB: bullish candle before strong down-move ───────────
        elif candle["close"] > candle["open"]:
            if float(candle["high"]) - impulse["low"].min() < impulse_atr_mult * atr:
                continue
            ob_high = float(candle["high"])
            ob_low  = float(candle["low"])
            # Invalidated if any close went above OB high
            if (subsequent["close"] > ob_high).any():
                continue
            if current_price <= ob_high * 1.002:
                obs.append({"type": "bearish", "high": ob_high, "low": ob_low,
                            "mid": (ob_high + ob_low) / 2, "formed_idx": i})

    # Return only the 2 most recent valid OBs of each type
    bullish = [o for o in obs if o["type"] == "bullish"][-2:]
    bearish = [o for o in obs if o["type"] == "bearish"][-2:]
    return bullish + bearish


# ===========================================================================
# Swing High / Low Index Finders (for trendlines + divergence)
# ===========================================================================

def find_swing_highs_idx(df: pd.DataFrame, left: int = 5, right: int = 3) -> list[tuple[int, float]]:
    """
    Return list of (iloc_index, price) for swing highs.
    A swing high is strictly higher than `left` candles left and `right` candles right.
    Operates on confirmed candles only (excludes last row).
    """
    result = []
    data = df.iloc[:-1]
    n    = len(data)
    for i in range(left, n - right):
        h = float(data.iloc[i]["high"])
        if (data.iloc[i - left : i]["high"] < h).all() and (data.iloc[i + 1 : i + right + 1]["high"] < h).all():
            result.append((i, h))
    return result


def find_swing_lows_idx(df: pd.DataFrame, left: int = 5, right: int = 3) -> list[tuple[int, float]]:
    """
    Return list of (iloc_index, price) for swing lows.
    A swing low is strictly lower than `left` candles left and `right` candles right.
    """
    result = []
    data = df.iloc[:-1]
    n    = len(data)
    for i in range(left, n - right):
        l = float(data.iloc[i]["low"])
        if (data.iloc[i - left : i]["low"] > l).all() and (data.iloc[i + 1 : i + right + 1]["low"] > l).all():
            result.append((i, l))
    return result


# ===========================================================================
# RSI Divergence Detection
# ===========================================================================

def rsi_bullish_divergence(
    df: pd.DataFrame,
    lookback: int = 50,
    min_rsi_diff: float = 3.0,
) -> tuple[bool, float, float, str]:
    """
    Regular bullish divergence: price makes a lower low, RSI makes a higher low.
    This signals that selling momentum is weakening — reversal likely.

    Returns (found, current_rsi, prior_rsi, reason_string).
    """
    if "rsi" not in df.columns or len(df) < lookback + 5:
        return False, 0.0, 0.0, ""

    curr      = df.iloc[-2]
    curr_low  = float(curr["low"])
    curr_rsi  = float(curr["rsi"])
    if pd.isna(curr_rsi):
        return False, 0.0, 0.0, ""

    window = df.iloc[-lookback : -2]
    if len(window) < 10:
        return False, 0.0, 0.0, ""

    prior_idx  = window["low"].idxmin()
    prior_row  = df.loc[prior_idx]
    prior_low  = float(prior_row["low"])
    prior_rsi  = float(prior_row["rsi"])
    if pd.isna(prior_rsi):
        return False, 0.0, 0.0, ""

    # Price lower low + RSI higher low = divergence
    if curr_low < prior_low and curr_rsi > prior_rsi + min_rsi_diff and curr_rsi < 60:
        reason = (
            f"Bull divergence | Price LL {curr_low:.5f} < {prior_low:.5f} | "
            f"RSI HL {curr_rsi:.1f} > {prior_rsi:.1f}"
        )
        return True, curr_rsi, prior_rsi, reason

    return False, 0.0, 0.0, ""


def rsi_bearish_divergence(
    df: pd.DataFrame,
    lookback: int = 50,
    min_rsi_diff: float = 3.0,
) -> tuple[bool, float, float, str]:
    """
    Regular bearish divergence: price makes a higher high, RSI makes a lower high.
    Selling momentum appears while price is still rising — reversal likely.

    Returns (found, current_rsi, prior_rsi, reason_string).
    """
    if "rsi" not in df.columns or len(df) < lookback + 5:
        return False, 0.0, 0.0, ""

    curr      = df.iloc[-2]
    curr_high = float(curr["high"])
    curr_rsi  = float(curr["rsi"])
    if pd.isna(curr_rsi):
        return False, 0.0, 0.0, ""

    window = df.iloc[-lookback : -2]
    if len(window) < 10:
        return False, 0.0, 0.0, ""

    prior_idx  = window["high"].idxmax()
    prior_row  = df.loc[prior_idx]
    prior_high = float(prior_row["high"])
    prior_rsi  = float(prior_row["rsi"])
    if pd.isna(prior_rsi):
        return False, 0.0, 0.0, ""

    # Price higher high + RSI lower high = divergence
    if curr_high > prior_high and curr_rsi < prior_rsi - min_rsi_diff and curr_rsi > 40:
        reason = (
            f"Bear divergence | Price HH {curr_high:.5f} > {prior_high:.5f} | "
            f"RSI LH {curr_rsi:.1f} < {prior_rsi:.1f}"
        )
        return True, curr_rsi, prior_rsi, reason

    return False, 0.0, 0.0, ""
