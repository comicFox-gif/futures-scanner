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
    if macd is None:
        df["macd"] = float("nan")
        df["macd_signal"] = float("nan")
        df["macd_hist"] = float("nan")
        return df
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


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX column (trend strength). adx >= 25 = trending."""
    adx = ta.adx(df["high"], df["low"], df["close"], length=period)
    if adx is None:
        df["adx"] = float("nan")
        return df
    # pandas_ta column name varies: ADX_14 — pick first column starting with ADX
    adx_col = next((c for c in adx.columns if c.startswith("ADX")), None)
    df["adx"] = adx[adx_col] if adx_col else float("nan")
    return df


def compute_bbands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Band columns: bb_upper, bb_lower, bb_mid, bb_width."""
    bb = ta.bbands(df["close"], length=period, std=std)
    if bb is None:
        for col in ("bb_upper", "bb_lower", "bb_mid", "bb_width"):
            df[col] = float("nan")
        return df
    # pandas_ta may output BBU_20_2.0 or BBU_20_2 depending on version — find dynamically
    def _find(prefix):
        return next((c for c in bb.columns if c.startswith(prefix)), None)
    u = _find("BBU_"); l = _find("BBL_"); m = _find("BBM_"); w = _find("BBB_")
    df["bb_upper"] = bb[u] if u else float("nan")
    df["bb_lower"] = bb[l] if l else float("nan")
    df["bb_mid"]   = bb[m] if m else float("nan")
    df["bb_width"]  = bb[w] if w else float("nan")
    return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily-anchored VWAP column.
    Requires DataFrame with a DatetimeIndex (set in ohlcv_to_df).
    Anchors reset at midnight UTC each day.
    """
    df = df.copy()
    df["_date"] = df.index.date
    df["_tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpvol"] = df["_tp"] * df["volume"]
    df["_cum_tpvol"] = df.groupby("_date")["_tpvol"].cumsum()
    df["_cum_vol"]   = df.groupby("_date")["volume"].cumsum()
    df["vwap"] = df["_cum_tpvol"] / df["_cum_vol"]
    df.drop(columns=["_date", "_tp", "_tpvol", "_cum_tpvol", "_cum_vol"], inplace=True)
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
    df = compute_adx(df)
    df = compute_bbands(df)
    df = compute_vwap(df)
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
# Stop Hunt / Liquidity Sweep Detection
# ===========================================================================

def detect_liquidity_sweep(
    df: pd.DataFrame,
    structure_lookback: int = 20,
    sweep_lookback: int = 5,
    wick_atr_min: float = 0.25,
) -> str | None:
    """
    Detect a recent stop hunt / liquidity sweep.

    Whales push price above a swing high to grab short stops (buy-side sweep),
    or below a swing low to grab long stops (sell-side sweep), then reverse.

    The wick must pierce the structure level by at least `wick_atr_min * ATR`
    and the candle must CLOSE BACK on the opposite side — confirming the sweep
    was a fake-out, not a real breakout.

    Returns:
      "buy_side"  — recent wick above structure high, closed below it
                    (whales hunted long/short stops above — bearish danger for longs)
      "sell_side" — recent wick below structure low, closed above it
                    (whales hunted long stops below — bullish danger for shorts)
      None        — no sweep in recent candles
    """
    min_len = structure_lookback + sweep_lookback + 3
    if len(df) < min_len:
        return None

    atr = float(df.iloc[-2].get("atr", float("nan")))
    if pd.isna(atr) or atr == 0:
        return None

    # Structure: the prior high/low pool where stops are resting
    struct_window = df.iloc[-(structure_lookback + sweep_lookback) : -sweep_lookback]
    struct_high   = float(struct_window["high"].max())
    struct_low    = float(struct_window["low"].min())
    min_pierce    = atr * wick_atr_min

    # Scan recent candles (confirmed closes only, exclude live candle)
    recent = df.iloc[-sweep_lookback - 1 : -1]
    for _, row in recent.iterrows():
        # Buy-side sweep: wick pierced above structure high but closed back below
        if row["high"] > struct_high + min_pierce and row["close"] < struct_high:
            return "buy_side"
        # Sell-side sweep: wick pierced below structure low but closed back above
        if row["low"] < struct_low - min_pierce and row["close"] > struct_low:
            return "sell_side"

    return None


def detect_bull_trap(
    df: pd.DataFrame,
    ema_col: str,
    pump_pct: float = 12.0,
    ema_dist_pct: float = 15.0,
    rsi_hot: float = 72.0,
    vol_decay: float = 0.6,
    min_votes: int = 2,
) -> bool:
    """
    Detect a bull trap before entering a long.

    Whales pump price hard → late buyers pile in → dump → SL hit.
    Fires when 2+ of these conditions are true:

      Vote 1 — Explosive pump: price rose 12%+ from the 5-candle low.
                Real breakouts rarely go 12% without pausing.
      Vote 2 — Too extended: price > 15% above EMA50.
                Extreme extension = rubber-band snap risk.
      Vote 3 — RSI overheated: RSI > 72 on the entry candle.
                Overbought at entry = no margin left for buyers.
      Vote 4 — Volume dying: current candle volume < 60% of previous.
                Pump on shrinking volume = no follow-through.

    Returns True  → likely bull trap, block the long.
    Returns False → setup looks clean, proceed normally.
    """
    if len(df) < 7:
        return False

    row = df.iloc[-2]
    votes = 0

    # Vote 1: explosive pump in last 5 candles
    recent_low = float(df.iloc[-7:-2]["low"].min())
    current_close = float(row["close"])
    if recent_low > 0 and (current_close - recent_low) / recent_low * 100 > pump_pct:
        votes += 1

    # Vote 2: price too far above EMA50
    ema50 = float(row.get(ema_col, float("nan")))
    if not pd.isna(ema50) and ema50 > 0:
        dist = (current_close - ema50) / ema50 * 100
        if dist > ema_dist_pct:
            votes += 1

    # Vote 3: RSI overheated
    rsi = float(row.get("rsi", float("nan")))
    if not pd.isna(rsi) and rsi > rsi_hot:
        votes += 1

    # Vote 4: volume dying after the pump
    if len(df) >= 3:
        curr_vol = float(df.iloc[-2]["volume"])
        prev_vol = float(df.iloc[-3]["volume"])
        if prev_vol > 0 and curr_vol < prev_vol * vol_decay:
            votes += 1

    return votes >= min_votes


def bull_trap_short_confirmed(df: pd.DataFrame, min_wick_ratio: float = 0.20) -> bool:
    """
    Extra confirmation before fading a bull trap with a short.
    Requires the trap candle to show upper wick rejection — price pumped
    but got sold into, leaving a visible wick at the top.
    Without this, we might short a genuine breakout.

    True  → upper wick >= min_wick_ratio of range → rejection confirmed, safe to short
    False → no clear rejection wick → skip the fade
    """
    if len(df) < 3:
        return False
    row = df.iloc[-2]
    rng = row["high"] - row["low"]
    if rng == 0:
        return False
    upper_wick = row["high"] - max(row["open"], row["close"])
    return (upper_wick / rng) >= min_wick_ratio


def bounce_candle_clean(row: pd.Series, direction: str, max_wick_ratio: float = 0.38) -> bool:
    """
    Check that the bounce candle isn't a wick-heavy fake pump/dump.

    For longs  : upper wick must be <= max_wick_ratio of total range.
                 A big upper wick = price pumped and got rejected = stop hunt candle.
    For shorts : lower wick must be <= max_wick_ratio of total range.
                 A big lower wick = price dumped and got bought back = fake dump.
    """
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False
    if direction == "long":
        upper_wick = row["high"] - max(row["open"], row["close"])
        return (upper_wick / candle_range) <= max_wick_ratio
    else:
        lower_wick = min(row["open"], row["close"]) - row["low"]
        return (lower_wick / candle_range) <= max_wick_ratio


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


# ===========================================================================
# Institutional Order Flow Detection
# ===========================================================================

def approx_volume_delta(row: pd.Series) -> float:
    """
    Approximate buy vs sell volume from a single OHLCV candle.
    Without Level 2 order book data, we estimate using price position within the candle.

    Positive delta → buyers dominated that candle
    Negative delta → sellers dominated that candle
    """
    rng = row["high"] - row["low"]
    if rng == 0:
        return 0.0
    buy_vol  = ((row["close"] - row["low"])  / rng) * row["volume"]
    sell_vol = ((row["high"]  - row["close"]) / rng) * row["volume"]
    return buy_vol - sell_vol


def detect_whale_entry(
    df: pd.DataFrame,
    vol_mult: float = 2.5,
    body_min: float = 0.55,
    delta_mult: float = 1.8,
    lookback: int = 3,
) -> dict | None:
    """
    Detect the FIRST aggressive institutional buy candle — the moment whales enter.

    Whales leave a clear footprint:
      1. Volume spike: 2.5x+ above the 20-period SMA (big money moving in)
      2. Decisive body: candle body >= 55% of range (no indecision)
      3. Bullish candle: close > open
      4. Positive volume delta: buy pressure >= 1.8x sell pressure
      5. Fresh: this is the FIRST spike (2+ earlier spikes = already too late)

    We enter on the NEXT candle after the whale candle closes.

    Returns dict with vol_ratio, body_ratio, delta, candle_idx, reason — or None.
    """
    min_len = lookback + 5
    if len(df) < min_len:
        return None

    vol_sma = float(df.iloc[-2].get("volume_sma", float("nan")))
    if pd.isna(vol_sma) or vol_sma == 0:
        return None

    for i in range(2, lookback + 2):
        if abs(i) > len(df) - 1:
            break
        row = df.iloc[-i]

        vol_ratio = row["volume"] / vol_sma
        rng       = row["high"] - row["low"]
        body      = abs(row["close"] - row["open"]) / rng if rng > 0 else 0
        bullish   = row["close"] > row["open"]
        delta     = approx_volume_delta(row)
        sell_vol  = ((row["high"] - row["close"]) / rng * row["volume"]) if rng > 0 else 1
        delta_ratio = delta / sell_vol if sell_vol > 0 else 0

        if vol_ratio >= vol_mult and body >= body_min and bullish and delta_ratio >= delta_mult:
            # Confirm it's the FIRST spike — not the 3rd or 4th candle of an existing pump
            earlier_spikes = 0
            for j in range(i + 1, i + 4):
                if abs(j) > len(df) - 1:
                    break
                prev_row = df.iloc[-j]
                if prev_row["volume"] / vol_sma >= vol_mult * 0.8 and prev_row["close"] > prev_row["open"]:
                    earlier_spikes += 1
            if earlier_spikes >= 2:
                continue  # already pumping — we'd be entering late

            return {
                "vol_ratio":  vol_ratio,
                "body_ratio": body,
                "delta":      delta,
                "candle_idx": i,
                "reason": (
                    f"Whale Entry {i-1} candle(s) ago | "
                    f"Vol={vol_ratio:.1f}x | Body={body:.0%} | ΔVol={delta_ratio:.1f}x"
                ),
            }

    return None


def detect_distribution(
    df: pd.DataFrame,
    vol_decay_pct: float = 0.65,
    lookback: int = 3,
) -> dict | None:
    """
    Detect when institutions are exiting (distribution phase).

    Signs whales are leaving:
      1. Volume declining vs recent peak (buying exhaustion)
      2. Negative volume delta on high-volume candle (selling into the crowd)
      3. Candle closes in lower 40% of its range (hidden selling)
      4. Three consecutive declining volume candles

    2+ signals = distribution confirmed → time to exit or short.
    Returns dict with reason_str, or None if momentum still healthy.
    """
    if len(df) < lookback + 3:
        return None

    reasons = []
    row     = df.iloc[-2]
    rng     = row["high"] - row["low"]
    vol_sma = float(row.get("volume_sma", float("nan")))

    # 1. Volume decay vs recent peak
    recent   = df.iloc[-lookback - 1 : -1]
    peak_vol = float(recent["volume"].max())
    last_vol = float(row["volume"])
    if peak_vol > 0 and last_vol < peak_vol * vol_decay_pct:
        reasons.append(f"Volume decaying — {last_vol/peak_vol:.0%} of peak")

    # 2. High-volume negative delta
    delta = approx_volume_delta(row)
    if delta < 0 and not pd.isna(vol_sma) and vol_sma > 0 and row["volume"] / vol_sma >= 1.5:
        reasons.append("High-volume sell delta — institutions selling into crowd")

    # 3. Closed in lower half of range
    if rng > 0 and (row["close"] - row["low"]) / rng < 0.40 and row["close"] > row["open"]:
        reasons.append("Closed in lower 40% of range — hidden selling pressure")

    # 4. Three consecutive declining volume candles
    vols = [float(df.iloc[-i]["volume"]) for i in range(2, 5) if len(df) > i]
    if len(vols) == 3 and vols[0] < vols[1] < vols[2]:
        reasons.append("3 consecutive candles of declining volume")

    if len(reasons) >= 2:
        return {"reasons": reasons, "reason_str": " | ".join(reasons)}

    return None


def detect_whale_sell(
    df: pd.DataFrame,
    vol_mult: float = 2.5,
    body_min: float = 0.55,
    delta_mult: float = 1.8,
    lookback: int = 3,
) -> dict | None:
    """
    Mirror of detect_whale_entry — detects the FIRST aggressive institutional SELL candle.

    Whales selling aggressively leave the same footprint in reverse:
      1. Volume spike: 2.5x+ above 20-period SMA
      2. Decisive body: >= 55% of range
      3. Bearish candle: close < open
      4. Negative volume delta: sell pressure >= 1.8x buy pressure
      5. Fresh: first such spike (not the 3rd candle of an existing dump — too late)

    Entry fires on the NEXT candle after the institutional sell candle.
    """
    min_len = lookback + 5
    if len(df) < min_len:
        return None

    vol_sma = float(df.iloc[-2].get("volume_sma", float("nan")))
    if pd.isna(vol_sma) or vol_sma == 0:
        return None

    for i in range(2, lookback + 2):
        if abs(i) > len(df) - 1:
            break
        row = df.iloc[-i]

        vol_ratio = row["volume"] / vol_sma
        rng       = row["high"] - row["low"]
        body      = abs(row["close"] - row["open"]) / rng if rng > 0 else 0
        bearish   = row["close"] < row["open"]
        delta     = approx_volume_delta(row)   # negative = sell dominated
        buy_vol   = ((row["close"] - row["low"]) / rng * row["volume"]) if rng > 0 else 1
        # sell pressure ratio: how much more selling than buying
        sell_delta_ratio = abs(delta) / buy_vol if buy_vol > 0 else 0

        if vol_ratio >= vol_mult and body >= body_min and bearish and sell_delta_ratio >= delta_mult:
            # Confirm it's the FIRST sell spike — not already deep in a dump
            earlier_spikes = 0
            for j in range(i + 1, i + 4):
                if abs(j) > len(df) - 1:
                    break
                prev_row = df.iloc[-j]
                if prev_row["volume"] / vol_sma >= vol_mult * 0.8 and prev_row["close"] < prev_row["open"]:
                    earlier_spikes += 1
            if earlier_spikes >= 2:
                continue  # already deep in a dump — we'd be entering late

            return {
                "vol_ratio":  vol_ratio,
                "body_ratio": body,
                "delta":      delta,
                "candle_idx": i,
                "reason": (
                    f"Whale Sell {i-1} candle(s) ago | "
                    f"Vol={vol_ratio:.1f}x | Body={body:.0%} | ΔVol={sell_delta_ratio:.1f}x"
                ),
            }

    return None


def detect_short_covering(
    df: pd.DataFrame,
    vol_decay_pct: float = 0.65,
    lookback: int = 3,
) -> dict | None:
    """
    Detect when institutions are covering their shorts (exiting short positions).

    When whales close shorts they buy back — this is the mirror of distribution:
      1. Volume declining on red candles — selling exhaustion
      2. Positive volume delta on high-volume candle — buying (covering) into the dump
      3. Candle closes in upper 60% of range — hidden buying pressure
      4. Three consecutive declining volume candles

    2+ signals = short covering likely → warn subscribers to take profits on shorts.
    """
    if len(df) < lookback + 3:
        return None

    reasons = []
    row     = df.iloc[-2]
    rng     = row["high"] - row["low"]
    vol_sma = float(row.get("volume_sma", float("nan")))

    # 1. Volume decay vs recent peak
    recent   = df.iloc[-lookback - 1 : -1]
    peak_vol = float(recent["volume"].max())
    last_vol = float(row["volume"])
    if peak_vol > 0 and last_vol < peak_vol * vol_decay_pct:
        reasons.append(f"Volume decaying — {last_vol/peak_vol:.0%} of peak")

    # 2. Positive delta on high volume (institutions buying back = covering shorts)
    delta = approx_volume_delta(row)
    if delta > 0 and not pd.isna(vol_sma) and vol_sma > 0 and row["volume"] / vol_sma >= 1.5:
        reasons.append("High-volume buy delta — institutions covering shorts")

    # 3. Closed in upper half of range (hidden buying on a down candle)
    if rng > 0 and (row["close"] - row["low"]) / rng > 0.60 and row["close"] < row["open"]:
        reasons.append("Closed in upper 60% of range — hidden buying pressure")

    # 4. Three consecutive declining volume candles
    vols = [float(df.iloc[-i]["volume"]) for i in range(2, 5) if len(df) > i]
    if len(vols) == 3 and vols[0] < vols[1] < vols[2]:
        reasons.append("3 consecutive candles of declining volume")

    if len(reasons) >= 2:
        return {"reasons": reasons, "reason_str": " | ".join(reasons)}

    return None
