"""
Confluence Detectors — Institutional-grade confluence filters.
---------------------------------------------------------------
Used AFTER a primary strategy fires to score signal quality.
None of these replace existing logic — they are additive filters only.

Factors (max 5):
  1. Bull Trap         — signal reason already contains "Bull Trap"
  2. Fair Value Gap    — price inside an unfilled FVG on entry_df
  3. Market Structure  — HTF swing structure broke in signal direction
     Shift (MSS)
  4. Liquidity Sweep   — recent stop hunt in entry_df aligned with direction
  5. Order Block (OB)  — price inside a significant HTF order block

Usage:
    from src.confluence import score_confluence
    conf_score, conf_labels = score_confluence(htf_df, entry_df, direction, price, is_bull_trap)
"""

from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger("futures_bot.confluence")


# ------------------------------------------------------------------
# 1. Fair Value Gap (FVG)
# ------------------------------------------------------------------

def detect_fvg_zones(df: pd.DataFrame, lookback: int = 40) -> list[dict]:
    """
    Scan last `lookback` confirmed candles for Fair Value Gaps.

    Bullish FVG : candle[i+2].low  > candle[i].high  → price gapped up
    Bearish FVG : candle[i+2].high < candle[i].low   → price gapped down

    Returns list of {'type': 'bull'|'bear', 'top': float, 'bot': float}
    ordered oldest → newest.
    """
    zones: list[dict] = []
    # Confirmed candles only (exclude live candle at -1)
    data = df.iloc[-(lookback + 2):-1].reset_index(drop=True)
    n = len(data)
    for i in range(n - 2):
        c1_high = float(data.iloc[i]["high"])
        c1_low  = float(data.iloc[i]["low"])
        c3_high = float(data.iloc[i + 2]["high"])
        c3_low  = float(data.iloc[i + 2]["low"])

        if c3_low > c1_high:           # bullish FVG — gap up
            zones.append({"type": "bull", "top": c3_low,  "bot": c1_high})
        elif c3_high < c1_low:         # bearish FVG — gap down
            zones.append({"type": "bear", "top": c1_low,  "bot": c3_high})
    return zones


def fvg_confluence(df: pd.DataFrame, direction: str, price: float,
                   lookback: int = 40) -> bool:
    """
    True if `price` is currently inside an unfilled FVG aligned with direction.
    Bullish FVG + LONG trade → price returning to fill the gap = confluence.
    Bearish FVG + SHORT trade → same logic.
    """
    try:
        zones = detect_fvg_zones(df, lookback)
        target_type = "bull" if direction == "long" else "bear"
        for z in zones:
            if z["type"] == target_type and z["bot"] <= price <= z["top"]:
                return True
        return False
    except Exception as e:
        logger.debug(f"[CONF] FVG check error: {e}")
        return False


# ------------------------------------------------------------------
# 2. Market Structure Shift (MSS)
# ------------------------------------------------------------------

def detect_mss(df: pd.DataFrame, direction: str,
               swing_left: int = 3, swing_right: int = 2) -> bool:
    """
    Detect a Market Structure Shift on df.

    Bullish MSS: latest confirmed close broke ABOVE a previous swing high
                 → buyers took control, uptrend confirmed.
    Bearish MSS: latest confirmed close broke BELOW a previous swing low
                 → sellers took control, downtrend confirmed.

    Uses a looser swing definition (left=3, right=2) to stay sensitive
    without firing on every minor pip move.
    """
    try:
        from src.indicators import find_swing_highs_idx, find_swing_lows_idx

        if len(df) < swing_left + swing_right + 10:
            return False

        close = float(df.iloc[-2]["close"])

        if direction == "long":
            highs = find_swing_highs_idx(df, left=swing_left, right=swing_right)
            # Exclude swings that are too recent (might be the breakout candle itself)
            relevant = [(i, p) for i, p in highs if i < len(df) - swing_right - 3]
            if not relevant:
                return False
            prev_high = relevant[-1][1]   # most recent qualifying swing high
            return close > prev_high

        else:  # short
            lows = find_swing_lows_idx(df, left=swing_left, right=swing_right)
            relevant = [(i, p) for i, p in lows if i < len(df) - swing_right - 3]
            if not relevant:
                return False
            prev_low = relevant[-1][1]    # most recent qualifying swing low
            return close < prev_low

    except Exception as e:
        logger.debug(f"[CONF] MSS check error: {e}")
        return False


# ------------------------------------------------------------------
# 3. Order Block (OB)
# ------------------------------------------------------------------

def detect_order_block(df: pd.DataFrame, direction: str, price: float,
                       lookback: int = 50, atr_body_min: float = 0.4) -> bool:
    """
    Check if `price` is inside a significant Order Block on df.

    Bullish OB : the last bearish candle (>= atr_body_min × ATR body) that
                 was immediately followed by 2+ bullish closes — institutions
                 used that zone as a buy origin; price returning = support.

    Bearish OB : the last bullish candle followed by 2+ bearish closes
                 — institutions sold from there; price returning = resistance.
    """
    try:
        data = df.iloc[-lookback:-1]          # confirmed candles
        if len(data) < 5:
            return False

        atr_val = float(df.iloc[-2].get("atr", 0))
        if pd.isna(atr_val) or atr_val == 0:
            return False

        min_body = atr_val * atr_body_min

        if direction == "long":
            # Hunt for last bearish candle followed by 2 bullish closes
            for i in range(len(data) - 3, 2, -1):
                c = data.iloc[i]
                c_open  = float(c["open"])
                c_close = float(c["close"])
                body    = c_open - c_close        # positive = bearish
                if body < min_body:
                    continue
                next2 = data.iloc[i + 1: i + 3]
                if len(next2) < 2:
                    continue
                if (next2["close"].values > next2["open"].values).all():
                    ob_high = c_open               # top of bearish OB
                    ob_low  = float(c["low"])
                    if ob_low <= price <= ob_high:
                        return True

        else:  # short
            # Hunt for last bullish candle followed by 2 bearish closes
            for i in range(len(data) - 3, 2, -1):
                c = data.iloc[i]
                c_open  = float(c["open"])
                c_close = float(c["close"])
                body    = c_close - c_open         # positive = bullish
                if body < min_body:
                    continue
                next2 = data.iloc[i + 1: i + 3]
                if len(next2) < 2:
                    continue
                if (next2["close"].values < next2["open"].values).all():
                    ob_high = float(c["high"])
                    ob_low  = c_open               # bottom of bullish OB
                    if ob_low <= price <= ob_high:
                        return True

        return False

    except Exception as e:
        logger.debug(f"[CONF] OB check error: {e}")
        return False


# ------------------------------------------------------------------
# Master scorer
# ------------------------------------------------------------------

def score_confluence(
    htf_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    direction: str,
    price: float,
    is_bull_trap: bool = False,
) -> tuple[int, list[str]]:
    """
    Score all confluence factors and return (score, label_list).

    Max score = 5.
      1  Bull Trap (only for bull-trap signals)
      2  Fair Value Gap on entry_df
      3  Market Structure Shift on htf_df
      4  Liquidity Sweep on entry_df
      5  Order Block on htf_df

    htf_df   : higher TF df (4h swing / 30m scalp) — used for MSS + OB
    entry_df : entry TF df (1h swing / 15m scalp)  — used for FVG + sweep
    """
    from src.indicators import detect_liquidity_sweep

    score  = 0
    labels: list[str] = []

    # ── 1. Bull Trap ─────────────────────────────────────────────
    if is_bull_trap:
        score += 1
        labels.append("✅ Bull Trap Detected")

    # ── 2. Fair Value Gap ─────────────────────────────────────────
    if fvg_confluence(entry_df, direction, price):
        score += 1
        tag = "Bearish" if direction == "short" else "Bullish"
        labels.append(f"✅ {tag} FVG Present")

    # ── 3. Market Structure Shift ─────────────────────────────────
    if detect_mss(htf_df, direction):
        score += 1
        tag = "Bearish" if direction == "short" else "Bullish"
        labels.append(f"✅ MSS Confirmed {tag}")

    # ── 4. Liquidity Sweep ────────────────────────────────────────
    try:
        sweep = detect_liquidity_sweep(entry_df)
        # buy_side sweep  = whales grabbed stops ABOVE swing high → SHORT confluence
        # sell_side sweep = whales grabbed stops BELOW swing low  → LONG confluence
        sweep_ok = (
            (direction == "short" and sweep == "buy_side") or
            (direction == "long"  and sweep == "sell_side")
        )
        if sweep_ok:
            score += 1
            labels.append("✅ Liquidity Sweep Detected")
    except Exception as e:
        logger.debug(f"[CONF] Sweep check error: {e}")

    # ── 5. Order Block ────────────────────────────────────────────
    if detect_order_block(htf_df, direction, price):
        score += 1
        labels.append("✅ Order Block Confluence")

    logger.debug(
        f"[CONF] {direction.upper()} score={score}/5 "
        f"trap={is_bull_trap} labels={labels}"
    )
    return score, labels


def confluence_strength_label(score: int) -> str:
    """Human-readable strength label for a confluence score."""
    if score >= 5:
        return "ULTRA 🔥🔥🔥🔥🔥"
    if score >= 4:
        return "STRONG 🔴🔴🔴🔴"
    if score >= 3:
        return "MEDIUM 🟡🟡🟡"
    return "WEAK 🟢🟢"
