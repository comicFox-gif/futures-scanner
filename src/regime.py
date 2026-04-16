"""
Market Regime Detector
-----------------------
5-vote BTC structural analysis → 'bull', 'bear', or 'neutral'

Votes:
  V1: Wyckoff phase on BTC Weekly  (accumulation=bull, distribution=bear)
  V2: Wyckoff phase on BTC Daily   (accumulation=bull, distribution=bear)
  V3: BTC Dominance level          (>52%=bull, <48%=bear)
  V4: Intermarket confirmation     (3+/5 macro factors aligned)
  V5: BTC 4H price structure       (Higher Highs=bull, Lower Lows=bear)

4+ bull votes → bull regime (allow longs, block shorts)
4+ bear votes → bear regime (allow shorts, block longs)
Otherwise     → neutral (all signals blocked)

No EMA or MACD — pure structural / macro analysis.
"""
from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger("futures_bot.regime")


def _has_higher_highs(df: pd.DataFrame, lookback: int = 12) -> bool:
    """True if recent 4H candles show ascending swing highs (bull structure)."""
    highs = df["high"].iloc[-lookback:].values
    if len(highs) < 6:
        return False
    n = len(highs) // 3
    if n == 0:
        return False
    return float(highs[-n:].max()) > float(highs[-2*n:-n].max()) > float(highs[:n].max())


def _has_lower_lows(df: pd.DataFrame, lookback: int = 12) -> bool:
    """True if recent 4H candles show descending swing lows (bear structure)."""
    lows = df["low"].iloc[-lookback:].values
    if len(lows) < 6:
        return False
    n = len(lows) // 3
    if n == 0:
        return False
    return float(lows[-n:].min()) < float(lows[-2*n:-n].min()) < float(lows[:n].min())


def _wyckoff_vote(df: pd.DataFrame | None, label: str) -> tuple[int, int, str]:
    """
    Run Wyckoff detection on df.
    Returns (bull_vote, bear_vote, note_str).
    """
    if df is None or len(df) < 30:
        return 0, 0, f"{label}:N/A"
    try:
        from src.advanced_confluence import detect_wyckoff
        w = detect_wyckoff(df)
        phase = w.get("phase", "unknown")
        if phase == "accumulation":
            return 1, 0, f"{label}:Accum"
        if phase == "distribution":
            return 0, 1, f"{label}:Dist"
    except Exception as e:
        logger.debug(f"[REGIME] Wyckoff {label}: {e}")
    return 0, 0, f"{label}:neutral"


def _btcd_vote() -> tuple[int, int, str]:
    """
    BTC Dominance vote.
    >52% = BTC leading = bull.  <48% = alt season = bear.  48-52% = neutral.
    """
    try:
        from src.sentiment import get_btc_dominance
        btcd = get_btc_dominance()
        if btcd is None:
            return 0, 0, "BTC.D:N/A"
        if btcd > 52.0:
            return 1, 0, f"BTC.D:{btcd:.1f}%↑"
        if btcd < 48.0:
            return 0, 1, f"BTC.D:{btcd:.1f}%↓"
        return 0, 0, f"BTC.D:{btcd:.1f}%(neutral)"
    except Exception as e:
        logger.debug(f"[REGIME] BTC.D: {e}")
        return 0, 0, "BTC.D:err"


def _intermarket_vote(exchange=None) -> tuple[int, int, str]:
    """
    Intermarket vote — check macro factors for both directions.
    Bull if 3+/5 align for long. Bear if 3+/5 align for short.
    """
    try:
        from src.advanced_confluence import get_intermarket_score
        bull_im = get_intermarket_score(exchange=exchange, direction="long")
        bear_im = get_intermarket_score(exchange=exchange, direction="short")
        if bull_im["n_align"] >= 3:
            return 1, 0, f"IM:{bull_im['n_align']}/5 bull"
        if bear_im["n_align"] >= 3:
            return 0, 1, f"IM:{bear_im['n_align']}/5 bear"
    except Exception as e:
        logger.debug(f"[REGIME] Intermarket: {e}")
    return 0, 0, "IM:neutral"


def detect_regime(
    btc_4h_df: pd.DataFrame | None,
    btc_daily_df: pd.DataFrame | None,
    btc_weekly_df: pd.DataFrame | None = None,
    exchange=None,
) -> dict:
    """
    Classify the crypto market regime using BTC structural analysis.

    Returns:
        {
            'regime':     'bull' | 'bear' | 'neutral',
            'bull_votes': int,
            'bear_votes': int,
            'label':      str,
        }
    """
    if btc_4h_df is None or len(btc_4h_df) < 10:
        return {
            "regime": "neutral", "bull_votes": 0, "bear_votes": 0,
            "label": "neutral (insufficient BTC data)",
        }

    bull = 0
    bear = 0
    notes: list[str] = []

    # V1: Wyckoff on Weekly BTC
    b, r, n = _wyckoff_vote(btc_weekly_df, "W-Wyck")
    bull += b; bear += r; notes.append(n)

    # V2: Wyckoff on Daily BTC
    b, r, n = _wyckoff_vote(btc_daily_df, "D-Wyck")
    bull += b; bear += r; notes.append(n)

    # V3: BTC Dominance
    b, r, n = _btcd_vote()
    bull += b; bear += r; notes.append(n)

    # V4: Intermarket macro confirmation
    b, r, n = _intermarket_vote(exchange)
    bull += b; bear += r; notes.append(n)

    # V5: BTC 4H price structure (HH or LL)
    if _has_higher_highs(btc_4h_df):
        bull += 1; notes.append("4H:HH")
    elif _has_lower_lows(btc_4h_df):
        bear += 1; notes.append("4H:LL")
    else:
        notes.append("4H:ranging")

    # Classify
    if bull >= 4:
        regime = "bull"
    elif bear >= 4:
        regime = "bear"
    else:
        regime = "neutral"

    label = f"{regime} ({bull}🟢 {bear}🔴 | {', '.join(notes)})"
    return {"regime": regime, "bull_votes": bull, "bear_votes": bear, "label": label}
