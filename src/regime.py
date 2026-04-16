"""
Market Regime Detector
-----------------------
5-vote BTC structural analysis → 'bull', 'bear', or 'neutral'

Votes:
  V1: BTC Weekly structural trend   (close > 10W SMA + trending up = bull)
  V2: BTC Daily structural trend    (close > 20D SMA + trending up = bull)
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


def _trend_vote(df: pd.DataFrame | None, label: str, sma_period: int = 20) -> tuple[int, int, str]:
    """
    Structural trend vote using SMA and recent price direction.
    Bull: close > SMA AND close is higher than N candles ago.
    Bear: close < SMA AND close is lower than N candles ago.
    Always fires — no dependency on rare volume climax events.
    """
    if df is None or len(df) < sma_period + 5:
        return 0, 0, f"{label}:N/A"
    try:
        close     = df["close"].iloc[-(sma_period + 5):]
        sma       = float(close.rolling(sma_period).mean().iloc[-2])
        cur_close = float(close.iloc[-2])
        past_close = float(close.iloc[-(sma_period // 2) - 2])  # half-period ago

        above_sma = cur_close > sma
        trending_up = cur_close > past_close

        if above_sma and trending_up:
            return 1, 0, f"{label}:↑{cur_close:.0f}>SMA{sma:.0f}"
        if not above_sma and not trending_up:
            return 0, 1, f"{label}:↓{cur_close:.0f}<SMA{sma:.0f}"
    except Exception as e:
        logger.debug(f"[REGIME] Trend {label}: {e}")
    return 0, 0, f"{label}:mixed"


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

    # V1: Weekly structural trend (close vs 10-week SMA + direction)
    b, r, n = _trend_vote(btc_weekly_df, "W-Trend", sma_period=10)
    bull += b; bear += r; notes.append(n)

    # V2: Daily structural trend (close vs 20-day SMA + direction)
    b, r, n = _trend_vote(btc_daily_df, "D-Trend", sma_period=20)
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
