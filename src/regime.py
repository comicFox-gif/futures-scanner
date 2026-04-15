"""
Market Regime Detector
-----------------------
5-vote BTC analysis → 'bull', 'bear', or 'neutral'

Votes:
  V1: BTC 4H close > EMA50
  V2: BTC 4H close > EMA200
  V3: BTC 4H MACD line > 0
  V4: BTC Daily close > EMA50
  V5: BTC 4H price structure — Higher Highs (bull) or Lower Lows (bear)

4+ bull votes → bull regime (allow longs, block shorts)
4+ bear votes → bear regime (allow shorts, block longs)
Otherwise     → neutral (all signals blocked)
"""
from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger("futures_bot.regime")


def _has_higher_highs(df: pd.DataFrame, lookback: int = 12) -> bool:
    """True if the last section of candles shows ascending highs (bull structure)."""
    highs = df["high"].iloc[-lookback:].values
    if len(highs) < 6:
        return False
    n = len(highs) // 3
    if n == 0:
        return False
    return float(highs[-n:].max()) > float(highs[-2*n:-n].max()) > float(highs[:n].max())


def _has_lower_lows(df: pd.DataFrame, lookback: int = 12) -> bool:
    """True if the last section of candles shows descending lows (bear structure)."""
    lows = df["low"].iloc[-lookback:].values
    if len(lows) < 6:
        return False
    n = len(lows) // 3
    if n == 0:
        return False
    return float(lows[-n:].min()) < float(lows[-2*n:-n].min()) < float(lows[:n].min())


def detect_regime(
    btc_4h_df: pd.DataFrame | None,
    btc_daily_df: pd.DataFrame | None,
) -> dict:
    """
    Classify the crypto market regime using BTC structure.

    Returns:
        {
            'regime':     'bull' | 'bear' | 'neutral',
            'bull_votes': int,
            'bear_votes': int,
            'label':      str,   # human-readable summary
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

    row   = btc_4h_df.iloc[-2]
    price = float(row["close"])

    # V1: 4H price vs EMA50
    ema50 = float(row.get("ema_50", float("nan")))
    if not pd.isna(ema50):
        if price > ema50:
            bull += 1; notes.append(f"4H>EMA50({ema50:.0f})")
        else:
            bear += 1; notes.append(f"4H<EMA50({ema50:.0f})")

    # V2: 4H price vs EMA200
    ema200 = float(row.get("ema_200", float("nan")))
    if not pd.isna(ema200):
        if price > ema200:
            bull += 1; notes.append("4H>EMA200")
        else:
            bear += 1; notes.append("4H<EMA200")

    # V3: 4H MACD line vs zero
    macd = float(row.get("macd", float("nan")))
    if not pd.isna(macd):
        if macd > 0:
            bull += 1; notes.append("MACD>0")
        else:
            bear += 1; notes.append("MACD<0")

    # V4: Daily EMA50
    if btc_daily_df is not None and len(btc_daily_df) >= 5:
        row_d   = btc_daily_df.iloc[-2]
        price_d = float(row_d["close"])
        ema50_d = float(row_d.get("ema_50", float("nan")))
        if not pd.isna(ema50_d):
            if price_d > ema50_d:
                bull += 1; notes.append("D>EMA50")
            else:
                bear += 1; notes.append("D<EMA50")
    else:
        # No daily data — skip this vote (don't penalise)
        pass

    # V5: Price structure
    if _has_higher_highs(btc_4h_df):
        bull += 1; notes.append("HH")
    elif _has_lower_lows(btc_4h_df):
        bear += 1; notes.append("LL")

    # Classify
    if bull >= 4:
        regime = "bull"
    elif bear >= 4:
        regime = "bear"
    else:
        regime = "neutral"

    label = f"{regime} ({bull}🟢 {bear}🔴 | {', '.join(notes)})"
    return {"regime": regime, "bull_votes": bull, "bear_votes": bear, "label": label}
