"""
Forex Data Fetcher
------------------
Uses yfinance to fetch OHLCV data for forex pairs.
No API key required — works globally including Railway.

Symbol format: EURUSD -> EURUSD=X (Yahoo Finance format)

Supported timeframes: 1m, 5m, 15m, 30m, 1h, 4h (resampled from 1h), 1d

Data availability:
  1m/5m/15m/30m : last 60 days
  1h            : last 730 days (~2 years)
  4h            : resampled from 1h (same history)
  1d            : last 10 years
"""

from __future__ import annotations
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger("forex_bot.data")

# Yahoo Finance interval strings
_YF_INTERVAL = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "1h",   # fetched as 1h, resampled to 4h
    "1d":  "1d",
}

# Period to fetch (must cover lookback_candles with headroom)
_YF_PERIOD = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "4h":  "730d",
    "1d":  "5y",
}


def pair_to_yf(pair: str) -> str:
    """EURUSD -> EURUSD=X"""
    return pair if "=X" in pair else pair + "=X"


def fetch_ohlcv(pair: str, timeframe: str, lookback: int = 300) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a forex pair.
    Returns a DataFrame indexed by UTC datetime with columns:
      open, high, low, close, volume
    Returns None on failure or insufficient data.
    """
    if timeframe not in _YF_INTERVAL:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return None

    yf_symbol = pair_to_yf(pair)
    interval  = _YF_INTERVAL[timeframe]
    period    = _YF_PERIOD[timeframe]

    try:
        ticker = yf.Ticker(yf_symbol)
        raw    = ticker.history(period=period, interval=interval, auto_adjust=True)

        if raw is None or len(raw) < 50:
            logger.warning(f"Insufficient data: {pair} {timeframe} ({len(raw) if raw is not None else 0} candles)")
            return None

        df = _normalize(raw)
        if df is None or len(df) < 50:
            return None

        # Resample 1h -> 4h
        if timeframe == "4h":
            df = _resample_4h(df)
            if df is None or len(df) < 30:
                logger.warning(f"4H resample insufficient: {pair}")
                return None

        return df.iloc[-lookback:].copy()

    except Exception as e:
        logger.error(f"fetch_ohlcv({pair}, {timeframe}): {e}")
        return None


def _normalize(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Standardise yfinance output to lowercase OHLCV, timezone-naive UTC index."""
    try:
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]

        # Drop yfinance-only columns
        for col in ("dividends", "stock splits", "capital gains"):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # Normalise index to timezone-naive UTC
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index)

        # Drop rows with zero/NaN close (gaps, weekends)
        df = df[df["close"] > 0].dropna(subset=["close"])
        return df
    except Exception as e:
        logger.error(f"_normalize failed: {e}")
        return None


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame | None:
    """Resample 1H DataFrame to 4H."""
    try:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df4 = df.resample("4h", label="left", closed="left").agg(agg).dropna(subset=["close"])
        df4 = df4[df4["close"] > 0]
        return df4
    except Exception as e:
        logger.error(f"_resample_4h failed: {e}")
        return None
