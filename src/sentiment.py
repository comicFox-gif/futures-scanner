"""
Sentiment Data Fetcher
-----------------------
External market sentiment signals for the elite confluence system.

Sources:
  - Fear & Greed Index: api.alternative.me (free, no auth, cached 1h)
  - Funding Rates:      ccxt exchange.fetch_funding_rate()
  - Open Interest:      ccxt exchange.fetch_open_interest()
  - Long/Short Ratio:   Coinglass (optional — needs COINGLASS_API_KEY env var)
  - Liquidation data:   Coinglass (optional — needs COINGLASS_API_KEY env var)

All functions degrade gracefully on failure: return None / default values.
Missing Coinglass key → those sentiment points are awarded neutrally (don't penalise).
"""
from __future__ import annotations
import logging
import os
import time

import requests

logger = logging.getLogger("futures_bot.sentiment")

# ── Fear & Greed cache ────────────────────────────────────────────────────────

_fng_cache: dict = {"value": 50, "label": "Neutral", "fetched_at": 0.0}
_FNG_TTL = 3600  # re-fetch at most once per hour


def get_fear_greed() -> dict:
    """
    Return the current Fear & Greed index.
    {'value': 0–100, 'label': 'Extreme Fear' | 'Fear' | 'Neutral' | 'Greed' | 'Extreme Greed'}
    Cached for 1 hour to avoid rate-limit issues.
    """
    now = time.time()
    if now - _fng_cache["fetched_at"] < _FNG_TTL:
        return {"value": _fng_cache["value"], "label": _fng_cache["label"]}
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1", timeout=10
        )
        data = resp.json()["data"][0]
        _fng_cache["value"]      = int(data["value"])
        _fng_cache["label"]      = data["value_classification"]
        _fng_cache["fetched_at"] = now
        logger.debug(f"[SENTIMENT] F&G updated: {_fng_cache['value']} ({_fng_cache['label']})")
    except Exception as e:
        logger.warning(f"[SENTIMENT] Fear & Greed fetch failed: {e}")
    return {"value": _fng_cache["value"], "label": _fng_cache["label"]}


# ── ccxt-based live data ──────────────────────────────────────────────────────

def get_funding_rate(exchange, symbol: str) -> float | None:
    """
    Return latest funding rate for symbol.
    Negative rate = longs paying shorts = bearish positioning → bullish contrarian
    Positive rate = shorts paying longs = bullish positioning → can be bearish contrarian
    Returns None on failure.
    """
    try:
        data = exchange.fetch_funding_rate(symbol)
        if data:
            rate = data.get("fundingRate") or data.get("lastFundingRate")
            return float(rate) if rate is not None else None
    except Exception as e:
        logger.debug(f"[SENTIMENT] Funding rate {symbol}: {e}")
    return None


def get_open_interest(exchange, symbol: str) -> float | None:
    """
    Return current open interest amount (in base units).
    Used to confirm trend: rising OI + rising price = real trend.
    Returns None on failure.
    """
    try:
        data = exchange.fetch_open_interest(symbol)
        if data:
            oi = data.get("openInterestAmount") or data.get("openInterest")
            return float(oi) if oi else None
    except Exception as e:
        logger.debug(f"[SENTIMENT] Open interest {symbol}: {e}")
    return None


# ── Coinglass (optional, requires API key) ────────────────────────────────────

_COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"
_cg_cache: dict = {}
_CG_TTL = 900  # cache 15 minutes


def _cg_get(endpoint: str, params: dict) -> dict | None:
    """Internal helper: hit Coinglass API with caching. Returns None if no key or on failure."""
    api_key = os.getenv("COINGLASS_API_KEY", "").strip()
    if not api_key:
        return None

    cache_key = f"{endpoint}|{sorted(params.items())}"
    entry = _cg_cache.get(cache_key)
    if entry and time.time() - entry["t"] < _CG_TTL:
        return entry["data"]

    try:
        resp = requests.get(
            f"{_COINGLASS_BASE}/{endpoint}",
            params=params,
            headers={"coinglassSecret": api_key},
            timeout=10,
        )
        body = resp.json()
        if body.get("success"):
            _cg_cache[cache_key] = {"data": body["data"], "t": time.time()}
            return body["data"]
        logger.debug(f"[COINGLASS] {endpoint} non-success: {body.get('msg')}")
    except Exception as e:
        logger.debug(f"[COINGLASS] {endpoint} error: {e}")
    return None


def get_long_short_ratio(symbol_base: str) -> float | None:
    """
    Return long/short ratio from Coinglass (global accounts).
    > 1.0 = more longs than shorts.
    < 1.0 = more shorts than longs.
    Returns None if Coinglass key not set or request fails.
    """
    data = _cg_get(
        "futures/globalLongShortAccountRatio",
        {"symbol": symbol_base + "USDT", "period": "4h", "limit": 1},
    )
    if data and isinstance(data, list) and data:
        row    = data[-1]
        longs  = float(row.get("longAccount",  50) or 50)
        shorts = float(row.get("shortAccount", 50) or 50)
        if shorts > 0:
            return round(longs / shorts, 3)
    return None


def get_liquidation_pressure(symbol_base: str) -> dict | None:
    """
    Return latest liquidation data from Coinglass.
    {'buy_liq': float, 'sell_liq': float}  — USD values for last 1h.
    buy_liq  = long liquidations (price fell, longs got rekt) → bearish
    sell_liq = short liquidations (price rose, shorts got rekt) → bullish
    Returns None if unavailable.
    """
    data = _cg_get("liquidation/coin", {"symbol": symbol_base, "range": "1h"})
    if data:
        return {
            "buy_liq":  float(data.get("buyLiquidationUSD",  0) or 0),
            "sell_liq": float(data.get("sellLiquidationUSD", 0) or 0),
        }
    return None
