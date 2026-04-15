"""
Sentiment Data Fetcher  —  free endpoints only, no API keys required
----------------------------------------------------------------------
Sources:
  Fear & Greed    alternative.me/fng/                    (free, no auth)
  Funding rate    Bybit REST /v5/market/tickers           (free, no auth)
  Open Interest   Bybit REST /v5/market/tickers           (free, no auth)
  Long/Short ratio Bybit REST /v5/market/account-ratio    (free, no auth)
  BTC Dominance   CoinGecko /api/v3/global                (free, no auth)

All functions are fail-safe: return None / neutral defaults on any error.
Callers award the sentiment point neutrally when data is unavailable.
"""
from __future__ import annotations
import logging
import time

import requests

logger = logging.getLogger("futures_bot.sentiment")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "futures-bot/2.0"})

BYBIT_BASE = "https://api.bybit.com"


# ── Shared cache helper ────────────────────────────────────────────────────────

def _cache_get(cache: dict, key: str, ttl: float):
    entry = cache.get(key)
    if entry and time.time() - entry["t"] < ttl:
        return entry["v"]
    return None


def _cache_set(cache: dict, key: str, value):
    cache[key] = {"v": value, "t": time.time()}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Fear & Greed Index  (alternative.me)
# ══════════════════════════════════════════════════════════════════════════════

_fng_cache: dict = {}
_FNG_TTL = 3600.0   # 1 hour


def get_fear_greed() -> dict:
    """
    Return {'value': 0-100, 'label': str}.
    Extreme Fear (0-25) = best long zone.
    Extreme Greed (75-100) = avoid longs.
    Cached 1 hour.
    """
    cached = _cache_get(_fng_cache, "fng", _FNG_TTL)
    if cached:
        return cached

    default = {"value": 50, "label": "Neutral"}
    try:
        resp = _SESSION.get(
            "https://api.alternative.me/fng/?limit=1", timeout=8
        )
        data = resp.json()["data"][0]
        result = {
            "value": int(data["value"]),
            "label": data["value_classification"],
        }
        _cache_set(_fng_cache, "fng", result)
        logger.debug(f"[SENTIMENT] F&G: {result['value']} ({result['label']})")
        return result
    except Exception as e:
        logger.debug(f"[SENTIMENT] F&G fetch failed: {e}")
    return default


# ══════════════════════════════════════════════════════════════════════════════
# 2 & 3.  Funding Rate + Open Interest  (Bybit /v5/market/tickers — no auth)
# ══════════════════════════════════════════════════════════════════════════════
# Both values come from the same ticker call, so we share one cache.

_ticker_cache: dict = {}
_TICKER_TTL = 300.0   # 5 minutes


def _get_bybit_ticker(symbol: str) -> dict:
    """
    Fetch linear ticker for `symbol` from Bybit and return the first list entry.
    `symbol` format: 'BTC/USDT:USDT'  →  'BTCUSDT'.
    Returns {} on any error.
    """
    bybit_sym = _to_bybit_sym(symbol)
    cache_key = f"ticker|{bybit_sym}"
    cached = _cache_get(_ticker_cache, cache_key, _TICKER_TTL)
    if cached is not None:
        return cached

    try:
        resp = _SESSION.get(
            f"{BYBIT_BASE}/v5/market/tickers",
            params={"category": "linear", "symbol": bybit_sym},
            timeout=8,
        )
        body = resp.json()
        if body.get("retCode") == 0:
            rows = body.get("result", {}).get("list", [])
            if rows:
                result = rows[0]
                _cache_set(_ticker_cache, cache_key, result)
                return result
    except Exception as e:
        logger.debug(f"[SENTIMENT] Bybit ticker {bybit_sym}: {e}")
    return {}


def get_funding_rate(exchange=None, symbol: str = "") -> float | None:
    """
    Return latest funding rate for symbol via Bybit public REST.
    Negative = longs paying shorts (bearish positioning → contrarian bullish).
    Positive = shorts paying longs (bullish positioning → contrarian bearish).

    `exchange` param kept for API compatibility but not used.
    `symbol` format: 'BTC/USDT:USDT'  →  internally converted to 'BTCUSDT'.
    """
    ticker = _get_bybit_ticker(symbol)
    rate = ticker.get("fundingRate")
    if rate is not None:
        result = float(rate)
        logger.debug(f"[SENTIMENT] Funding rate {symbol}: {result:.6f}")
        return result
    return None


def get_open_interest(exchange=None, symbol: str = "") -> float | None:
    """
    Return current open interest in USD via Bybit public REST.
    Used to confirm trend: rising OI + rising price = genuine directional move.

    `exchange` param kept for API compatibility but not used.
    """
    ticker = _get_bybit_ticker(symbol)
    oi = ticker.get("openInterestValue")   # USD-denominated OI
    if oi is not None:
        result = float(oi)
        logger.debug(f"[SENTIMENT] OI {symbol}: {result:,.0f}")
        return result
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Long / Short Ratio  (Bybit public REST — no auth)
# ══════════════════════════════════════════════════════════════════════════════

_ls_cache: dict = {}
_LS_TTL = 600.0   # 10 minutes


def get_long_short_ratio(symbol_base: str) -> float | None:
    """
    Return long/short account ratio from Bybit (global accounts, 4H period).
    > 1.0  = more longs than shorts.
    < 1.0  = more shorts than longs.
    Returns None on failure.

    `symbol_base` should be the base currency string, e.g. 'BTC', 'ETH'.
    """
    bybit_sym = f"{symbol_base.upper()}USDT"
    cache_key = f"ls|{bybit_sym}"
    cached = _cache_get(_ls_cache, cache_key, _LS_TTL)
    if cached is not None:
        return cached

    try:
        resp = _SESSION.get(
            f"{BYBIT_BASE}/v5/market/account-ratio",
            params={"category": "linear", "symbol": bybit_sym, "period": "4h", "limit": 1},
            timeout=8,
        )
        body = resp.json()
        if body.get("retCode") == 0:
            rows = body.get("result", {}).get("list", [])
            if rows:
                row    = rows[0]
                longs  = float(row.get("buyRatio",  0.5) or 0.5)
                shorts = float(row.get("sellRatio", 0.5) or 0.5)
                ratio  = round(longs / shorts, 3) if shorts > 0 else None
                if ratio is not None:
                    _cache_set(_ls_cache, cache_key, ratio)
                    logger.debug(f"[SENTIMENT] L/S ratio {bybit_sym}: {ratio:.3f}")
                    return ratio
    except Exception as e:
        logger.debug(f"[SENTIMENT] L/S ratio {bybit_sym}: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BTC Dominance  (CoinGecko — free, no auth)
# ══════════════════════════════════════════════════════════════════════════════

_btcd_cache: dict = {}
_BTCD_TTL = 3600.0   # 1 hour (dominance changes slowly)


def get_btc_dominance() -> float | None:
    """
    Return BTC market cap dominance as a percentage (e.g. 52.3).
    > 55% = BTC-only market; < 45% = alt-season.
    Returns None on failure.
    """
    cached = _cache_get(_btcd_cache, "btcd", _BTCD_TTL)
    if cached is not None:
        return cached

    try:
        resp = _SESSION.get(
            "https://api.coingecko.com/api/v3/global", timeout=8
        )
        data = resp.json().get("data", {})
        pct  = data.get("market_cap_percentage", {}).get("btc")
        if pct is not None:
            result = float(pct)
            _cache_set(_btcd_cache, "btcd", result)
            logger.debug(f"[SENTIMENT] BTC.D: {result:.1f}%")
            return result
    except Exception as e:
        logger.debug(f"[SENTIMENT] BTC.D fetch: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Liquidity zones — internal (price-action only, no external API)
# ══════════════════════════════════════════════════════════════════════════════
#
# These are computed from OHLCV data in advanced_confluence.detect_liquidity().
# No function needed here; provided for callers that want a direct import.
# See: src/advanced_confluence.py  → detect_liquidity(df)

def get_liquidation_pressure(symbol_base: str) -> dict | None:
    """
    Stub — Coinglass liquidation data removed.
    Returns None so callers award the sentiment point neutrally.
    Use advanced_confluence.detect_liquidity() for price-action liquidity zones.
    """
    return None


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_bybit_sym(symbol: str) -> str:
    """'BTC/USDT:USDT' → 'BTCUSDT'"""
    return symbol.split(":")[0].replace("/", "")
