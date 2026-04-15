"""
Sentiment Data Fetcher  —  free endpoints only, no API keys required
----------------------------------------------------------------------
Sources:
  Fear & Greed    alternative.me/fng/                    (free, no auth)
  Funding rate    MEXC REST /api/v1/contract/funding_rate (free, no auth)
  Open Interest   MEXC REST /api/v1/contract/open_interest(free, no auth)
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

MEXC_BASE  = "https://contract.mexc.com"
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
# 2.  Funding Rate  (MEXC REST — no auth)
# ══════════════════════════════════════════════════════════════════════════════

_fr_cache: dict = {}
_FR_TTL = 300.0   # 5 minutes (funding settles every 8h, but rate updates frequently)


def get_funding_rate(exchange=None, symbol: str = "") -> float | None:
    """
    Return latest funding rate for symbol via MEXC public REST.
    Negative = longs paying shorts (bearish positioning → contrarian bullish).
    Positive = shorts paying longs (bullish positioning → contrarian bearish).

    `exchange` param kept for API compatibility but not used.
    `symbol` format: 'BTC/USDT:USDT'  →  internally converted to 'BTC_USDT'.
    """
    mexc_sym = _to_mexc_sym(symbol)
    cache_key = f"fr|{mexc_sym}"
    cached = _cache_get(_fr_cache, cache_key, _FR_TTL)
    if cached is not None:
        return cached

    try:
        resp = _SESSION.get(
            f"{MEXC_BASE}/api/v1/contract/funding_rate",
            params={"symbol": mexc_sym},
            timeout=8,
        )
        body = resp.json()
        if body.get("success"):
            rate = body.get("data", {}).get("fundingRate")
            if rate is not None:
                result = float(rate)
                _cache_set(_fr_cache, cache_key, result)
                logger.debug(f"[SENTIMENT] Funding rate {mexc_sym}: {result:.6f}")
                return result
    except Exception as e:
        logger.debug(f"[SENTIMENT] Funding rate {mexc_sym}: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Open Interest  (MEXC REST — no auth)
# ══════════════════════════════════════════════════════════════════════════════

_oi_cache: dict = {}
_OI_TTL = 300.0   # 5 minutes


def get_open_interest(exchange=None, symbol: str = "") -> float | None:
    """
    Return current open interest in USD via MEXC public REST.
    Used to confirm trend: rising OI + rising price = genuine directional move.

    `exchange` param kept for API compatibility but not used.
    """
    mexc_sym = _to_mexc_sym(symbol)
    cache_key = f"oi|{mexc_sym}"
    cached = _cache_get(_oi_cache, cache_key, _OI_TTL)
    if cached is not None:
        return cached

    try:
        resp = _SESSION.get(
            f"{MEXC_BASE}/api/v1/contract/open_interest",
            params={"symbol": mexc_sym},
            timeout=8,
        )
        body = resp.json()
        if body.get("success"):
            data = body.get("data", {})
            # MEXC returns holdVol (contracts) and openInterestUSD
            oi = data.get("openInterestUSD") or data.get("holdVol")
            if oi is not None:
                result = float(oi)
                _cache_set(_oi_cache, cache_key, result)
                logger.debug(f"[SENTIMENT] OI {mexc_sym}: {result:,.0f}")
                return result
    except Exception as e:
        logger.debug(f"[SENTIMENT] OI {mexc_sym}: {e}")
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

def _to_mexc_sym(symbol: str) -> str:
    """'BTC/USDT:USDT' → 'BTC_USDT'"""
    return symbol.split(":")[0].replace("/", "_")
