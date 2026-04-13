"""
MEXC Futures Executor
-----------------------------------
Places real orders on MEXC perpetual futures using the REST API.
Drop-in replacement for BybitExecutor — identical public interface.

Env vars:
  MEXC_KEY      — API key
  MEXC_SECRET   — API secret
  MEXC_LEVERAGE — leverage per trade (default 10)

Risk sizing (mirrors BybitExecutor):
  risk_usdt     = $4 fixed
  qty_coin      = risk_usdt / sl_dist
  vol           = floor(qty_coin / contract_size)   ← whole contracts
  notional_cap  = balance * leverage * 0.80

MEXC Futures API:
  Base URL    : https://contract.mexc.com
  Auth        : ApiKey + Request-Time + Signature headers
  Signature   : HmacSHA256(api_key + timestamp + body_or_query_string)
  Side values : 1=open long, 2=close long, 3=open short, 4=close short
  Order type  : 5 = market
  Open type   : 2 = cross margin
"""

from __future__ import annotations
import hashlib
import hmac
import json
import logging
import math
import os
import time
import traceback
from datetime import datetime, timedelta

import requests

logger = logging.getLogger("futures_bot.mexc")

BASE = "https://contract.mexc.com"


class MexcExecutor:
    def __init__(self, api_key: str = "", api_secret: str = "",
                 leverage: int = 10, risk_pct: float = 0.01,
                 max_positions: int = 50, max_risk_usdt: float = 10.0):
        self.api_key       = api_key
        self.api_secret    = api_secret
        self.leverage      = leverage
        self.risk_pct      = risk_pct
        self.max_positions = max_positions
        self.max_risk_usdt = max_risk_usdt
        self.enabled       = bool(api_key and api_secret)
        self._contract_cache: dict[str, dict] = {}
        self._last_order:    dict[str, datetime] = {}
        self._order_cooldown_min = 20

        # Proxy support — set MEXC_PROXY env var to route around datacenter IP blocks
        # Format: http://user:pass@host:port  or  socks5://user:pass@host:port
        proxy_url = os.getenv("MEXC_PROXY", "")
        self._proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        if proxy_url:
            host = proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url.split("//")[-1]
            logger.info(f"[MEXC] Proxy configured: {host}")

        if not self.enabled:
            logger.info("[MEXC] No API keys — executor disabled")
            return

        masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "???"
        logger.info(f"[MEXC] Executor ready | key={masked} | leverage={leverage}x")

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _headers(self, body_or_qs: str = "") -> dict:
        ts  = str(int(time.time() * 1000))
        msg = self.api_key + ts + body_or_qs
        sig = hmac.new(
            self.api_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "ApiKey":        self.api_key,
            "Request-Time":  ts,
            "Signature":     sig,
            "Content-Type":  "application/json",
            "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        params = params or {}
        qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        resp = requests.get(
            BASE + path + (f"?{qs}" if qs else ""),
            headers=self._headers(qs),
            proxies=self._proxies,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            raise RuntimeError(data.get("message", str(data)))
        return data

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body, separators=(",", ":"))
        resp = requests.post(
            BASE + path,
            headers=self._headers(body_str),
            data=body_str,
            proxies=self._proxies,
            timeout=15,
        )
        if resp.status_code == 403:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:200]
            raise RuntimeError(
                f"403 Forbidden — API key missing 'Trade' permission for futures, "
                f"or IP not whitelisted. MEXC detail: {detail}"
            )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            raise RuntimeError(data.get("message", str(data)))
        return data

    # ------------------------------------------------------------------
    # Symbol + contract specs
    # ------------------------------------------------------------------

    def _to_symbol(self, symbol: str) -> str:
        """'BTC/USDT:USDT' → 'BTC_USDT'"""
        return symbol.split(":")[0].replace("/", "_")

    def _get_contract(self, symbol: str) -> dict:
        """Fetch and cache contract specs for a symbol."""
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]
        try:
            data = self._get("/api/v1/contract/detail", {"symbol": symbol})
            info = data["data"]
            spec = {
                "contract_size": float(info["contractSize"]),
                "min_vol":       int(info.get("minVol", 1)),
                "max_vol":       int(info.get("maxVol", 1_000_000)),
                "price_scale":   int(info.get("priceScale", 2)),
            }
            logger.info(
                f"[MEXC] {symbol} spec — contractSize={spec['contract_size']} "
                f"minVol={spec['min_vol']} priceScale={spec['price_scale']}"
            )
        except Exception as e:
            logger.warning(f"[MEXC] get_contract({symbol}) failed: {e} — using defaults")
            spec = {"contract_size": 0.0001, "min_vol": 1, "max_vol": 1_000_000, "price_scale": 2}
        self._contract_cache[symbol] = spec
        return spec

    def _round_price(self, price: float, scale: int) -> float:
        return round(price, scale)

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def _get_balance(self) -> float:
        try:
            data = self._get("/api/v1/private/account/assets")
            for asset in data.get("data", []):
                if asset.get("currency") == "USDT":
                    return float(asset.get("availableBalance", 0))
            return 0.0
        except Exception as e:
            logger.warning(f"[MEXC] get_balance failed: {e} — using 210 fallback")
            return 210.0

    # ------------------------------------------------------------------
    # Position check
    # ------------------------------------------------------------------

    def has_open_position(self, symbol: str) -> bool:
        try:
            data = self._get("/api/v1/private/position/open_positions", {"symbol": symbol})
            for pos in data.get("data", []):
                if float(pos.get("holdVol", 0)) > 0:
                    logger.info(
                        f"[MEXC] Skipping {symbol} — position already open "
                        f"(vol={pos['holdVol']})"
                    )
                    return True
            return False
        except Exception as e:
            logger.warning(f"[MEXC] has_open_position({symbol}): {e} — assuming no position")
            return False

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def _set_leverage(self, symbol: str, position_type: int) -> bool:
        """position_type: 1=long side, 2=short side."""
        try:
            self._post("/api/v1/private/position/change_leverage", {
                "symbol":       symbol,
                "leverage":     self.leverage,
                "openType":     2,
                "positionType": position_type,
            })
            logger.info(f"[MEXC] Leverage set to {self.leverage}x for {symbol}")
            return True
        except Exception as e:
            msg = str(e).lower()
            if "not modified" in msg or "same" in msg or "no change" in msg:
                return True
            logger.warning(f"[MEXC] set_leverage({symbol}): {e}")
            return False

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> dict:
        """
        Place a market entry + SL + TP on MEXC.
        Returns {"order_id": str} or {} on failure.
        """
        if not self.enabled:
            return {}

        symbol = self._to_symbol(signal["symbol"])

        if self.has_open_position(symbol):
            return {}

        last = self._last_order.get(symbol)
        if last and datetime.utcnow() - last < timedelta(minutes=self._order_cooldown_min):
            remaining = int(
                (timedelta(minutes=self._order_cooldown_min) -
                 (datetime.utcnow() - last)).total_seconds() / 60
            )
            logger.info(f"[MEXC] Skipping {symbol} — cooldown ({remaining}min remaining)")
            return {}

        direction = signal["direction"]
        entry     = float(signal["entry"])
        sl        = float(signal["sl"])
        tp3       = float(signal["tp3"])
        side      = 1 if direction == "long" else 3   # 1=open long, 3=open short

        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            logger.warning(f"[MEXC] Zero SL distance for {symbol} — skipping")
            return {}

        if direction == "long" and not (sl < entry < tp3):
            logger.warning(f"[MEXC] Bad levels LONG {symbol}: SL={sl} E={entry} TP={tp3}")
            return {}
        if direction == "short" and not (tp3 < entry < sl):
            logger.warning(f"[MEXC] Bad levels SHORT {symbol}: TP={tp3} E={entry} SL={sl}")
            return {}

        spec          = self._get_contract(symbol)
        contract_size = spec["contract_size"]
        scale         = spec["price_scale"]

        # Convert coin qty → contracts
        risk_usdt = 4.0
        qty_coin  = risk_usdt / sl_dist
        vol       = math.floor(qty_coin / contract_size)

        # Notional cap: never exceed 80% of available margin × leverage
        balance      = self._get_balance()
        max_notional = balance * self.leverage * 0.80
        vol_cap      = math.floor(max_notional / (entry * contract_size))
        if vol > vol_cap:
            logger.warning(
                f"[MEXC] {symbol} vol={vol} notional≈{vol * contract_size * entry:.2f} "
                f"exceeds cap (bal={balance:.2f} lev={self.leverage}x) — capping to {vol_cap}"
            )
            vol = vol_cap

        vol = max(spec["min_vol"], min(spec["max_vol"], vol))
        if vol <= 0:
            logger.warning(f"[MEXC] vol rounds to 0 for {symbol} — skipping")
            return {}

        notional       = vol * contract_size * entry
        effective_risk = vol * contract_size * sl_dist
        sl_price       = self._round_price(sl, scale)
        tp_price       = self._round_price(tp3, scale)

        logger.info(
            f"[MEXC] {symbol} | Bal=${balance:.2f} | Risk=${effective_risk:.2f} "
            f"| vol={vol} contracts | notional≈{notional:.2f} | SL={sl_price} TP={tp_price}"
        )

        pos_type = 1 if direction == "long" else 2
        if not self._set_leverage(symbol, pos_type):
            logger.error(f"[MEXC] Aborting {symbol} — could not set leverage")
            return {}

        return self._submit_order(symbol, side, vol, sl_price, tp_price, spec)

    def _submit_order(self, symbol: str, side: int, vol: int,
                      sl_price: float, tp_price: float, spec: dict) -> dict:
        """Submit market order; on margin error halve vol and retry once."""
        for attempt in range(2):
            try:
                body = {
                    "symbol":          symbol,
                    "price":           0,
                    "vol":             vol,
                    "leverage":        self.leverage,
                    "side":            side,
                    "type":            5,     # market
                    "openType":        2,     # cross margin
                    "stopLossPrice":   sl_price,
                    "takeProfitPrice": tp_price,
                }
                resp     = self._post("/api/v1/private/order/submit", body)
                order_id = str(resp.get("data", ""))
                self._last_order[symbol] = datetime.utcnow()
                side_label = "LONG" if side == 1 else "SHORT"
                logger.info(
                    f"[MEXC] MARKET ORDER {side_label} {symbol} vol={vol} "
                    f"| SL={sl_price} | TP={tp_price} | id={order_id}"
                )
                return {"order_id": order_id}

            except Exception as e:
                detail = str(e).lower()
                if attempt == 0 and any(
                    k in detail for k in ("insufficient", "not enough", "too large", "exceed")
                ):
                    vol = max(spec["min_vol"], vol // 2)
                    logger.warning(f"[MEXC] order rejected — halving vol to {vol} and retrying")
                    continue
                logger.error(
                    f"[MEXC] place_order({symbol}): {e}\n{traceback.format_exc()}"
                )
                return {}
        return {}

    # ------------------------------------------------------------------
    # Close position (market)
    # ------------------------------------------------------------------

    def close_position(self, symbol: str, direction: str) -> bool:
        if not self.enabled:
            return False
        mexc_sym = self._to_symbol(symbol)
        try:
            data = self._get(
                "/api/v1/private/position/open_positions", {"symbol": mexc_sym}
            )
            vol = 0
            for pos in data.get("data", []):
                v = float(pos.get("holdVol", 0))
                if v > 0:
                    vol = int(v)
                    break
            if vol == 0:
                logger.info(f"[MEXC] close_position: no open position for {mexc_sym}")
                return False
            close_side = 2 if direction == "long" else 4   # 2=close long, 4=close short
            self._post("/api/v1/private/order/submit", {
                "symbol":   mexc_sym,
                "price":    0,
                "vol":      vol,
                "leverage": self.leverage,
                "side":     close_side,
                "type":     5,
                "openType": 2,
            })
            logger.info(f"[MEXC] Position CLOSED {mexc_sym} vol={vol}")
            return True
        except Exception as e:
            logger.error(f"[MEXC] close_position({mexc_sym}): {e}")
            return False

    # ------------------------------------------------------------------
    # Move SL to break-even
    # ------------------------------------------------------------------

    def move_sl_to_breakeven(self, symbol: str, direction: str,
                             entry_price: float) -> bool:
        """Place a trigger order to close the position at entry price (break-even)."""
        if not self.enabled:
            return False
        mexc_sym = self._to_symbol(symbol)
        spec     = self._get_contract(mexc_sym)
        be_price = self._round_price(entry_price, spec["price_scale"])
        try:
            data = self._get(
                "/api/v1/private/position/open_positions", {"symbol": mexc_sym}
            )
            vol = 0
            for pos in data.get("data", []):
                v = float(pos.get("holdVol", 0))
                if v > 0:
                    vol = int(v)
                    break
            if vol == 0:
                logger.info(f"[MEXC] move_sl_to_breakeven: no open position for {mexc_sym}")
                return False

            close_side = 2 if direction == "long" else 4
            self._post("/api/v1/private/planorder/place", {
                "symbol":       mexc_sym,
                "side":         close_side,
                "vol":          vol,
                "type":         2,           # limit trigger
                "triggerPrice": be_price,
                "executePrice": be_price,
                "triggerType":  1,           # triggered by last price
                "openType":     2,
            })
            logger.info(f"[MEXC] BE stop placed at {be_price} for {mexc_sym}")
            return True
        except Exception as e:
            logger.error(f"[MEXC] move_sl_to_breakeven({mexc_sym}): {e}")
            return False
