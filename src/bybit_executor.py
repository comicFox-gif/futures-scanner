"""
Bybit Futures Executor (pybit SDK)
------------------------------------
Places real orders on Bybit demo/testnet when confirmed signals fire.
Uses official pybit library — entry + SL + TP in a single place_order() call.

pip install pybit

Env vars:
  BYBIT_KEY      — Bybit API key
  BYBIT_SECRET   — Bybit API secret
  BYBIT_DEMO     — "true" (default) → api-demo.bybit.com
  BYBIT_TESTNET  — "true" → api-testnet.bybit.com
  BYBIT_LEVERAGE — default 10

Risk sizing:
  risk_usdt = balance * risk_pct
  qty       = risk_usdt / sl_dist          (coins to lose exactly risk_usdt if SL hit)
  cap       = balance * leverage * 0.8     (never exceed 80% of available margin notional)
  qty       = min(qty, cap / entry_price)  (apply cap)
  qty       = rounded to instrument qtyStep, clamped to [minOrderQty, maxOrderQty]
"""

from __future__ import annotations
import logging
import math
import traceback
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("futures_bot.bybit")

DEMO_HOST    = "https://api-demo.bybit.com"
TESTNET_HOST = "https://api-testnet.bybit.com"
LIVE_HOST    = "https://api.bybit.com"


class BybitExecutor:
    def __init__(self, api_key: str = "", api_secret: str = "",
                 demo: bool = True, testnet: bool = False,
                 leverage: int = 10, risk_pct: float = 0.01,
                 max_positions: int = 50, max_risk_usdt: float = 10.0):
        self.leverage      = leverage
        self.risk_pct      = risk_pct
        self.max_positions = max_positions
        self.max_risk_usdt = max_risk_usdt
        self.enabled  = bool(api_key and api_secret)
        self._instrument_cache: dict[str, dict] = {}
        self._last_order: dict[str, datetime] = {}   # symbol → last order time
        self._order_cooldown_min = 20

        if not self.enabled:
            logger.info("[BYBIT] No API keys — executor disabled")
            return

        try:
            from pybit.unified_trading import HTTP
            if demo:
                env_label = "DEMO"
                host      = DEMO_HOST
            elif testnet:
                env_label = "TESTNET"
                host      = TESTNET_HOST
            else:
                env_label = "LIVE"
                host      = LIVE_HOST

            self.session = HTTP(
                api_key=api_key,
                api_secret=api_secret,
                demo=demo,
                testnet=(testnet and not demo),
            )
            masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "???"
            logger.info(
                f"[BYBIT] Executor ready | {env_label} ({host}) | key={masked_key} | "
                f"leverage={leverage}x | risk={risk_pct*100:.1f}% per trade | max_risk=${max_risk_usdt:.0f}"
            )
            self._set_position_mode()
        except ImportError:
            logger.error("[BYBIT] pybit not installed — run: pip install pybit")
            self.enabled = False
        except Exception as e:
            logger.error(f"[BYBIT] Init failed: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Instrument specs (cached per symbol)
    # ------------------------------------------------------------------

    def _get_instrument_info(self, symbol: str) -> dict:
        """Fetch and cache lotSizeFilter + priceFilter for a symbol."""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        try:
            resp = self.session.get_instruments_info(category="linear", symbol=symbol)
            info = resp["result"]["list"][0]
            lot  = info["lotSizeFilter"]
            pf   = info["priceFilter"]
            spec = {
                "min_qty":   float(lot["minOrderQty"]),
                "max_qty":   float(lot["maxOrderQty"]),
                "qty_step":  float(lot["qtyStep"]),
                "tick_size": float(pf["tickSize"]),
            }
            logger.info(
                f"[BYBIT] {symbol} spec — minQty={spec['min_qty']} maxQty={spec['max_qty']} "
                f"step={spec['qty_step']} tick={spec['tick_size']}"
            )
        except Exception as e:
            logger.warning(f"[BYBIT] get_instrument_info({symbol}) failed: {e} — using defaults")
            spec = {"min_qty": 1.0, "max_qty": 9e9, "qty_step": 1.0, "tick_size": 0.0001}
        self._instrument_cache[symbol] = spec
        return spec

    @staticmethod
    def _decimal_places(value: float) -> int:
        s = f"{value:.10f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0

    def _round_qty(self, qty: float, spec: dict) -> float:
        """Floor qty to nearest qtyStep, clamp to [minOrderQty, maxOrderQty]."""
        step = spec["qty_step"]
        qty  = math.floor(qty / step) * step
        qty  = max(spec["min_qty"], min(spec["max_qty"], qty))
        return round(qty, self._decimal_places(step))

    def _round_price(self, price: float, spec: dict) -> str:
        """Round price to nearest tick_size and return as string."""
        tick     = spec["tick_size"]
        price    = round(round(price / tick) * tick, 8)
        decimals = self._decimal_places(tick)
        return f"{price:.{decimals}f}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_symbol(self, symbol: str) -> str:
        """'BTC/USDT:USDT' → 'BTCUSDT'"""
        return symbol.split(":")[0].replace("/", "")

    def _get_balance(self) -> float:
        """Fetch available USDT balance from Bybit unified account."""
        try:
            resp  = self.session.get_wallet_balance(accountType="UNIFIED")
            coins = resp["result"]["list"][0]["coin"]
            for c in coins:
                if c["coin"] == "USDT":
                    return float(c["availableToWithdraw"] or c["walletBalance"] or 0)
            return 0.0
        except Exception as e:
            logger.warning(f"[BYBIT] get_balance failed: {e} — using 431 fallback")
            return 431.0

    def has_open_position(self, symbol: str) -> bool:
        """Return True if there is already an open position for this symbol."""
        try:
            resp = self.session.get_positions(category="linear", symbol=symbol)
            for pos in resp["result"]["list"]:
                if float(pos.get("size", 0)) > 0:
                    logger.info(f"[BYBIT] Skipping {symbol} — position already open (size={pos['size']})")
                    return True
            return False
        except Exception as e:
            logger.warning(f"[BYBIT] get_positions({symbol}): {e} — assuming no position")
            return False

    def _set_position_mode(self):
        """Switch account to one-way mode (positionIdx=0). Run once on init."""
        try:
            self.session.switch_position_mode(category="linear", coin="USDT", mode=0)
            logger.info("[BYBIT] Position mode set to one-way")
        except Exception as e:
            # Error 110025 = already in one-way mode — fine
            if "110025" in str(e) or "already" in str(e).lower():
                logger.info("[BYBIT] Already in one-way mode")
            else:
                logger.warning(f"[BYBIT] switch_position_mode: {e}")

    def _set_leverage(self, symbol: str) -> bool:
        """Set leverage for the symbol. Returns True on success."""
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage),
            )
            logger.info(f"[BYBIT] Leverage set to {self.leverage}x for {symbol}")
            return True
        except Exception as e:
            if "leverage not modified" in str(e).lower() or "110043" in str(e):
                logger.info(f"[BYBIT] Leverage already {self.leverage}x for {symbol}")
                return True
            logger.warning(f"[BYBIT] set_leverage({symbol}): {e}")
            return False

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> dict:
        """
        Place market entry + SL + TP3 on Bybit in a single API call.
        Returns {"order_id": str} or {} on failure.
        """
        if not self.enabled:
            return {}

        symbol    = self._to_symbol(signal["symbol"])

        if self.has_open_position(symbol):
            return {}

        last = self._last_order.get(symbol)
        if last and datetime.utcnow() - last < timedelta(minutes=self._order_cooldown_min):
            remaining = int((timedelta(minutes=self._order_cooldown_min) - (datetime.utcnow() - last)).total_seconds() / 60)
            logger.info(f"[BYBIT] Skipping {symbol} — order cooldown ({remaining}min remaining)")
            return {}

        direction = signal["direction"]
        entry     = float(signal["entry"])
        sl        = float(signal["sl"])
        tp3       = float(signal["tp3"])
        side      = "Buy" if direction == "long" else "Sell"

        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            logger.warning(f"[BYBIT] Zero SL distance for {symbol} — skipping")
            return {}

        # Direction sanity check
        if direction == "long" and not (sl < entry < tp3):
            logger.warning(f"[BYBIT] Bad levels LONG {symbol}: SL={sl} E={entry} TP={tp3}")
            return {}
        if direction == "short" and not (tp3 < entry < sl):
            logger.warning(f"[BYBIT] Bad levels SHORT {symbol}: TP={tp3} E={entry} SL={sl}")
            return {}

        risk_usdt = 4.0  # fixed $4 risk per trade

        # qty = exactly the number of coins that loses $4 if SL is hit
        qty = risk_usdt / sl_dist

        # --- Round to instrument step size and clamp to [min, maxOrderQty] ---
        spec = self._get_instrument_info(symbol)
        qty  = self._round_qty(qty, spec)

        if qty <= 0:
            logger.warning(f"[BYBIT] qty rounds to 0 for {symbol} — skipping")
            return {}

        sl_price  = self._round_price(sl, spec)
        tp_price  = self._round_price(tp3, spec)
        notional  = qty * entry
        effective_risk = qty * sl_dist

        logger.info(
            f"[BYBIT] {symbol} | Risk=${effective_risk:.2f} "
            f"| qty={qty} | notional≈{notional:.2f} | SL={sl_price} TP={tp_price}"
        )

        if not self._set_leverage(symbol):
            logger.error(f"[BYBIT] Aborting {symbol} — could not set leverage")
            return {}

        # Limit price: 2 ticks inside market for near-instant fill with maker fees
        tick = spec["tick_size"]
        if side == "Buy":
            limit_price = self._round_price(entry - tick * 2, spec)
        else:
            limit_price = self._round_price(entry + tick * 2, spec)

        return self._submit_order(symbol, side, qty, limit_price, sl_price, tp_price, spec)

    def _submit_order(self, symbol: str, side: str, qty: float,
                      limit_price: str, sl_price: str, tp_price: str, spec: dict) -> dict:
        """Submit a limit order 2 ticks from market; on 'too large' halve qty and retry once."""
        for attempt in range(2):
            try:
                resp = self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Limit",
                    price=limit_price,
                    timeInForce="GTC",
                    qty=str(qty),
                    positionIdx=0,
                    stopLoss=sl_price,
                    takeProfit=tp_price,
                    tpslMode="Full",
                    slOrderType="Market",
                    tpOrderType="Market",
                )
                order_id = resp["result"]["orderId"]
                self._last_order[symbol] = datetime.utcnow()
                logger.info(
                    f"[BYBIT] LIMIT ORDER {side.upper()} {symbol} qty={qty} "
                    f"@ {limit_price} | SL={sl_price} | TP={tp_price} | id={order_id}"
                )
                return {"order_id": order_id}

            except Exception as e:
                detail = str(getattr(e, "message", e))
                if attempt == 0 and ("too large" in detail.lower() or "exceeds maximum" in detail.lower()):
                    qty = self._round_qty(qty / 2, spec)
                    logger.warning(f"[BYBIT] qty too large — halving to {qty} and retrying")
                    continue
                logger.error(f"[BYBIT] place_order({symbol}): {detail}\n{traceback.format_exc()}")
                return {}
        return {}

    # ------------------------------------------------------------------
    # Move SL to break-even (called when TP2 hit)
    # ------------------------------------------------------------------

    def close_position(self, symbol: str, direction: str) -> bool:
        """Market-close an open position immediately (whale exit / manual close)."""
        if not self.enabled:
            return False
        bybit_symbol = self._to_symbol(symbol)
        try:
            resp = self.session.get_positions(category="linear", symbol=bybit_symbol)
            size = 0.0
            for pos in resp["result"]["list"]:
                s = float(pos.get("size", 0))
                if s > 0:
                    size = s
                    break
            if size == 0:
                logger.info(f"[BYBIT] close_position: no open position for {bybit_symbol}")
                return False
            close_side = "Sell" if direction == "long" else "Buy"
            self.session.place_order(
                category="linear",
                symbol=bybit_symbol,
                side=close_side,
                orderType="Market",
                qty=str(size),
                positionIdx=0,
                reduceOnly=True,
            )
            logger.info(f"[BYBIT] Position CLOSED {bybit_symbol} size={size} (whale exit)")
            return True
        except Exception as e:
            logger.error(f"[BYBIT] close_position({bybit_symbol}): {e}")
            return False

    def move_sl_to_breakeven(self, symbol: str, direction: str, entry_price: float) -> bool:
        """Move SL to entry price (break-even) using set_trading_stop."""
        if not self.enabled:
            return False

        bybit_symbol = self._to_symbol(symbol)
        spec         = self._get_instrument_info(bybit_symbol)
        be_price     = self._round_price(entry_price, spec)
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=bybit_symbol,
                stopLoss=be_price,
                positionIdx=0,
                tpslMode="Full",
            )
            logger.info(f"[BYBIT] BE SL moved to {be_price} for {bybit_symbol}")
            return True
        except Exception as e:
            logger.error(f"[BYBIT] move_sl_to_breakeven({bybit_symbol}): {e}")
            return False
