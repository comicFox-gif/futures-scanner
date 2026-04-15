"""
Bybit Futures Executor (V5 API)
-----------------------------------
Places real orders on Bybit perpetual futures using the pybit library.
Drop-in replacement for MexcExecutor — identical public interface.

Env vars:
  BYBIT_KEY      — API key
  BYBIT_SECRET   — API secret
  BYBIT_LEVERAGE — leverage per trade (default 10)
  BYBIT_MAX_RISK — max risk in USDT per trade (default 10)

Risk sizing:
  risk_usdt     = from signal dict (score-based $5/$7/$10/$12/$15)
  qty_coin      = risk_usdt / sl_dist
  notional_cap  = balance * leverage * 0.80

Bybit V5 API:
  Library     : pybit>=5.6.0 (unified_trading.HTTP)
  Category    : "linear" (USDT perpetuals)
  SL/TP       : set at order creation, triggered by last price
  Trail stop  : update via set_trading_stop after position opens
"""

from __future__ import annotations
import logging
import math
import time
import traceback
from datetime import datetime, timedelta

logger = logging.getLogger("futures_bot.bybit")


def _to_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' → 'BTCUSDT'"""
    return symbol.split(":")[0].replace("/", "")


def _round_to_step(value: float, step: float) -> float:
    """Round value down to nearest multiple of step."""
    if step <= 0:
        return value
    factor = 1.0 / step
    return math.floor(value * factor) / factor


class BybitExecutor:
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
        self._instrument_cache: dict[str, dict] = {}
        self._last_order:       dict[str, datetime] = {}
        self._order_cooldown_min = 20
        self._session = None

        if not self.enabled:
            logger.info("[BYBIT] No API keys — executor disabled")
            return

        try:
            from pybit.unified_trading import HTTP
            self._session = HTTP(
                testnet=False,
                api_key=api_key,
                api_secret=api_secret,
            )
            masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "???"
            logger.info(f"[BYBIT] Executor ready | key={masked} | leverage={leverage}x")
        except Exception as e:
            logger.error(f"[BYBIT] Failed to initialise session: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Symbol + instrument specs
    # ------------------------------------------------------------------

    def _get_instrument(self, symbol: str) -> dict:
        """Fetch and cache instrument info (tick size, qty step, min qty)."""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        try:
            resp = self._session.get_instruments_info(category="linear", symbol=symbol)
            info = resp["result"]["list"][0]
            lot  = info["lotSizeFilter"]
            price = info["priceFilter"]
            spec = {
                "qty_step":  float(lot["qtyStep"]),
                "min_qty":   float(lot["minOrderQty"]),
                "max_qty":   float(lot.get("maxOrderQty", 999999)),
                "tick_size": float(price["tickSize"]),
            }
            logger.info(
                f"[BYBIT] {symbol} spec — qtyStep={spec['qty_step']} "
                f"minQty={spec['min_qty']} tickSize={spec['tick_size']}"
            )
        except Exception as e:
            logger.warning(f"[BYBIT] get_instrument({symbol}) failed: {e} — using defaults")
            spec = {"qty_step": 0.001, "min_qty": 0.001, "max_qty": 999999, "tick_size": 0.01}
        self._instrument_cache[symbol] = spec
        return spec

    def _round_price(self, price: float, tick: float) -> str:
        """Round price to the instrument's tick size, returned as string."""
        if tick <= 0 or price <= 0:
            return str(round(price, 8))
        factor = 1.0 / tick
        rounded = round(math.floor(price * factor) / factor, 10)
        # Determine decimal places from tick
        tick_str = f"{tick:.10f}".rstrip("0")
        decimals = len(tick_str.split(".")[-1]) if "." in tick_str else 0
        return f"{rounded:.{decimals}f}"

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def _get_balance(self) -> float:
        """Return available USDT balance from Unified or Contract account."""
        try:
            for acct_type in ("UNIFIED", "CONTRACT"):
                try:
                    resp  = self._session.get_wallet_balance(accountType=acct_type, coin="USDT")
                    coins = resp["result"]["list"][0].get("coin", [])
                    for c in coins:
                        if c.get("coin") == "USDT":
                            avail = c.get("availableToWithdraw") or c.get("walletBalance", 0)
                            return float(avail)
                except Exception:
                    continue
            return 100.0
        except Exception as e:
            logger.warning(f"[BYBIT] get_balance failed: {e} — using 100 fallback")
            return 100.0

    # ------------------------------------------------------------------
    # Position check
    # ------------------------------------------------------------------

    def has_open_position(self, symbol: str) -> bool:
        try:
            resp = self._session.get_positions(category="linear", symbol=symbol)
            for pos in resp["result"]["list"]:
                if float(pos.get("size", 0)) > 0:
                    logger.info(
                        f"[BYBIT] Skipping {symbol} — position already open "
                        f"(size={pos['size']})"
                    )
                    return True
            return False
        except Exception as e:
            logger.warning(f"[BYBIT] has_open_position({symbol}): {e} — assuming no position")
            return False

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def _set_leverage(self, symbol: str) -> bool:
        try:
            lev_str = str(self.leverage)
            self._session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=lev_str,
                sellLeverage=lev_str,
            )
            logger.info(f"[BYBIT] Leverage set to {self.leverage}x for {symbol}")
            return True
        except Exception as e:
            msg = str(e).lower()
            if "leverage not modified" in msg or "110043" in msg:
                return True   # already at correct leverage
            logger.warning(f"[BYBIT] set_leverage({symbol}): {e}")
            return False

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> dict:
        """
        Place a market entry with SL + TP on Bybit (USDT perpetual).
        Returns {"order_id": str} or {} on failure.
        """
        if not self.enabled:
            return {}

        symbol = _to_symbol(signal["symbol"])

        if self.has_open_position(symbol):
            return {}

        last = self._last_order.get(symbol)
        if last and datetime.utcnow() - last < timedelta(minutes=self._order_cooldown_min):
            remaining = int(
                (timedelta(minutes=self._order_cooldown_min) -
                 (datetime.utcnow() - last)).total_seconds() / 60
            )
            logger.info(f"[BYBIT] Skipping {symbol} — cooldown ({remaining}min remaining)")
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

        if direction == "long" and not (sl < entry < tp3):
            logger.warning(f"[BYBIT] Bad levels LONG {symbol}: SL={sl} E={entry} TP={tp3}")
            return {}
        if direction == "short" and not (tp3 < entry < sl):
            logger.warning(f"[BYBIT] Bad levels SHORT {symbol}: TP={tp3} E={entry} SL={sl}")
            return {}

        spec     = self._get_instrument(symbol)
        tick     = spec["tick_size"]

        # Qty sizing: risk_usdt / sl_dist → rounded to qtyStep
        risk_usdt = float(signal.get("risk_usdt", 5.0))
        qty_coin  = risk_usdt / sl_dist
        qty       = _round_to_step(qty_coin, spec["qty_step"])

        # Notional cap: never exceed 80% of available margin × leverage
        balance      = self._get_balance()
        max_notional = balance * self.leverage * 0.80
        qty_cap      = _round_to_step(max_notional / entry, spec["qty_step"])
        if qty > qty_cap:
            logger.warning(
                f"[BYBIT] {symbol} qty={qty} notional≈{qty * entry:.2f} "
                f"exceeds cap (bal={balance:.2f} lev={self.leverage}x) — capping to {qty_cap}"
            )
            qty = qty_cap

        qty = max(spec["min_qty"], min(spec["max_qty"], qty))
        if qty <= 0:
            logger.warning(f"[BYBIT] qty rounds to 0 for {symbol} — skipping")
            return {}

        sl_str  = self._round_price(sl, tick)
        tp_str  = self._round_price(tp3, tick)
        qty_str = f"{qty:.{self._decimal_places(spec['qty_step'])}f}"

        notional       = qty * entry
        effective_risk = qty * sl_dist
        logger.info(
            f"[BYBIT] {symbol} | Bal=${balance:.2f} | Risk≈${effective_risk:.2f} "
            f"| qty={qty_str} | notional≈{notional:.2f} | SL={sl_str} TP={tp_str}"
        )

        if not self._set_leverage(symbol):
            logger.error(f"[BYBIT] Aborting {symbol} — could not set leverage")
            return {}

        return self._submit_order(symbol, side, qty_str, sl_str, tp_str, spec)

    @staticmethod
    def _decimal_places(step: float) -> int:
        step_str = f"{step:.10f}".rstrip("0")
        return len(step_str.split(".")[-1]) if "." in step_str else 0

    def _submit_order(self, symbol: str, side: str, qty_str: str,
                      sl_str: str, tp_str: str, spec: dict) -> dict:
        """Submit market order with SL/TP attached at creation."""
        for attempt in range(2):
            try:
                resp = self._session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=qty_str,
                    stopLoss=sl_str,
                    takeProfit=tp_str,
                    slTriggerBy="LastPrice",
                    tpTriggerBy="LastPrice",
                    timeInForce="IOC",
                    reduceOnly=False,
                )
                order_id = resp["result"].get("orderId", "")
                self._last_order[symbol] = datetime.utcnow()
                logger.info(
                    f"[BYBIT] ORDER {side.upper()} {symbol} qty={qty_str} "
                    f"| SL={sl_str} | TP={tp_str} | id={order_id}"
                )
                return {"order_id": order_id}

            except Exception as e:
                detail = str(e).lower()
                if attempt == 0 and any(
                    k in detail for k in ("insufficient", "not enough", "too large", "exceed", "110007")
                ):
                    # Halve qty and retry once
                    qty_val = float(qty_str)
                    qty_val = max(spec["min_qty"],
                                  _round_to_step(qty_val / 2, spec["qty_step"]))
                    qty_str = f"{qty_val:.{self._decimal_places(spec['qty_step'])}f}"
                    logger.warning(f"[BYBIT] order rejected — halving qty to {qty_str} and retrying")
                    continue
                logger.error(
                    f"[BYBIT] place_order({symbol}): {e}\n{traceback.format_exc()}"
                )
                return {}
        return {}

    # ------------------------------------------------------------------
    # Close position (market)
    # ------------------------------------------------------------------

    def close_position(self, symbol: str, direction: str) -> bool:
        if not self.enabled:
            return False
        bybit_sym = _to_symbol(symbol)
        try:
            resp = self._session.get_positions(category="linear", symbol=bybit_sym)
            size = 0.0
            for pos in resp["result"]["list"]:
                s = float(pos.get("size", 0))
                if s > 0:
                    size = s
                    break
            if size == 0:
                logger.info(f"[BYBIT] close_position: no open position for {bybit_sym}")
                return False

            close_side = "Sell" if direction == "long" else "Buy"
            spec       = self._get_instrument(bybit_sym)
            qty_str    = f"{size:.{self._decimal_places(spec['qty_step'])}f}"
            self._session.place_order(
                category="linear",
                symbol=bybit_sym,
                side=close_side,
                orderType="Market",
                qty=qty_str,
                reduceOnly=True,
                timeInForce="IOC",
            )
            logger.info(f"[BYBIT] Position CLOSED {bybit_sym} qty={qty_str}")
            return True
        except Exception as e:
            logger.error(f"[BYBIT] close_position({bybit_sym}): {e}")
            return False

    # ------------------------------------------------------------------
    # Update trailing stop-loss
    # ------------------------------------------------------------------

    def update_trail_sl(self, symbol: str, direction: str,
                        new_sl: float, tp: float) -> bool:
        """
        Update the trailing stop-loss for an open Bybit position.
        Uses set_trading_stop to atomically replace SL/TP on the position.
        """
        if not self.enabled:
            return False
        bybit_sym = _to_symbol(symbol)
        spec      = self._get_instrument(bybit_sym)
        tick      = spec["tick_size"]
        sl_str    = self._round_price(new_sl, tick)
        tp_str    = self._round_price(tp, tick)

        try:
            self._session.set_trading_stop(
                category="linear",
                symbol=bybit_sym,
                stopLoss=sl_str,
                takeProfit=tp_str,
                slTriggerBy="LastPrice",
                tpTriggerBy="LastPrice",
                positionIdx=0,   # one-way mode
            )
            logger.info(
                f"[BYBIT] Trail SL updated | {bybit_sym} | new SL={sl_str} | TP={tp_str}"
            )
            return True
        except Exception as e:
            logger.error(f"[BYBIT] update_trail_sl({bybit_sym}): {e}")
            return False

    # ------------------------------------------------------------------
    # Move SL to break-even
    # ------------------------------------------------------------------

    def move_sl_to_breakeven(self, symbol: str, direction: str,
                             entry_price: float) -> bool:
        """Move SL to entry price (break-even) using set_trading_stop."""
        if not self.enabled:
            return False
        bybit_sym = _to_symbol(symbol)
        spec      = self._get_instrument(bybit_sym)
        be_str    = self._round_price(entry_price, spec["tick_size"])

        try:
            self._session.set_trading_stop(
                category="linear",
                symbol=bybit_sym,
                stopLoss=be_str,
                slTriggerBy="LastPrice",
                positionIdx=0,
            )
            logger.info(f"[BYBIT] BE stop placed at {be_str} for {bybit_sym}")
            return True
        except Exception as e:
            logger.error(f"[BYBIT] move_sl_to_breakeven({bybit_sym}): {e}")
            return False
