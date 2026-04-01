"""
Order Manager
--------------
Handles real exchange orders via ccxt AND paper trading simulation.
All public methods work identically in both modes.
"""

from __future__ import annotations
import uuid
import time
import logging
from typing import Optional

logger = logging.getLogger("futures_bot.orders")


class OrderManager:
    def __init__(self, exchange, paper_trading: bool, paper_balance: float):
        self.exchange = exchange
        self.paper = paper_trading
        # Paper trading state
        self._paper_balance = paper_balance
        self._paper_positions: dict = {}   # symbol -> position dict
        self._paper_orders: dict = {}

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        if self.paper:
            return self._paper_balance
        try:
            bal = self.exchange.fetch_balance()
            return float(bal["USDT"]["free"])
        except Exception as e:
            logger.error(f"fetch_balance failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Market orders
    # ------------------------------------------------------------------

    def open_market(
        self,
        symbol: str,
        side: str,           # "buy" | "sell"
        size: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        leverage: int = 10,
    ) -> dict:
        """
        Opens a market order with SL.
        Returns order dict with at least: id, symbol, side, size, price, status
        """
        if self.paper:
            return self._paper_open(symbol, side, size, stop_loss, take_profit)

        try:
            # Set leverage first
            self.exchange.set_leverage(leverage, symbol)
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=size,
                params={
                    "stopLoss": {
                        "triggerPrice": stop_loss,
                        "type": "market",
                    },
                },
            )
            logger.info(f"Opened {side} {size} {symbol} @ market | SL={stop_loss} | id={order['id']}")
            return order
        except Exception as e:
            logger.error(f"open_market failed for {symbol}: {e}")
            raise

    def close_partial(self, symbol: str, side: str, size: float) -> dict:
        """Close `size` units of an existing position."""
        close_side = "sell" if side == "buy" else "buy"

        if self.paper:
            return self._paper_close_partial(symbol, size, close_side)

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=close_side,
                amount=size,
                params={"reduceOnly": True},
            )
            logger.info(f"Partial close {size} {symbol} | id={order['id']}")
            return order
        except Exception as e:
            logger.error(f"close_partial failed for {symbol}: {e}")
            raise

    def close_all(self, symbol: str, side: str, size: float) -> dict:
        """Close entire position."""
        return self.close_partial(symbol, side, size)

    def update_stop_loss(self, symbol: str, order_id: str, new_sl: float, size: float, side: str) -> bool:
        """Cancel existing SL and place new one (or update if exchange supports it)."""
        if self.paper:
            return self._paper_update_sl(symbol, new_sl)

        try:
            # Most exchanges: cancel old SL and place new one
            try:
                self.exchange.cancel_order(order_id, symbol)
            except Exception:
                pass  # May already be cancelled

            close_side = "sell" if side == "buy" else "buy"
            self.exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=close_side,
                amount=size,
                params={
                    "stopPrice": new_sl,
                    "reduceOnly": True,
                },
            )
            logger.info(f"Updated SL for {symbol} to {new_sl}")
            return True
        except Exception as e:
            logger.error(f"update_stop_loss failed for {symbol}: {e}")
            return False

    def get_current_price(self, symbol: str) -> float:
        """Get latest mark/mid price."""
        if self.paper:
            # In paper mode, bot.py fetches candle data directly
            return 0.0
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.error(f"get_current_price failed for {symbol}: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Paper trading internals
    # ------------------------------------------------------------------

    def _paper_open(self, symbol: str, side: str, size: float, sl: float, tp: Optional[float]) -> dict:
        oid = str(uuid.uuid4())[:8]
        self._paper_orders[oid] = {
            "id": oid, "symbol": symbol, "side": side,
            "size": size, "sl": sl, "tp": tp,
            "status": "open", "price": None,   # filled on first price check
        }
        logger.info(f"[PAPER] Queued {side} {size} {symbol} | SL={sl}")
        return {"id": oid, "symbol": symbol, "side": side, "size": size, "status": "open"}

    def paper_fill_at(self, order_id: str, price: float):
        """Mark a paper order as filled at given price."""
        if order_id in self._paper_orders:
            self._paper_orders[order_id]["price"] = price
            self._paper_orders[order_id]["status"] = "filled"

    def _paper_close_partial(self, symbol: str, size: float, close_side: str) -> dict:
        oid = str(uuid.uuid4())[:8]
        logger.info(f"[PAPER] Close partial {size} {symbol}")
        return {"id": oid, "symbol": symbol, "side": close_side, "size": size, "status": "filled"}

    def _paper_update_sl(self, symbol: str, new_sl: float) -> bool:
        logger.info(f"[PAPER] Updated SL for {symbol} to {new_sl}")
        return True

    def update_paper_balance(self, pnl: float):
        self._paper_balance += pnl
        logger.info(f"[PAPER] PnL: {pnl:+.2f} | Balance: {self._paper_balance:.2f}")
