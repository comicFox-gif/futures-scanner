"""
Bybit Demo/Testnet Order Executor
-----------------------------------
Places real orders on Bybit testnet when confirmed signals fire.
Runs alongside internal paper trading for comparison.

Env vars required:
  BYBIT_KEY    — testnet API key
  BYBIT_SECRET — testnet API secret

Set BYBIT_DEMO=true (default) to use testnet, false for live (not recommended).
"""

from __future__ import annotations
import logging
import os
import ccxt

logger = logging.getLogger("futures_bot.bybit_executor")

# Bybit minimum order quantities (base currency) for common pairs
# ccxt will raise InvalidOrder if below minimum — we catch and skip
_MIN_QTY = {
    "BTC/USDT:USDT": 0.001,
    "ETH/USDT:USDT": 0.01,
    "SOL/USDT:USDT": 0.1,
    "BNB/USDT:USDT": 0.01,
    "XRP/USDT:USDT": 1.0,
}
_DEFAULT_MIN_QTY = 0.1


class BybitExecutor:
    def __init__(self, risk_pct: float = 0.01):
        self.risk_pct = risk_pct
        api_key    = os.getenv("BYBIT_KEY", "")
        api_secret = os.getenv("BYBIT_SECRET", "")
        use_demo   = os.getenv("BYBIT_DEMO", "true").lower() != "false"

        if not api_key or not api_secret:
            logger.warning("BYBIT_KEY / BYBIT_SECRET not set — Bybit executor disabled")
            self.enabled = False
            self.exchange = None
            return

        self.exchange = ccxt.bybit({
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},  # USDT perpetuals
        })
        if use_demo:
            self.exchange.set_sandbox_mode(True)

        self.enabled = True
        mode = "TESTNET" if use_demo else "LIVE"
        logger.info(f"Bybit executor ready ({mode})")

    # ------------------------------------------------------------------
    # Fetch available USDT balance
    # ------------------------------------------------------------------

    def _get_balance(self) -> float:
        try:
            bal = self.exchange.fetch_balance({"type": "linear"})
            return float(bal.get("USDT", {}).get("free", 0) or 0)
        except Exception as e:
            logger.error(f"Bybit fetch_balance failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> bool:
        """
        Place a market order with SL + TP3 on Bybit testnet.
        signal must have: symbol, direction, entry, sl, tp3, atr
        Returns True if order placed successfully.
        """
        if not self.enabled:
            return False

        symbol    = signal["symbol"]
        direction = signal["direction"]
        sl_price  = signal["sl"]
        tp_price  = signal["tp3"]   # use TP3 as the exchange take-profit
        entry     = signal["entry"]
        side      = "buy" if direction == "long" else "sell"

        # Size from risk
        balance     = self._get_balance()
        if balance <= 0:
            logger.warning("Bybit: zero balance, skipping order")
            return False

        sl_dist = abs(entry - sl_price)
        if sl_dist == 0:
            return False

        risk_amount = balance * self.risk_pct
        size        = round(risk_amount / sl_dist, 4)

        # Enforce minimum qty
        min_qty = _MIN_QTY.get(symbol, _DEFAULT_MIN_QTY)
        if size < min_qty:
            size = min_qty

        try:
            params = {
                "stopLoss":   {"triggerPrice": round(sl_price, 6),  "orderType": "Market"},
                "takeProfit": {"triggerPrice": round(tp_price, 6), "orderType": "Market"},
                "positionIdx": 0,  # one-way mode
            }
            order = self.exchange.create_order(symbol, "market", side, size, params=params)
            order_id = order.get("id", "?")
            logger.info(
                f"[BYBIT] ORDER PLACED {direction.upper()} {symbol} "
                f"| Size={size} | SL={sl_price:.4f} | TP={tp_price:.4f} "
                f"| OrderID={order_id} | Balance=${balance:.2f}"
            )
            return True

        except ccxt.InvalidOrder as e:
            logger.warning(f"[BYBIT] Invalid order {symbol}: {e}")
        except ccxt.InsufficientFunds as e:
            logger.warning(f"[BYBIT] Insufficient funds {symbol}: {e}")
        except Exception as e:
            logger.error(f"[BYBIT] Order failed {symbol}: {e}")
        return False

    # ------------------------------------------------------------------
    # Fetch open positions (for status reporting)
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            positions = self.exchange.fetch_positions()
            open_pos  = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
            return open_pos
        except Exception as e:
            logger.error(f"[BYBIT] fetch_positions failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Status summary string
    # ------------------------------------------------------------------

    def status_summary(self) -> str:
        if not self.enabled:
            return "Bybit: disabled"
        positions = self.get_open_positions()
        balance   = self._get_balance()
        if not positions:
            return f"Bybit testnet | Balance: ${balance:.2f} | No open positions"
        lines = []
        for p in positions:
            sym  = p.get("symbol", "?")
            side = p.get("side", "?")
            pnl  = p.get("unrealizedPnl", 0) or 0
            lines.append(f"  {sym} {side} | uPnL: {float(pnl):+.2f}")
        return f"Bybit testnet | ${balance:.2f}\n" + "\n".join(lines)
