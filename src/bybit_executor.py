"""
Bybit Testnet Order Executor
------------------------------
Places real orders on Bybit testnet when confirmed signals fire.
Runs alongside internal paper trading.

Env vars required (set in Railway → Variables):
  BYBIT_KEY    — testnet API key
  BYBIT_SECRET — testnet API secret
  BYBIT_DEMO   — "true" (default) uses testnet, "false" uses live
"""

from __future__ import annotations
import logging
import os
import ccxt

logger = logging.getLogger("futures_bot.bybit")

# Minimum contract sizes for Bybit linear perpetuals
_MIN_QTY = {
    "BTC/USDT:USDT":  0.001,
    "ETH/USDT:USDT":  0.01,
    "SOL/USDT:USDT":  0.1,
    "BNB/USDT:USDT":  0.01,
    "XRP/USDT:USDT":  1.0,
    "ADA/USDT:USDT":  1.0,
    "DOGE/USDT:USDT": 1.0,
    "MATIC/USDT:USDT":1.0,
}
_DEFAULT_MIN_QTY = 1.0


class BybitExecutor:
    def __init__(self, risk_pct: float = 0.01):
        self.risk_pct = risk_pct
        api_key    = os.getenv("BYBIT_KEY", "")
        api_secret = os.getenv("BYBIT_SECRET", "")
        use_testnet = os.getenv("BYBIT_DEMO", "true").lower() != "false"

        if not api_key or not api_secret:
            logger.warning("BYBIT_KEY / BYBIT_SECRET not set in env — Bybit disabled")
            self.enabled  = False
            self.exchange = None
            return

        self.exchange = ccxt.bybit({
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType":    "linear",
                "defaultSubType": "linear",
            },
        })

        if use_testnet:
            self.exchange.set_sandbox_mode(True)

        # Verify connection
        try:
            self.exchange.load_markets()
            self.enabled = True
            mode = "TESTNET" if use_testnet else "LIVE"
            logger.info(f"Bybit executor connected ({mode})")
        except Exception as e:
            logger.error(f"Bybit connection failed: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def _get_balance(self) -> float:
        try:
            bal = self.exchange.fetch_balance(params={"category": "linear"})
            usdt = bal.get("USDT") or bal.get("usdt") or {}
            free = usdt.get("free") or usdt.get("total") or 0
            return float(free or 0)
        except Exception as e:
            logger.error(f"Bybit fetch_balance error: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> bool:
        """
        Market order with SL + TP on Bybit testnet.
        signal keys: symbol, direction, entry, sl, tp3
        """
        if not self.enabled:
            return False

        symbol    = signal["symbol"]
        direction = signal["direction"]
        entry     = float(signal["entry"])
        sl_price  = float(signal["sl"])
        tp_price  = float(signal["tp3"])
        side      = "buy" if direction == "long" else "sell"

        # Validate prices make sense
        if direction == "long" and not (sl_price < entry < tp_price):
            logger.warning(f"[BYBIT] Invalid levels for LONG {symbol}: SL={sl_price} entry={entry} TP={tp_price}")
            return False
        if direction == "short" and not (tp_price < entry < sl_price):
            logger.warning(f"[BYBIT] Invalid levels for SHORT {symbol}: TP={tp_price} entry={entry} SL={sl_price}")
            return False

        sl_dist = abs(entry - sl_price)
        if sl_dist == 0:
            return False

        balance = self._get_balance()
        if balance < 1:
            logger.warning(f"[BYBIT] Balance too low: ${balance:.2f}")
            return False

        # Position size
        risk_amount = balance * self.risk_pct
        raw_size    = risk_amount / sl_dist
        min_qty     = _MIN_QTY.get(symbol, _DEFAULT_MIN_QTY)
        size        = max(round(raw_size, 3), min_qty)

        try:
            # Bybit v5 unified params — plain numbers for SL/TP
            params = {
                "category":   "linear",
                "positionIdx": 0,          # one-way mode
                "stopLoss":   str(round(sl_price, 6)),
                "takeProfit": str(round(tp_price, 6)),
                "slTriggerBy": "LastPrice",
                "tpTriggerBy": "LastPrice",
            }
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=size,
                params=params,
            )
            order_id = order.get("id", "?")
            logger.info(
                f"[BYBIT] ✅ {direction.upper()} {symbol} | "
                f"Size={size} | SL={sl_price:.5f} | TP={tp_price:.5f} | "
                f"Balance=${balance:.2f} | ID={order_id}"
            )
            return True

        except ccxt.InvalidOrder as e:
            logger.error(f"[BYBIT] InvalidOrder {symbol}: {e}")
        except ccxt.InsufficientFunds as e:
            logger.error(f"[BYBIT] InsufficientFunds {symbol}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"[BYBIT] ExchangeError {symbol}: {e}")
        except Exception as e:
            logger.error(f"[BYBIT] Unexpected error {symbol}: {e}")
        return False

    # ------------------------------------------------------------------
    # Open positions summary
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            positions = self.exchange.fetch_positions(params={"category": "linear"})
            return [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
        except Exception as e:
            logger.error(f"[BYBIT] fetch_positions error: {e}")
            return []

    def status_summary(self) -> str:
        if not self.enabled:
            return "Bybit: disabled (check BYBIT_KEY/BYBIT_SECRET env vars)"
        balance   = self._get_balance()
        positions = self.get_open_positions()
        if not positions:
            return f"Bybit testnet | Balance: ${balance:.2f} | No open positions"
        lines = [f"Bybit testnet | Balance: ${balance:.2f}"]
        for p in positions:
            sym = p.get("symbol", "?")
            side = p.get("side", "?")
            pnl  = float(p.get("unrealizedPnl", 0) or 0)
            lines.append(f"  {sym} {side} | uPnL: {pnl:+.2f}")
        return "\n".join(lines)
