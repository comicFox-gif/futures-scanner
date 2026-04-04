"""
OKX Demo Trading Executor
---------------------------
Places real orders on OKX simulated/demo trading when confirmed signals fire.
Uses the same OKX API keys already configured — just adds x-simulated-trading header.
No geo-blocking issues since OKX already works on Railway.

Env vars (same ones already set for the scanner):
  API_KEY        — OKX API key
  API_SECRET     — OKX API secret
  API_PASSPHRASE — OKX passphrase

OKX simulated trading runs on real market data with virtual funds.
"""

from __future__ import annotations
import logging
import os
import ccxt

logger = logging.getLogger("futures_bot.demo_executor")

# OKX minimum order sizes (contracts) for USDT swap pairs
_MIN_QTY = {
    "BTC/USDT:USDT":  0.01,
    "ETH/USDT:USDT":  0.1,
    "SOL/USDT:USDT":  1.0,
    "BNB/USDT:USDT":  0.1,
    "XRP/USDT:USDT":  10.0,
    "ADA/USDT:USDT":  10.0,
    "DOGE/USDT:USDT": 10.0,
}
_DEFAULT_MIN_QTY = 1.0


class BybitExecutor:  # keep class name so bot.py import doesn't break
    """OKX demo trading executor (Bybit was geo-blocked on Railway)."""

    def __init__(self, risk_pct: float = 0.01):
        self.risk_pct = risk_pct
        api_key    = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        passphrase = os.getenv("API_PASSPHRASE", "")

        if not api_key or not api_secret or not passphrase:
            logger.warning("OKX API keys not set — demo executor disabled")
            self.enabled  = False
            self.exchange = None
            return

        # x-simulated-trading: 1 switches OKX to demo/paper mode
        self.exchange = ccxt.okx({
            "apiKey":    api_key,
            "secret":    api_secret,
            "password":  passphrase,
            "enableRateLimit": True,
            "options":   {"defaultType": "swap"},
            "headers":   {"x-simulated-trading": "1"},
        })

        try:
            self.exchange.load_markets()
            self.enabled = True
            logger.info("OKX demo executor connected (simulated trading ON)")
        except Exception as e:
            logger.error(f"OKX demo connection failed: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def _get_balance(self) -> float:
        try:
            bal  = self.exchange.fetch_balance(params={"type": "swap"})
            usdt = bal.get("USDT", {})
            return float(usdt.get("free") or usdt.get("total") or 0)
        except Exception as e:
            logger.error(f"OKX demo fetch_balance error: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> bool:
        """
        Market order with SL + TP on OKX demo trading.
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

        # Sanity check
        if direction == "long" and not (sl_price < entry < tp_price):
            logger.warning(f"[DEMO] Bad levels LONG {symbol}: SL={sl_price} E={entry} TP={tp_price}")
            return False
        if direction == "short" and not (tp_price < entry < sl_price):
            logger.warning(f"[DEMO] Bad levels SHORT {symbol}: TP={tp_price} E={entry} SL={sl_price}")
            return False

        sl_dist = abs(entry - sl_price)
        if sl_dist == 0:
            return False

        balance = self._get_balance()
        if balance < 1:
            logger.warning(f"[DEMO] Balance too low: ${balance:.2f}")
            return False

        risk_amount = balance * self.risk_pct
        raw_size    = risk_amount / sl_dist
        min_qty     = _MIN_QTY.get(symbol, _DEFAULT_MIN_QTY)
        size        = max(round(raw_size, 2), min_qty)

        try:
            params = {
                "tdMode": "cross",      # cross margin for swaps
                "posSide": "net",       # one-way mode
                "slOrdPx": str(round(sl_price, 6)),
                "slTriggerPx": str(round(sl_price, 6)),
                "tpOrdPx": str(round(tp_price, 6)),
                "tpTriggerPx": str(round(tp_price, 6)),
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
                f"[DEMO] ✅ {direction.upper()} {symbol} | "
                f"Size={size} | SL={sl_price:.5f} | TP={tp_price:.5f} | "
                f"Balance=${balance:.2f} | ID={order_id}"
            )
            return True

        except ccxt.InvalidOrder as e:
            logger.error(f"[DEMO] InvalidOrder {symbol}: {e}")
        except ccxt.InsufficientFunds as e:
            logger.error(f"[DEMO] InsufficientFunds {symbol}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"[DEMO] ExchangeError {symbol}: {e}")
        except Exception as e:
            logger.error(f"[DEMO] Unexpected error {symbol}: {e}")
        return False

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]
        except Exception as e:
            logger.error(f"[DEMO] fetch_positions error: {e}")
            return []

    def status_summary(self) -> str:
        if not self.enabled:
            return "OKX Demo: disabled"
        balance   = self._get_balance()
        positions = self.get_open_positions()
        if not positions:
            return f"OKX Demo | Balance: ${balance:.2f} | No open positions"
        lines = [f"OKX Demo | Balance: ${balance:.2f}"]
        for p in positions:
            sym  = p.get("symbol", "?")
            side = p.get("side", "?")
            pnl  = float(p.get("unrealizedPnl", 0) or 0)
            lines.append(f"  {sym} {side} | uPnL: {pnl:+.2f}")
        return "\n".join(lines)
