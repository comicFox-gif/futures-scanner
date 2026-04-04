"""
Gate.io Futures Executor — Testnet
------------------------------------
Places real orders on Gate.io futures testnet when confirmed signals fire.

Entry  : market order
TP3    : limit close order (reduce-only)
SL     : stop-market close order (reduce-only)
TP2 BE : cancels old SL, places new SL at entry price
"""

from __future__ import annotations
import logging
import traceback

logger = logging.getLogger("futures_bot.gate")

TESTNET_FUTURES_URL = "https://fx-api-testnet.gateio.ws/api/v4"


class GateExecutor:
    def __init__(self, api_key: str = "", api_secret: str = "",
                 testnet: bool = True, leverage: int = 10, risk_usdt: float = 10.0):
        self.leverage  = leverage
        self.risk_usdt = risk_usdt
        self.enabled   = bool(api_key and api_secret)

        if not self.enabled:
            logger.info("[GATE] No API keys — executor disabled")
            return

        try:
            import ccxt
            self.exchange = ccxt.gate({
                "apiKey":  api_key,
                "secret":  api_secret,
                "options": {"defaultType": "swap"},
                "enableRateLimit": True,
            })
            if testnet:
                # Point all private/public calls to the testnet endpoint
                for key in list(self.exchange.urls.get("api", {})):
                    self.exchange.urls["api"][key] = TESTNET_FUTURES_URL
            logger.info(
                f"[GATE] Executor ready | testnet={testnet} | "
                f"leverage={leverage}x | risk=${risk_usdt}/trade"
            )
        except Exception as e:
            logger.error(f"[GATE] Init failed: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _settle_params(self) -> dict:
        return {"settle": "usdt"}

    def _set_leverage(self, symbol: str):
        try:
            self.exchange.set_leverage(
                self.leverage, symbol,
                params=self._settle_params()
            )
        except Exception as e:
            logger.warning(f"[GATE] set_leverage({symbol}): {e}")

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> dict:
        """
        Place entry + TP3 limit + SL stop-market on Gate.io testnet.
        Returns {"sl_order_id": str, "tp_order_id": str} or empty dict on failure.
        """
        if not self.enabled:
            return {}

        symbol    = signal["symbol"]
        direction = signal["direction"]
        entry     = float(signal["entry"])
        sl        = float(signal["sl"])
        tp3       = float(signal["tp3"])

        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            return {}

        # Size = risk / sl_distance  (same formula as paper trading)
        size       = round(self.risk_usdt / sl_dist, 6)
        side       = "buy"  if direction == "long"  else "sell"
        close_side = "sell" if direction == "long"  else "buy"

        result = {}
        try:
            self._set_leverage(symbol)

            # ── Entry (market) ──────────────────────────────────────────
            entry_order = self.exchange.create_order(
                symbol=symbol, type="market", side=side, amount=size,
                params=self._settle_params()
            )
            logger.info(
                f"[GATE] ENTRY {side.upper()} {symbol} "
                f"size={size} @ market | id={entry_order.get('id', '?')}"
            )

            # ── TP3 limit (reduce-only) ─────────────────────────────────
            tp_order = self.exchange.create_order(
                symbol=symbol, type="limit", side=close_side,
                amount=size, price=tp3,
                params={**self._settle_params(), "reduce_only": True}
            )
            result["tp_order_id"] = str(tp_order.get("id", ""))
            logger.info(
                f"[GATE] TP3 limit @ {tp3} | id={result['tp_order_id']}"
            )

            # ── SL stop-market (reduce-only) ────────────────────────────
            sl_order = self.exchange.create_order(
                symbol=symbol, type="stop_market", side=close_side,
                amount=size, price=None,
                params={**self._settle_params(),
                        "stopPrice": sl, "reduce_only": True}
            )
            result["sl_order_id"] = str(sl_order.get("id", ""))
            logger.info(
                f"[GATE] SL stop @ {sl} | id={result['sl_order_id']}"
            )

        except Exception as e:
            logger.error(f"[GATE] place_order({symbol}): {e}\n{traceback.format_exc()}")

        return result

    # ------------------------------------------------------------------
    # Move SL to break-even (called when TP2 hit internally)
    # ------------------------------------------------------------------

    def move_sl_to_breakeven(self, symbol: str, direction: str,
                              size: float, entry_price: float,
                              old_sl_order_id: str):
        """Cancel old SL order and place new SL at entry price (break-even)."""
        if not self.enabled:
            return

        close_side = "sell" if direction == "long" else "buy"

        # Cancel old SL
        if old_sl_order_id:
            try:
                self.exchange.cancel_order(
                    old_sl_order_id, symbol,
                    params=self._settle_params()
                )
                logger.info(f"[GATE] Cancelled old SL order {old_sl_order_id} for {symbol}")
            except Exception as e:
                logger.warning(f"[GATE] Cancel SL failed ({old_sl_order_id}): {e}")

        # Place new SL at entry (break-even)
        try:
            new_sl = self.exchange.create_order(
                symbol=symbol, type="stop_market", side=close_side,
                amount=size, price=None,
                params={**self._settle_params(),
                        "stopPrice": entry_price, "reduce_only": True}
            )
            new_id = str(new_sl.get("id", ""))
            logger.info(
                f"[GATE] BE SL placed @ {entry_price} for {symbol} | id={new_id}"
            )
            return new_id
        except Exception as e:
            logger.error(f"[GATE] move_sl_to_breakeven({symbol}): {e}")
            return ""
