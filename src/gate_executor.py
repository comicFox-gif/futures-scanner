"""
Gate.io Futures Executor — Testnet (official gate-api SDK)
------------------------------------------------------------
Uses Gate.io's own Python SDK. Testnet = just set host= in Configuration.

pip install gate-api

Entry  : market order (FuturesOrder, price="0", tif="ioc")
TP3    : limit order  (FuturesOrder, reduce_only=True, tif="gtc")
SL     : price-triggered stop (FuturesPriceTriggeredOrder)
TP2 BE : cancel old SL → place new FuturesPriceTriggeredOrder at entry
"""

from __future__ import annotations
import logging
import traceback

logger = logging.getLogger("futures_bot.gate")

TESTNET_HOST = "https://api-testnet.gateapi.io/api/v4"
LIVE_HOST    = "https://api.gateio.ws/api/v4"


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
            from gate_api import ApiClient, Configuration, FuturesApi
            host = TESTNET_HOST if testnet else LIVE_HOST
            config = Configuration(key=api_key, secret=api_secret, host=host)
            self._api    = FuturesApi(ApiClient(config))
            self._settle = "usdt"
            # Log masked key so we can verify the right value is loaded from env
            masked_key    = api_key[:6]  + "..." + api_key[-4:]  if len(api_key)    > 10 else "???"
            masked_secret = api_secret[:4] + "..." + api_secret[-4:] if len(api_secret) > 8  else "???"
            logger.info(
                f"[GATE] Executor ready | host={host} | "
                f"key={masked_key} | secret={masked_secret} | "
                f"leverage={leverage}x | risk=${risk_usdt}/trade"
            )
        except ImportError:
            logger.error("[GATE] gate-api not installed — add gate-api to requirements.txt")
            self.enabled = False
        except Exception as e:
            logger.error(f"[GATE] Init failed: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_contract(self, symbol: str) -> str:
        """'BTC/USDT:USDT' → 'BTC_USDT'"""
        return symbol.split(":")[0].replace("/", "_")

    def _get_quanto_multiplier(self, contract: str):
        """Return quanto_multiplier, or None if contract doesn't exist on this exchange."""
        try:
            info = self._api.get_futures_contract(self._settle, contract)
            return float(info.quanto_multiplier)
        except Exception as e:
            logger.info(f"[GATE] {contract} not available on testnet — skipping order")
            return None

    def _set_leverage(self, contract: str):
        try:
            self._api.update_position_leverage(
                self._settle, contract, str(self.leverage),
                cross_leverage_limit=str(self.leverage)
            )
        except Exception as e:
            logger.warning(f"[GATE] set_leverage({contract}): {e}")

    def _n_contracts(self, sl_dist: float, quanto: float) -> int:
        """Dollar risk → number of integer contracts."""
        risk_per_contract = sl_dist * quanto
        if risk_per_contract <= 0:
            return 1
        return max(1, round(self.risk_usdt / risk_per_contract))

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> dict:
        """
        Place entry + TP3 limit + SL stop-market on Gate.io.
        Returns {"sl_order_id": str, "tp_order_id": str} or {} on failure.
        """
        if not self.enabled:
            return {}

        from gate_api import FuturesOrder, FuturesPriceTriggeredOrder, FuturesPriceTrigger, FuturesInitialOrder

        symbol    = signal["symbol"]
        direction = signal["direction"]
        entry     = float(signal["entry"])
        sl        = float(signal["sl"])
        tp3       = float(signal["tp3"])
        contract  = self._to_contract(symbol)

        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            return {}

        quanto = self._get_quanto_multiplier(contract)
        if quanto is None:
            return {}   # contract not listed on this testnet — skip silently

        n          = self._n_contracts(sl_dist, quanto)
        entry_size =  n if direction == "long" else -n   # positive=buy, negative=sell
        close_size = -n if direction == "long" else  n   # opposite direction to close

        result = {}
        try:
            self._set_leverage(contract)

            # ── Entry (market) ──────────────────────────────────────────
            entry_order = self._api.create_futures_order(
                self._settle,
                FuturesOrder(contract=contract, size=entry_size, price="0", tif="ioc")
            )
            logger.info(
                f"[GATE] ENTRY {direction.upper()} {contract} "
                f"size={entry_size} @ market | id={entry_order.id}"
            )

            # ── TP3 limit (reduce-only) ─────────────────────────────────
            tp_order = self._api.create_futures_order(
                self._settle,
                FuturesOrder(
                    contract=contract, size=close_size,
                    price=str(tp3), tif="gtc", reduce_only=True
                )
            )
            result["tp_order_id"] = str(tp_order.id)
            logger.info(f"[GATE] TP3 limit @ {tp3} | id={result['tp_order_id']}")

            # ── SL price-triggered stop-market (reduce-only) ────────────
            # Long SL: trigger when last_price <= sl  → rule=2
            # Short SL: trigger when last_price >= sl → rule=1
            trigger_rule = 2 if direction == "long" else 1
            sl_order = self._api.create_price_triggered_order(
                self._settle,
                FuturesPriceTriggeredOrder(
                    trigger=FuturesPriceTrigger(
                        strategy_type=0,   # 0 = by price
                        price_type=0,      # 0 = last price
                        price=str(sl),
                        rule=trigger_rule,
                        expiration=604800  # 7 days
                    ),
                    initial=FuturesInitialOrder(
                        contract=contract,
                        size=close_size,
                        price="0",
                        tif="ioc",
                        reduce_only=True
                    )
                )
            )
            result["sl_order_id"] = str(sl_order.id)
            logger.info(f"[GATE] SL stop @ {sl} | id={result['sl_order_id']}")

        except Exception as e:
            # Log the full Gate.io error response to diagnose auth/param issues
            detail = getattr(e, "body", "") or getattr(e, "reason", "") or str(e)
            logger.error(f"[GATE] place_order({contract}): {e} | detail={detail}\n{traceback.format_exc()}")

        return result

    # ------------------------------------------------------------------
    # Move SL to break-even (called when TP2 hit)
    # ------------------------------------------------------------------

    def move_sl_to_breakeven(self, symbol: str, direction: str,
                              size: float, entry_price: float,
                              old_sl_order_id: str) -> str:
        """Cancel old SL, place new price-triggered order at entry (break-even)."""
        if not self.enabled:
            return ""

        from gate_api import FuturesPriceTriggeredOrder, FuturesPriceTrigger, FuturesInitialOrder

        contract   = self._to_contract(symbol)
        close_size = -round(size) if direction == "long" else round(size)

        # Cancel old SL
        if old_sl_order_id:
            try:
                self._api.cancel_price_triggered_order(self._settle, old_sl_order_id)
                logger.info(f"[GATE] Cancelled SL order {old_sl_order_id} for {contract}")
            except Exception as e:
                logger.warning(f"[GATE] Cancel SL failed ({old_sl_order_id}): {e}")

        # New SL at entry price (break-even)
        try:
            trigger_rule = 2 if direction == "long" else 1
            new_sl = self._api.create_price_triggered_order(
                self._settle,
                FuturesPriceTriggeredOrder(
                    trigger=FuturesPriceTrigger(
                        strategy_type=0, price_type=0,
                        price=str(entry_price),
                        rule=trigger_rule,
                        expiration=604800
                    ),
                    initial=FuturesInitialOrder(
                        contract=contract,
                        size=close_size,
                        price="0",
                        tif="ioc",
                        reduce_only=True
                    )
                )
            )
            new_id = str(new_sl.id)
            logger.info(f"[GATE] BE SL placed @ {entry_price} for {contract} | id={new_id}")
            return new_id
        except Exception as e:
            logger.error(f"[GATE] move_sl_to_breakeven({contract}): {e}")
            return ""
