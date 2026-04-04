"""
Gate.io Futures Executor — Demo Trading (official gate-api SDK)
------------------------------------------------------------
Uses Gate.io's own Python SDK. Demo trading testnet host set via Configuration.

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
            masked_key    = api_key[:6]    + "..." + api_key[-4:]    if len(api_key)    > 10 else "???"
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

    @staticmethod
    def _fmt_price(price: float) -> str:
        """Format price to max 8 decimal places — Gate.io rejects > 12 significant digits."""
        return f"{price:.8f}".rstrip("0").rstrip(".")

    def _get_quanto_multiplier(self, contract: str):
        """Return quanto_multiplier, or None if contract doesn't exist on this exchange."""
        try:
            info = self._api.get_futures_contract(self._settle, contract)
            return float(info.quanto_multiplier)
        except Exception as e:
            logger.info(f"[GATE] {contract} not available on testnet — skipping order")
            return None

    def _set_leverage(self, contract: str) -> bool:
        """Set leverage. Returns True on success, False on failure."""
        try:
            # Try isolated margin leverage first
            self._api.update_position_leverage(
                self._settle, contract, str(self.leverage)
            )
            logger.info(f"[GATE] Leverage set to {self.leverage}x for {contract}")
            return True
        except Exception as e:
            logger.warning(f"[GATE] set_leverage({contract}): {e}")
            return False

    def _n_contracts(self, entry_price: float, quanto: float) -> int:
        """
        Size by margin = risk_usdt at the given leverage.
        margin_per_contract = entry_price × quanto / leverage
        n = risk_usdt / margin_per_contract
        → total margin locked ≤ risk_usdt ($10) regardless of SL distance.
        """
        margin_per_contract = entry_price * quanto / self.leverage
        if margin_per_contract <= 0:
            return 1
        return max(1, int(self.risk_usdt / margin_per_contract))

    def _close_position(self, contract: str, size: int):
        """Emergency close — used if TP/SL placement fails after entry."""
        try:
            from gate_api import FuturesOrder
            self._api.create_futures_order(
                self._settle,
                FuturesOrder(contract=contract, size=size, price="0", tif="ioc", reduce_only=True)
            )
            logger.warning(f"[GATE] Emergency close sent for {contract}")
        except Exception as e:
            logger.error(f"[GATE] Emergency close failed for {contract}: {e}")

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------

    def place_order(self, signal: dict) -> dict:
        """
        Place entry + TP3 limit + SL stop-market on Gate.io.
        Returns {"sl_order_id": str, "tp_order_id": str} or {} on failure.
        If TP/SL placement fails after entry, sends emergency close.
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
            return {}   # contract not listed on testnet — skip silently

        n          = self._n_contracts(entry, quanto)
        entry_size =  n if direction == "long" else -n   # positive=buy, negative=sell
        close_size = -n if direction == "long" else  n   # opposite to close

        # Must set leverage before entry — abort if it fails
        if not self._set_leverage(contract):
            logger.error(f"[GATE] Aborting {contract} — could not set {self.leverage}x leverage")
            return {}

        result     = {}
        entry_done = False
        try:
            # ── Entry (market) ──────────────────────────────────────────
            entry_order = self._api.create_futures_order(
                self._settle,
                FuturesOrder(contract=contract, size=entry_size, price="0", tif="ioc")
            )
            entry_done = True
            logger.info(
                f"[GATE] ENTRY {direction.upper()} {contract} "
                f"size={entry_size} @ market | id={entry_order.id}"
            )

            # ── TP3 limit (reduce-only) ─────────────────────────────────
            tp_order = self._api.create_futures_order(
                self._settle,
                FuturesOrder(
                    contract=contract, size=close_size,
                    price=self._fmt_price(tp3), tif="gtc", reduce_only=True
                )
            )
            result["tp_order_id"] = str(tp_order.id)
            logger.info(f"[GATE] TP3 limit @ {self._fmt_price(tp3)} | id={result['tp_order_id']}")

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
                        price=self._fmt_price(sl),
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
            logger.info(f"[GATE] SL stop @ {self._fmt_price(sl)} | id={result['sl_order_id']}")

        except Exception as e:
            detail = getattr(e, "body", "") or str(e)
            logger.error(f"[GATE] place_order({contract}): {detail}\n{traceback.format_exc()}")
            # If entry went through but TP/SL failed, close immediately
            if entry_done and not result:
                self._close_position(contract, close_size)

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
                        price=self._fmt_price(entry_price),
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
            logger.info(f"[GATE] BE SL placed @ {self._fmt_price(entry_price)} for {contract} | id={new_id}")
            return new_id
        except Exception as e:
            logger.error(f"[GATE] move_sl_to_breakeven({contract}): {e}")
            return ""
