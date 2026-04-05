"""
Bybit Futures Executor (pybit SDK)
------------------------------------
Places real orders on Bybit testnet when confirmed signals fire.
Uses official pybit library — entry + SL + TP in a single place_order() call.

pip install pybit

Env vars:
  BYBIT_KEY      — Bybit API key
  BYBIT_SECRET   — Bybit API secret
  BYBIT_TESTNET  — "true" (default) or "false" for live

Risk: 1% of account balance per trade (SL always costs exactly 1%).
"""

from __future__ import annotations
import logging
import traceback

logger = logging.getLogger("futures_bot.bybit")


DEMO_HOST    = "https://api-demo.bybit.com"
TESTNET_HOST = "https://api-testnet.bybit.com"
LIVE_HOST    = "https://api.bybit.com"


class BybitExecutor:
    def __init__(self, api_key: str = "", api_secret: str = "",
                 demo: bool = True, testnet: bool = False,
                 leverage: int = 10, risk_pct: float = 0.01):
        self.leverage = leverage
        self.risk_pct = risk_pct
        self.enabled  = bool(api_key and api_secret)

        if not self.enabled:
            logger.info("[BYBIT] No API keys — executor disabled")
            return

        try:
            from pybit.unified_trading import HTTP
            # Demo trading uses api-demo.bybit.com (NOT testnet)
            if demo:
                base_url  = DEMO_HOST
                env_label = "DEMO"
            elif testnet:
                base_url  = TESTNET_HOST
                env_label = "TESTNET"
            else:
                base_url  = LIVE_HOST
                env_label = "LIVE"

            self.session = HTTP(
                api_key=api_key,
                api_secret=api_secret,
                base_url=base_url,
            )
            masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "???"
            logger.info(
                f"[BYBIT] Executor ready | {env_label} ({base_url}) | key={masked_key} | "
                f"leverage={leverage}x | risk={risk_pct*100:.0f}% per trade"
            )
        except ImportError:
            logger.error("[BYBIT] pybit not installed — run: pip install pybit")
            self.enabled = False
        except Exception as e:
            logger.error(f"[BYBIT] Init failed: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_symbol(self, symbol: str) -> str:
        """'BTC/USDT:USDT' → 'BTCUSDT'"""
        return symbol.split(":")[0].replace("/", "")

    @staticmethod
    def _fmt(price: float, decimals: int = 4) -> str:
        """Format price to avoid floating-point noise."""
        return f"{price:.{decimals}f}".rstrip("0").rstrip(".")

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
            logger.warning(f"[BYBIT] get_balance failed: {e} — using 1000 fallback")
            return 1000.0

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
            # Bybit returns error if leverage is already set to the same value — that's fine
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
        direction = signal["direction"]
        entry     = float(signal["entry"])
        sl        = float(signal["sl"])
        tp3       = float(signal["tp3"])
        side      = "Buy" if direction == "long" else "Sell"

        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            return {}

        # Sanity check direction
        if direction == "long" and not (sl < entry < tp3):
            logger.warning(f"[BYBIT] Bad levels LONG {symbol}: SL={sl} E={entry} TP={tp3}")
            return {}
        if direction == "short" and not (tp3 < entry < sl):
            logger.warning(f"[BYBIT] Bad levels SHORT {symbol}: TP={tp3} E={entry} SL={sl}")
            return {}

        balance   = self._get_balance()
        risk_usdt = round(balance * self.risk_pct, 2)
        # qty in base currency: how many coins to buy/sell so SL = risk_usdt
        qty       = round(risk_usdt / sl_dist, 6)

        logger.info(
            f"[BYBIT] Balance={balance:.2f} | Risk={self.risk_pct*100:.0f}%={risk_usdt:.2f} USDT | "
            f"qty={qty} {symbol}"
        )

        if not self._set_leverage(symbol):
            logger.error(f"[BYBIT] Aborting {symbol} — could not set leverage")
            return {}

        try:
            resp = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                stopLoss=self._fmt(sl),
                takeProfit=self._fmt(tp3),
                tpslMode="Full",        # apply TP/SL to full position
                slOrderType="Market",   # SL triggers as market order
                tpOrderType="Limit",    # TP as limit order
            )
            order_id = resp["result"]["orderId"]
            logger.info(
                f"[BYBIT] ENTRY {side.upper()} {symbol} qty={qty} | "
                f"SL={self._fmt(sl)} | TP3={self._fmt(tp3)} | id={order_id}"
            )
            return {"order_id": order_id}

        except Exception as e:
            detail = getattr(e, "message", str(e))
            logger.error(f"[BYBIT] place_order({symbol}): {detail}\n{traceback.format_exc()}")
            return {}

    # ------------------------------------------------------------------
    # Move SL to break-even (called when TP2 hit)
    # ------------------------------------------------------------------

    def move_sl_to_breakeven(self, symbol: str, direction: str, entry_price: float) -> bool:
        """Move SL to entry price (break-even) using set_trading_stop."""
        if not self.enabled:
            return False

        bybit_symbol = self._to_symbol(symbol)
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=bybit_symbol,
                stopLoss=self._fmt(entry_price),
                positionIdx=0,      # one-way mode
                tpslMode="Full",
            )
            logger.info(f"[BYBIT] BE SL moved to {self._fmt(entry_price)} for {bybit_symbol}")
            return True
        except Exception as e:
            logger.error(f"[BYBIT] move_sl_to_breakeven({bybit_symbol}): {e}")
            return False
