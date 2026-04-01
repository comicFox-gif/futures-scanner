"""
Dynamic Pair Selector
----------------------
Fetches top N USDT perpetual futures from the exchange
ranked by 24h trading volume.

Refreshes on a configurable interval so the bot always
tracks the hottest markets, not a stale hardcoded list.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("futures_bot.pairs")

# Pairs to always exclude (stablecoins, low quality derivatives)
BLACKLIST = {
    "USDC/USDT:USDT", "BUSD/USDT:USDT", "TUSD/USDT:USDT",
    "USDP/USDT:USDT", "DAI/USDT:USDT",  "FRAX/USDT:USDT",
}


class PairSelector:
    def __init__(self, exchange, cfg: dict):
        self.exchange   = exchange
        dp              = cfg.get("dynamic_pairs", {})
        self.enabled    = dp.get("enabled", True)
        self.top_n      = dp.get("top_n", 30)
        self.refresh_h  = dp.get("refresh_hours", 4)
        self.min_volume = dp.get("min_volume_usd", 5_000_000)
        self.fallback   = cfg.get("symbols", [])  # use config list if fetch fails

        self._symbols:      list[str]          = []
        self._last_refresh: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_symbols(self) -> list[str]:
        """Return current hot pairs, refreshing if stale."""
        if self.enabled and self._should_refresh():
            self._refresh()
        return self._symbols if self._symbols else self.fallback

    def force_refresh(self):
        """Manually trigger a refresh."""
        self._refresh()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _should_refresh(self) -> bool:
        if self._last_refresh is None:
            return True
        return datetime.utcnow() - self._last_refresh > timedelta(hours=self.refresh_h)

    def _refresh(self):
        logger.info("Fetching top pairs by 24h volume...")
        try:
            # Load markets first to get all active swap symbols
            markets = self.exchange.load_markets()
            swap_symbols = [
                s for s, m in markets.items()
                if m.get("swap") and m.get("quote") == "USDT"
                and m.get("active", True) and s not in BLACKLIST
            ]
            logger.info(f"Found {len(swap_symbols)} active USDT swap markets")

            # Fetch tickers for swap symbols (batch to avoid rate limits)
            tickers = self.exchange.fetch_tickers(swap_symbols)
        except Exception as e:
            logger.error(f"PairSelector: fetch failed: {e}")
            if not self._symbols:
                self._symbols = self.fallback
            return

        # Rank by 24h USD volume
        candidates = []
        for symbol, t in tickers.items():
            if symbol in BLACKLIST:
                continue
            # Try quoteVolume first, fall back to baseVolume * last price
            vol_usd = t.get("quoteVolume") or 0
            if not vol_usd:
                base_vol = t.get("baseVolume") or 0
                last     = t.get("last") or 0
                vol_usd  = base_vol * last
            if vol_usd < self.min_volume:
                continue
            candidates.append((symbol, vol_usd))

        if not candidates:
            logger.warning("PairSelector: no candidates after volume filter — lowering threshold")
            # Retry with no volume filter to at least get something
            candidates = [
                (s, t.get("quoteVolume") or 0)
                for s, t in tickers.items()
                if s not in BLACKLIST and s.endswith(":USDT")
            ]
            if not candidates:
                logger.error("PairSelector: still no candidates, keeping current list")
                if not self._symbols:
                    self._symbols = self.fallback
                return

        # Sort by 24h volume descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        new_symbols = [s for s, _ in candidates[:self.top_n]]

        # Log changes from previous list
        added   = [s for s in new_symbols if s not in self._symbols]
        removed = [s for s in self._symbols if s not in new_symbols]
        if added or removed:
            if added:
                logger.info(f"Pairs added:   {[s.split('/')[0] for s in added]}")
            if removed:
                logger.info(f"Pairs removed: {[s.split('/')[0] for s in removed]}")
        else:
            logger.info("Pairs unchanged")

        self._symbols      = new_symbols
        self._last_refresh = datetime.utcnow()

        logger.info(
            f"Top {len(new_symbols)} pairs: "
            f"{' | '.join(s.split('/')[0] for s in new_symbols)}"
        )
