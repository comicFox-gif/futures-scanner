"""
Forex Pair Selector
--------------------
Dynamically selects the top N most active forex pairs from a universe of 40+
major, minor, and cross pairs. Refreshes every `refresh_hours` hours.

Selection metric: momentum score = abs(1D % change) × (ATR / price × 100)
Combines directional movement with volatility — pairs that are both moving
and expanding in range are the best candidates for signals.

Data source: yfinance (free, no API key needed).
"""

from __future__ import annotations
import logging
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger("forex_bot.pair_selector")

# Full universe of tradeable forex pairs
FOREX_UNIVERSE = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    # Euro crosses
    "EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD",
    # GBP crosses
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    # JPY crosses
    "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY", "SGDJPY",
    # AUD crosses
    "AUDCAD", "AUDCHF", "AUDNZD", "AUDSGD",
    # CAD crosses
    "CADCHF",
    # NZD crosses
    "NZDCAD", "NZDCHF", "NZDSGD",
    # Commodity FX
    "USDNOK", "USDSEK", "USDDKK", "USDMXN", "USDZAR", "USDTRY",
    # Asian / EM
    "USDSGD", "USDCNH", "USDTHB", "USDHKD",
    # Scandinavian crosses
    "EURNOK", "EURSEK", "GBPNOK", "GBPSEK",
    # Other EM
    "USDPLN", "USDHUF", "USDCZK",
    # Additional crosses
    "EURCHF", "GBPCHF", "NZDCHF", "CADCHF",
    # Exotic minors
    "EURMXN", "GBPMXN", "EURZAR", "GBPZAR",
]

# Always include majors regardless of score
ALWAYS_INCLUDE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]


class ForexPairSelector:
    def __init__(self, cfg: dict):
        dp = cfg.get("dynamic_pairs", {})
        self.top_n         = dp.get("top_n", 30)
        self.refresh_hours = dp.get("refresh_hours", 4)
        self.universe      = FOREX_UNIVERSE

        self._pairs: list[str] = cfg.get("pairs", ALWAYS_INCLUDE)
        self._last_refresh: float = 0.0

    def get_pairs(self) -> list[str]:
        now = time.time()
        if now - self._last_refresh > self.refresh_hours * 3600:
            self._refresh()
            self._last_refresh = now
        return self._pairs

    def _refresh(self):
        logger.info(f"ForexPairSelector: refreshing top {self.top_n} pairs from {len(self.universe)}-pair universe...")
        scores: list[tuple[str, float]] = []

        for pair in self.universe:
            try:
                yf_sym = pair + "=X"
                ticker = yf.Ticker(yf_sym)
                hist   = ticker.history(period="5d", interval="1d", auto_adjust=True)
                if hist is None or len(hist) < 2:
                    continue

                hist.columns = [c.lower() for c in hist.columns]
                hist = hist[hist["close"] > 0].dropna(subset=["close"])
                if len(hist) < 2:
                    continue

                prev_close  = float(hist["close"].iloc[-2])
                last_close  = float(hist["close"].iloc[-1])
                high        = float(hist["high"].iloc[-1])
                low         = float(hist["low"].iloc[-1])

                if prev_close == 0:
                    continue

                pct_change  = abs((last_close - prev_close) / prev_close * 100)
                day_range   = (high - low) / last_close * 100  # range as % of price
                score       = pct_change * day_range

                scores.append((pair, score))

            except Exception as e:
                logger.debug(f"ForexPairSelector: skipping {pair}: {e}")

        if not scores:
            logger.warning("ForexPairSelector: no scores computed, keeping current list")
            return

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Always include major pairs, fill rest from top scored
        selected = list(ALWAYS_INCLUDE)
        for pair, score in scores:
            if len(selected) >= self.top_n:
                break
            if pair not in selected:
                selected.append(pair)

        self._pairs = selected
        top_display = " | ".join(selected[:10]) + (f" ... +{len(selected)-10} more" if len(selected) > 10 else "")
        logger.info(f"ForexPairSelector: selected {len(selected)} pairs — {top_display}")
