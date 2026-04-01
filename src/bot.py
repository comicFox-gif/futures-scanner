"""
Signal Scanner Bot
-------------------
Scans 15 symbols every 60s.
Sends Telegram alerts at two stages:

  Stage 1 — WARNING    : Trend aligned + RSI + Volume ready, MACD forming
  Stage 2 — CONFIRMED  : All conditions met → enter trade manually

No orders are placed. You trade manually.
"""

from __future__ import annotations
import time
import logging
import traceback
from datetime import datetime, date, timedelta
from typing import Optional

import ccxt
import pandas as pd

from src.strategy import Strategy, Signal
from src.notifier import Notifier

logger = logging.getLogger("futures_bot")

COOLDOWN_KEY = tuple  # (symbol, direction, stage)


def ohlcv_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


class Bot:
    def __init__(self, cfg: dict, env: dict):
        self.cfg = cfg
        self.symbols: list[str] = cfg["symbols"]
        self.tf_trend: str = cfg["timeframe_trend"]
        self.tf_entry: str = cfg["timeframe_entry"]
        self.lookback: int = cfg["strategy"]["lookback_candles"]
        self.poll_interval: int = cfg["bot"]["poll_interval_seconds"]
        self.daily_summary_hour: int = cfg["bot"].get("daily_summary_utc_hour", 0)
        self.cooldown_min: int = cfg["signal"].get("signal_cooldown_minutes", 240)

        self.strategy = Strategy(cfg)
        self.notifier = Notifier()
        self.exchange = self._init_exchange(cfg, env)

        # Cooldown tracking: (symbol, direction, stage) -> datetime of last alert
        self._last_alert: dict[tuple, datetime] = {}

        # Daily stats
        self._daily_alerts: list[dict] = []
        self._last_summary_date: Optional[date] = None
        self._running = False

    # ------------------------------------------------------------------
    # Exchange init
    # ------------------------------------------------------------------

    def _init_exchange(self, cfg: dict, env: dict):
        exchange_id = env.get("EXCHANGE", cfg.get("exchange", "okx"))

        # OKX uses 'swap' for perpetuals; all others use 'future'
        default_type = "swap" if exchange_id == "okx" else "future"

        api_key = env.get("API_KEY", "")
        api_secret = env.get("API_SECRET", "")
        passphrase = env.get("API_PASSPHRASE", "")  # OKX requires passphrase

        options = {
            "defaultType": default_type,
            "adjustForTimeDifference": True,
        }

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "password": passphrase,   # used by OKX, ignored by others
            "enableRateLimit": True,
            "timeout": 15000,
            "options": options,
        })
        logger.info(f"Exchange: {exchange_id} ({default_type}) | Scanning {len(self.symbols)} symbols")
        return exchange

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.lookback)
            if not raw or len(raw) < 50:
                logger.warning(f"Not enough data: {symbol} {timeframe}")
                return None
            return ohlcv_to_df(raw)
        except Exception as e:
            logger.error(f"fetch_ohlcv({symbol}, {timeframe}): {e}")
            return None

    # ------------------------------------------------------------------
    # Cooldown check
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, symbol: str, direction: str, stage: int) -> bool:
        key = (symbol, direction, stage)
        last = self._last_alert.get(key)
        if last is None:
            return False
        return datetime.utcnow() - last < timedelta(minutes=self.cooldown_min)

    def _mark_sent(self, symbol: str, direction: str, stage: int):
        self._last_alert[(symbol, direction, stage)] = datetime.utcnow()

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def _maybe_send_daily_summary(self):
        now = datetime.utcnow()
        today = now.date()
        if now.hour != self.daily_summary_hour:
            return
        if self._last_summary_date == today:
            return
        self._last_summary_date = today

        alerts = self._daily_alerts
        confirmed = [a for a in alerts if a["stage"] == 2]
        warnings  = [a for a in alerts if a["stage"] == 1]

        longs  = [a for a in confirmed if a["direction"] == "long"]
        shorts = [a for a in confirmed if a["direction"] == "short"]

        symbols_hit = list({a["symbol"] for a in confirmed})
        sym_text = ", ".join(symbols_hit) if symbols_hit else "None"

        self.notifier.send(
            f"📊 <b>Daily Scanner Summary — {today}</b>\n"
            f"─────────────────────────\n"
            f"Confirmed signals: <code>{len(confirmed)}</code>  "
            f"(🟢 {len(longs)} Long / 🔴 {len(shorts)} Short)\n"
            f"Warning alerts:    <code>{len(warnings)}</code>\n"
            f"Symbols triggered: {sym_text}\n"
            f"─────────────────────────\n"
            f"<i>Next summary at {self.daily_summary_hour:02d}:00 UTC</i>"
        )
        self._daily_alerts = []

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick(self):
        for symbol in self.symbols:
            try:
                htf_raw   = self._fetch_ohlcv(symbol, self.tf_trend)
                entry_raw = self._fetch_ohlcv(symbol, self.tf_entry)
                if htf_raw is None or entry_raw is None:
                    continue

                htf_df   = self.strategy.enrich(htf_raw.copy())
                entry_df = self.strategy.enrich(entry_raw.copy())

                signal = self.strategy.generate_signal(symbol, htf_df, entry_df)

                if signal.stage == 0:
                    logger.debug(f"{symbol}: no signal")
                    continue

                if self._is_on_cooldown(symbol, signal.direction, signal.stage):
                    logger.debug(f"{symbol}: cooldown active ({signal.direction} stage {signal.stage})")
                    continue

                # Log it
                stage_label = "CONFIRMED" if signal.stage == 2 else "WARNING"
                logger.info(
                    f"[{stage_label}] {signal.direction.upper()} {symbol} "
                    f"@ {signal.entry_price:.4f} | RSI={signal.rsi:.1f} "
                    f"| Vol={signal.volume_ratio:.2f}x | {signal.reason}"
                )

                # Send Telegram alert
                if signal.stage == 2:
                    self.notifier.confirmed_signal(signal)
                else:
                    self.notifier.warning_signal(signal)

                self._mark_sent(symbol, signal.direction, signal.stage)
                self._daily_alerts.append({"stage": signal.stage, "direction": signal.direction, "symbol": symbol})

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}\n{traceback.format_exc()}")
                self.notifier.error_alert(f"Scanning {symbol}", str(e)[:200])

        self._maybe_send_daily_summary()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        logger.info("=" * 60)
        logger.info(f"Signal Scanner started — {len(self.symbols)} symbols")
        logger.info(f"Trend TF: {self.tf_trend} | Entry TF: {self.tf_entry}")
        logger.info(f"Cooldown: {self.cooldown_min} min per signal")
        logger.info(f"Daily summary at: {self.daily_summary_hour:02d}:00 UTC")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)

        self.notifier.scanner_started(self.symbols, self.tf_trend, self.tf_entry, self.cooldown_min)

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Scanner stopped by user.")
                break
            except Exception as e:
                logger.error(f"Tick error: {e}\n{traceback.format_exc()}")
                self.notifier.error_alert("Main loop", str(e)[:200])

            logger.debug(f"Sleeping {self.poll_interval}s...")
            time.sleep(self.poll_interval)

        self.notifier.send("🔴 <b>Signal Scanner stopped.</b>")

    def stop(self):
        self._running = False
