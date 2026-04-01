"""
Signal Scanner + Paper Trading Bot
------------------------------------
Every 60s:
  1. Scans 15 symbols for Stage 1 (warning) and Stage 2 (confirmed) signals
  2. Sends Telegram alert for every signal
  3. If paper_trading=true: opens a simulated trade on every Stage 2 signal
     and tracks it through TP1 → BE → TP2 → trail → TP3 → exit
"""

from __future__ import annotations
import time
import logging
import traceback
from datetime import datetime, date, timedelta
from typing import Optional

import ccxt
import pandas as pd

from src.strategy import Strategy, Signal, Position
from src.notifier import Notifier

logger = logging.getLogger("futures_bot")


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

        # Paper trading
        paper_cfg = cfg.get("paper_trading", {})
        self.paper_enabled: bool = paper_cfg.get("enabled", False)
        self.paper_balance: float = paper_cfg.get("balance", 1000.0)
        self.paper_risk_pct: float = paper_cfg.get("risk_per_trade_pct", 1.0) / 100.0
        self.paper_start_balance: float = self.paper_balance

        self.strategy = Strategy(cfg)
        self.notifier = Notifier()
        self.exchange = self._init_exchange(cfg, env)

        # Signal cooldown: (symbol, direction, stage) -> last alert time
        self._last_alert: dict[tuple, datetime] = {}

        # Paper positions: symbol -> Position
        self._paper_positions: dict[str, Position] = {}

        # Stats
        self._daily_alerts: list[dict] = []
        self._paper_trades: list[dict] = []   # closed paper trades for daily summary
        self._last_summary_date: Optional[date] = None
        self._running = False

    # ------------------------------------------------------------------
    # Exchange init
    # ------------------------------------------------------------------

    def _init_exchange(self, cfg: dict, env: dict):
        exchange_id = env.get("EXCHANGE", cfg.get("exchange", "okx"))
        default_type = "swap" if exchange_id == "okx" else "future"
        api_key = env.get("API_KEY", "")
        api_secret = env.get("API_SECRET", "")
        passphrase = env.get("API_PASSPHRASE", "")

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "password": passphrase,
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {
                "defaultType": default_type,
                "adjustForTimeDifference": True,
            },
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
    # Cooldown
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
    # Paper trading
    # ------------------------------------------------------------------

    def _paper_open(self, signal: Signal):
        """Open a simulated position from a confirmed signal."""
        if signal.symbol in self._paper_positions:
            logger.debug(f"[PAPER] Already in position for {signal.symbol}, skipping")
            return

        sl_dist = abs(signal.entry_price - signal.stop_loss)
        if sl_dist == 0:
            return

        risk_amount = self.paper_balance * self.paper_risk_pct
        size = round(risk_amount / sl_dist, 6)
        if size <= 0:
            return

        pos = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            size=size,
            size_remaining=size,
        )
        self._paper_positions[signal.symbol] = pos

        sl_pct = sl_dist / signal.entry_price * 100
        logger.info(
            f"[PAPER] OPENED {signal.direction.upper()} {signal.symbol} "
            f"@ {signal.entry_price:.4f} | Size={size:.4f} | "
            f"SL={signal.stop_loss:.4f} (-{sl_pct:.2f}%) | "
            f"TP1={signal.tp1:.4f} | TP2={signal.tp2:.4f} | TP3={signal.tp3:.4f}"
        )
        self.notifier.paper_opened(pos, self.paper_balance)

    def _paper_tick(self, symbol: str, current_price: float):
        """Check open paper position and act on TP/SL hits."""
        pos = self._paper_positions.get(symbol)
        if not pos:
            return

        actions = self.strategy.check_position(pos, current_price)
        if not actions:
            return

        for action in actions:
            act    = action["action"]
            reason = action["reason"]

            if act == "close_all":
                pnl = self._calc_pnl(pos, current_price, pos.size_remaining)
                pos.closed_pnl += pnl
                self.paper_balance += pnl
                tp_level = action.get("tp_level", 0)

                logger.info(
                    f"[PAPER] CLOSED {symbol} | {reason} "
                    f"@ {current_price:.4f} | PnL={pnl:+.2f} | "
                    f"Balance={self.paper_balance:.2f}"
                )
                self.notifier.paper_closed(pos, reason, current_price, pos.closed_pnl, self.paper_balance, tp_level)
                self._paper_trades.append({
                    "symbol": symbol,
                    "direction": pos.direction,
                    "pnl": pos.closed_pnl,
                    "result": "win" if pos.closed_pnl > 0 else "loss",
                    "tp_level": tp_level,
                })
                del self._paper_positions[symbol]
                return

            elif act == "close_partial":
                pct        = action["pct"]
                tp_level   = action.get("tp_level", 0)
                close_size = round(pos.size_remaining * pct, 6)
                pnl        = self._calc_pnl(pos, current_price, close_size)
                pos.size_remaining -= close_size
                pos.closed_pnl += pnl
                self.paper_balance += pnl

                logger.info(
                    f"[PAPER] TP{tp_level} {symbol} | "
                    f"{pct*100:.0f}% closed @ {current_price:.4f} | "
                    f"PnL={pnl:+.2f} | Balance={self.paper_balance:.2f}"
                )
                self.notifier.paper_tp_hit(pos, tp_level, current_price, pnl, self.paper_balance)

            elif act == "move_sl":
                new_sl    = action["new_sl"]
                pos.stop_loss = new_sl
                be_note   = " → Break-Even" if new_sl == pos.entry_price else f" → {new_sl:.4f}"
                logger.info(f"[PAPER] SL moved{be_note} for {symbol}")

    def _calc_pnl(self, pos: Position, exit_price: float, size: float) -> float:
        if pos.direction == "long":
            return (exit_price - pos.entry_price) * size
        return (pos.entry_price - exit_price) * size

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def _maybe_send_daily_summary(self):
        now   = datetime.utcnow()
        today = now.date()
        if now.hour != self.daily_summary_hour:
            return
        if self._last_summary_date == today:
            return
        self._last_summary_date = today

        alerts    = self._daily_alerts
        confirmed = [a for a in alerts if a["stage"] == 2]
        warnings  = [a for a in alerts if a["stage"] == 1]
        longs     = [a for a in confirmed if a["direction"] == "long"]
        shorts    = [a for a in confirmed if a["direction"] == "short"]
        sym_text  = ", ".join({a["symbol"] for a in confirmed}) or "None"

        # Paper stats
        paper_section = ""
        if self.paper_enabled and self._paper_trades:
            wins      = [t for t in self._paper_trades if t["result"] == "win"]
            losses    = [t for t in self._paper_trades if t["result"] == "loss"]
            total_pnl = sum(t["pnl"] for t in self._paper_trades)
            win_rate  = len(wins) / len(self._paper_trades) * 100 if self._paper_trades else 0
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            paper_section = (
                f"\n─────────────────────────\n"
                f"📄 <b>Paper Trading</b>\n"
                f"Trades: <code>{len(self._paper_trades)}</code>  "
                f"(W: {len(wins)} / L: {len(losses)})  Win rate: <code>{win_rate:.0f}%</code>\n"
                f"{pnl_emoji} Day PnL: <code>{total_pnl:+.2f} USDT</code>\n"
                f"Balance: <code>{self.paper_balance:.2f} USDT</code>  "
                f"(started: {self.paper_start_balance:.2f})"
            )

        self.notifier.send(
            f"📊 <b>Daily Summary — {today}</b>\n"
            f"─────────────────────────\n"
            f"Confirmed signals: <code>{len(confirmed)}</code>  "
            f"(🟢 {len(longs)} Long / 🔴 {len(shorts)} Short)\n"
            f"Warnings:          <code>{len(warnings)}</code>\n"
            f"Symbols triggered: {sym_text}"
            f"{paper_section}\n"
            f"─────────────────────────\n"
            f"<i>Next summary {self.daily_summary_hour:02d}:00 UTC</i>"
        )
        self._daily_alerts  = []
        self._paper_trades  = []

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick(self):
        now = datetime.utcnow().strftime("%H:%M:%S")
        open_pos = len(self._paper_positions)
        paper_bal = f"${self.paper_balance:.2f}" if self.paper_enabled else ""
        paper_info = f" | Paper balance: {paper_bal} | Open positions: {open_pos}" if self.paper_enabled else ""
        logger.info(f">>> Scanning {len(self.symbols)} symbols @ {now} UTC{paper_info}")

        signals_found = 0
        for symbol in self.symbols:
            try:
                htf_raw   = self._fetch_ohlcv(symbol, self.tf_trend)
                entry_raw = self._fetch_ohlcv(symbol, self.tf_entry)
                if htf_raw is None or entry_raw is None:
                    continue

                htf_df   = self.strategy.enrich(htf_raw.copy())
                entry_df = self.strategy.enrich(entry_raw.copy())
                current_price = float(entry_df.iloc[-2]["close"])

                # Paper position management (always runs if position is open)
                if self.paper_enabled and symbol in self._paper_positions:
                    self._paper_tick(symbol, current_price)

                # Signal detection (skip if already in paper position for this symbol)
                if self.paper_enabled and symbol in self._paper_positions:
                    continue

                signal = self.strategy.generate_signal(symbol, htf_df, entry_df)

                if signal.stage == 0:
                    logger.debug(f"{symbol}: no signal")
                    continue

                if self._is_on_cooldown(symbol, signal.direction, signal.stage):
                    logger.debug(f"{symbol}: cooldown ({signal.direction} stage {signal.stage})")
                    continue

                stage_label = "CONFIRMED" if signal.stage == 2 else "WARNING"
                logger.info(
                    f"[{stage_label}] {signal.direction.upper()} {symbol} "
                    f"@ {signal.entry_price:.4f} | RSI={signal.rsi:.1f} "
                    f"| Vol={signal.volume_ratio:.2f}x | {signal.reason}"
                )

                # Telegram alert
                if signal.stage == 2:
                    self.notifier.confirmed_signal(signal)
                    # Open paper trade on confirmed signals
                    if self.paper_enabled:
                        self._paper_open(signal)
                else:
                    self.notifier.warning_signal(signal)

                self._mark_sent(symbol, signal.direction, signal.stage)
                self._daily_alerts.append({
                    "stage": signal.stage,
                    "direction": signal.direction,
                    "symbol": symbol,
                })
                signals_found += 1

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}\n{traceback.format_exc()}")
                self.notifier.error_alert(f"Scanning {symbol}", str(e)[:200])

        signal_note = f" | {signals_found} signal(s) fired" if signals_found > 0 else " | No signals"
        logger.info(f"<<< Scan complete{signal_note} | Next scan in {self.poll_interval}s")

        self._maybe_send_daily_summary()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        paper_note = (
            f" | Paper: ON (balance={self.paper_balance:.0f} USDT, risk={self.paper_risk_pct*100:.1f}%)"
            if self.paper_enabled else " | Paper: OFF"
        )
        logger.info("=" * 60)
        logger.info(f"Scanner started — {len(self.symbols)} symbols{paper_note}")
        logger.info(f"Trend TF: {self.tf_trend} | Entry TF: {self.tf_entry}")
        logger.info(f"Cooldown: {self.cooldown_min}min | Summary: {self.daily_summary_hour:02d}:00 UTC")
        logger.info("=" * 60)

        self.notifier.scanner_started(
            self.symbols, self.tf_trend, self.tf_entry,
            self.cooldown_min, self.paper_enabled, self.paper_balance,
        )

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as e:
                logger.error(f"Tick error: {e}\n{traceback.format_exc()}")
                self.notifier.error_alert("Main loop", str(e)[:200])

            logger.debug(f"Sleeping {self.poll_interval}s...")
            time.sleep(self.poll_interval)

        self.notifier.send("🔴 <b>Scanner stopped.</b>")

    def stop(self):
        self._running = False
