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
import os
import time
import logging
import traceback
import requests
from datetime import datetime, date, timedelta
from typing import Optional

import ccxt
import pandas as pd

from src.strategy import Strategy, Signal, Position
from src.pair_selector import PairSelector
from src.notifier import Notifier
from src.mexc_executor import MexcExecutor
from src.state_manager import save_state, load_state
from src.regime import detect_regime
from src.elite_strategy import EliteStrategy

logger = logging.getLogger("futures_bot")


def ohlcv_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


class Bot:
    def __init__(self, cfg: dict, env: dict):
        self.env  = env   # keep full env dict for later use in run()
        # Mode switch: "scalp" swaps signal + filter params before strategies load
        self.mode = cfg.get("mode", "swing")
        self.send_warnings = False   # confirmed signals only — no setup/warning alerts
        if self.mode == "scalp":
            if "scalp_signal" in cfg:
                cfg = {**cfg, "signal": cfg["scalp_signal"]}
            if "scalp_filters" in cfg:
                cfg = {**cfg, "filters": cfg["scalp_filters"]}
            if "scalp_structure_break" in cfg:
                cfg = {**cfg, "structure_break": cfg["scalp_structure_break"]}
            if "scalp_macd_zero_cross" in cfg:
                cfg = {**cfg, "macd_zero_cross": cfg["scalp_macd_zero_cross"]}
            if "scalp_ema_ribbon_pullback" in cfg:
                cfg = {**cfg, "ema_ribbon_pullback": cfg["scalp_ema_ribbon_pullback"]}
            if "scalp_whale_momentum" in cfg:
                cfg = {**cfg, "whale_momentum": cfg["scalp_whale_momentum"]}
        self.cfg = cfg
        self.tf_trend: str     = cfg["timeframe_trend"]
        self.tf_entry: str     = cfg["timeframe_entry"]
        self.tf_precision: str = "15m"  # 15m for both swing and scalp — 5m is too noisy
        self.lookback: int = cfg["strategy"]["lookback_candles"]
        self.poll_interval: int = cfg["bot"]["poll_interval_seconds"]
        self.daily_summary_hour: int = cfg["bot"].get("daily_summary_utc_hour", 0)
        self.cooldown_min: int = cfg["signal"].get("signal_cooldown_minutes", 240)

        # Paper trading
        paper_cfg = cfg.get("paper_trading", {})
        self.paper_enabled: bool = paper_cfg.get("enabled", False)
        self.paper_balance: float = paper_cfg.get("balance", 1000.0)
        self.risk_pct: float = paper_cfg.get("risk_pct", 0.03)  # 3% of balance per trade
        self.paper_start_balance: float = self.paper_balance

        self.tf_sr: str = cfg.get("timeframe_sr", "4h")

        # Strategy kept for paper-position management (check_position / Position dataclass)
        self.strategy    = Strategy(cfg)
        self.notifier    = Notifier(
            channel_name=cfg.get("channel_name", ""),
            forex_symbols=set(cfg.get("forex_symbols", [])),
        )
        self.exchange    = self._init_exchange(cfg, env)
        self.bybit       = MexcExecutor(
            api_key       = env.get("MEXC_KEY", ""),
            api_secret    = env.get("MEXC_SECRET", ""),
            leverage      = int(env.get("MEXC_LEVERAGE", "10")),
            risk_pct      = self.risk_pct,
            max_positions = 10,
            max_risk_usdt = float(env.get("MEXC_MAX_RISK", "10")),
        )
        self.pair_selector = PairSelector(self.exchange, cfg)

        # Signal cooldown: (symbol, direction, stage) -> last alert time
        self._last_alert: dict[tuple, datetime] = {}

        # Paper positions: symbol -> Position
        self._paper_positions: dict[str, Position] = {}

        # MEXC live position tracking (direction + entry for whale-exit detection)
        self._mexc_positions: dict[str, dict] = {}

        # Forex paper trading — parallel tracker, sends to forex channel
        forex_cfg = cfg.get("forex_paper", {})
        self.forex_paper_balance: float = forex_cfg.get("balance", 1000.0)
        self.forex_paper_start:   float = self.forex_paper_balance
        self._forex_positions:    dict[str, Position] = {}
        self._forex_stats = {"tp2": 0, "whale": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}

        # Session tracking: 10 trades opened → 3h pause → reset
        self._session_count     = 0          # trades opened this session
        self._session_paused    = False
        self._resume_at         = None       # datetime when pause ends
        self._session_start_bal = self.paper_balance
        self._session_trades: list[dict] = []   # all closed trades this session

        # Lifetime trade stats (reset each session)
        self._trade_stats    = {"sl": 0, "tp2": 0, "whale": 0, "be_sl": 0, "total": 0, "wins": 0}
        self._strategy_stats: dict[str, dict] = {}

        # Stats
        self._daily_alerts: list[dict] = []
        self._paper_trades: list[dict] = []
        self._last_summary_date         = None
        self._last_positions_report: datetime = datetime.utcnow()
        self._running = False

        # ── Elite 4H system ────────────────────────────────────────────────
        self.elite_strategy   = EliteStrategy(cfg)
        elite_cfg             = cfg.get("elite", {})
        self._daily_sig_limit  = elite_cfg.get("daily_limit", 5)
        self._max_concurrent   = elite_cfg.get("max_concurrent", 3)
        self._pending_signals: dict[int, dict] = {}   # id → {symbol, signal, live_price, sent_at}
        self._next_pending_id  = 0
        self._daily_elite_count = 0
        self._daily_elite_date  = None                # date when count was last reset
        self._consecutive_sl    = 0                   # consecutive SLs hit (triggers elite pause)
        self._elite_paused      = False
        self._elite_resume_at: datetime | None = None

        # Elite trailing stop state: symbol → {direction, entry, sl_dist,
        #   trail_activate, current_sl, tp, atr, activated}
        self._elite_trail_state: dict[str, dict] = {}

        # Admin command polling — uses dedicated ADMIN_BOT_TOKEN (separate from signal bot)
        self._admin_token  = os.getenv("ADMIN_BOT_TOKEN", "").strip()
        self._admin_id     = os.getenv("TELEGRAM_ADMIN_ID", "").strip()
        self._cmd_offset   = 0
        logger.info(f"[CMD] ADMIN_BOT_TOKEN={'SET' if self._admin_token else 'MISSING'} | TELEGRAM_ADMIN_ID={'SET' if self._admin_id else 'MISSING'}")

        # Mode recommendation tracking — alert once when scalp/swing conditions change
        self._mode_rec: str | None = None           # last recommendation sent
        self._last_mode_alert: datetime | None = None
        self._mode_rec_reason: str = ""             # human-readable reason for last rec

        # Confluence threshold — minimum score (1-5) to fire a signal
        self._confluence_min: int     = int(os.getenv("CONF_MIN", "2"))
        self._confluence_enabled: bool = True   # toggle via Telegram button

        # Restore paper state from previous session (survives redeploy)
        load_state(self)

    # ------------------------------------------------------------------
    # Exchange init
    # ------------------------------------------------------------------

    def _init_exchange(self, cfg: dict, env: dict):
        exchange_id = env.get("EXCHANGE", cfg.get("exchange", "bybit"))
        type_map    = {"bybit": "linear", "okx": "swap", "mexc": "swap"}
        default_type = type_map.get(exchange_id, "linear")
        api_key    = env.get("API_KEY", "")
        api_secret = env.get("API_SECRET", "")
        passphrase = env.get("API_PASSPHRASE", "")

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "apiKey":    api_key,
            "secret":    api_secret,
            "password":  passphrase,
            "enableRateLimit": True,
            "timeout":   15000,
            "options": {
                "defaultType": default_type,
                "adjustForTimeDifference": True,
            },
        })
        logger.info(f"Exchange: {exchange_id} ({default_type})")
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

    def _fetch_live_price(self, symbol: str) -> float | None:
        """Fetch the current live mark/last price from MEXC via ccxt."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price  = ticker.get("last") or ticker.get("close")
            return float(price) if price else None
        except Exception as e:
            logger.warning(f"fetch_live_price({symbol}): {e}")
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

    def _check_session_resume(self):
        """Auto-resume after 5h pause and reset everything."""
        if self._session_paused and self._resume_at and datetime.utcnow() >= self._resume_at:
            self._session_paused    = False
            self._resume_at         = None
            self._session_count     = 0
            self._session_trades    = []
            self._paper_positions   = {}
            self.paper_balance      = self.paper_start_balance
            self._session_start_bal = self.paper_balance
            self._trade_stats       = {"sl": 0, "tp2": 0, "whale": 0, "be_sl": 0, "total": 0, "wins": 0}
            self._strategy_stats    = {}
            self.notifier.send(
                f"🔄 <b>New Session Started</b>\n"
                f"Balance reset to <code>${self.paper_balance:.0f}</code> — scanning for signals..."
            )
            logger.info("[PAPER] Session reset — new 10-trade cycle started")

    def _paper_open(self, signal, strategy_name: str = "", live_price: float = None):
        """
        Open a simulated position from a confirmed signal.
        live_price: live MEXC ticker price — used as the actual entry price.
                    Falls back to signal.entry_price if not provided.
        """
        symbol = signal.symbol if hasattr(signal, "symbol") else signal["symbol"]
        if symbol in self._paper_positions:
            logger.debug(f"[PAPER] Already in position for {symbol}, skipping")
            return

        sig_entry = signal.entry_price if hasattr(signal, "entry_price") else signal["entry"]
        sig_sl    = signal.stop_loss   if hasattr(signal, "stop_loss")   else signal["sl"]
        sig_tp1   = signal.tp1         if hasattr(signal, "tp1")         else signal["tp1"]
        sig_tp2   = signal.tp2         if hasattr(signal, "tp2")         else signal["tp2"]
        sig_tp3   = signal.tp3         if hasattr(signal, "tp3")         else signal["tp3"]
        direction = signal.direction   if hasattr(signal, "direction")   else signal["direction"]

        # Use live MEXC price as entry; recalculate SL/TP offsets from it
        entry   = live_price if live_price else sig_entry
        sl_dist = abs(sig_entry - sig_sl)   # original distance from signal
        if sl_dist == 0:
            return

        # Shift SL and TPs by the difference between live entry and signal entry
        offset = entry - sig_entry
        sl  = sig_sl  + offset
        tp1 = sig_tp1 + offset
        tp2 = sig_tp2 + offset
        tp3 = sig_tp3 + offset

        # Hard cap: max $10 SL exposure per paper trade
        risk_amount = round(min(self.paper_balance * self.risk_pct, 10.0), 2)
        size = round(risk_amount / sl_dist, 6)
        if size <= 0:
            return

        self.paper_balance -= risk_amount

        pos = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            size=size,
            size_remaining=size,
            margin_locked=risk_amount,
            strategy_name=strategy_name,
        )
        self._paper_positions[symbol] = pos
        self._session_count += 1

        sl_pct     = sl_dist / entry * 100
        open_count = len(self._paper_positions)
        logger.info(
            f"[PAPER] OPENED {direction.upper()} {symbol} "
            f"@ {entry:.8g} (live) | Risk=${risk_amount:.2f} | Size={size:.4f} | "
            f"SL={sl:.8g} (-{sl_pct:.2f}%) | Available=${self.paper_balance:.2f} | "
            f"Session: {self._session_count}/10"
        )
        self.notifier.paper_opened(pos, self.paper_balance, open_count, self._session_count)
        save_state(self)


    def _paper_close(self, symbol: str, exit_price: float, reason: str, tp_level: int = 0):
        """Close a paper position and send the closed alert. Used by tick and distribution exit."""
        pos = self._paper_positions.get(symbol)
        if not pos:
            return

        pnl = self._calc_pnl(pos, exit_price, pos.size_remaining)
        pos.closed_pnl += pnl
        self.paper_balance += pos.margin_locked + pnl

        if reason == "SL hit" and pos.be_activated:
            result = "be_sl";  self._trade_stats["be_sl"]  += 1
        elif reason == "SL hit":
            result = "sl";     self._trade_stats["sl"]     += 1
        elif reason == "Whale exit":
            result = "whale";  self._trade_stats["whale"]  += 1
        elif tp_level == 2:
            result = "tp2";    self._trade_stats["tp2"]    += 1
        else:
            result = "other"
        self._trade_stats["total"] += 1
        if pos.closed_pnl > 0:
            self._trade_stats["wins"] += 1

        sn = pos.strategy_name or "Unknown"
        if sn not in self._strategy_stats:
            self._strategy_stats[sn] = {"tp2": 0, "whale": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}
        ss = self._strategy_stats[sn]
        ss["total"] += 1
        if result == "tp2":     ss["tp2"]   += 1
        elif result == "whale": ss["whale"] += 1
        elif result == "be_sl": ss["be_sl"] += 1
        elif result == "sl":    ss["sl"]    += 1
        if pos.closed_pnl > 0: ss["wins"]  += 1

        self._session_trades.append({"pnl": pos.closed_pnl, "result": result, "strategy": sn})
        del self._paper_positions[symbol]

        # Log actual RR achieved and clean up trail state
        trail = self._elite_trail_state.pop(symbol, None)
        if trail:
            entry   = trail["entry"]
            sl_dist = trail["sl_dist"]
            if sl_dist > 0:
                actual_rr = abs(exit_price - entry) / sl_dist
                logger.info(
                    f"[ELITE] {symbol} closed @ {exit_price:.6g} | "
                    f"Actual RR achieved: {actual_rr:.2f}:1 | reason={reason}"
                )

        open_count = len(self._paper_positions)

        # Consecutive SL tracking — 2 in a row triggers 4H elite pause
        if result == "sl":
            self._consecutive_sl += 1
            if self._consecutive_sl >= 2 and not self._elite_paused:
                self._elite_paused    = True
                self._elite_resume_at = datetime.utcnow() + timedelta(hours=4)
                resume_str = self._elite_resume_at.strftime("%H:%M UTC")
                logger.info(f"[ELITE] 2 consecutive SLs — elite scanning paused until {resume_str}")
                self.notifier.send(
                    f"⏸ <b>Elite Scanner Paused</b>\n"
                    f"2 consecutive stop losses hit — new signals paused for 4 hours.\n"
                    f"Resumes at <b>{resume_str}</b>"
                )
        else:
            self._consecutive_sl = 0


        logger.info(
            f"[PAPER] CLOSED {symbol} | {reason} "
            f"@ {exit_price:.4f} | PnL={pos.closed_pnl:+.2f} | "
            f"Balance=${self.paper_balance:.2f} | Open: {open_count}"
        )
        self.notifier.paper_closed(pos, reason, exit_price, pos.closed_pnl,
                                   self.paper_balance, tp_level, self._trade_stats)
        self._paper_trades.append({
            "symbol": symbol, "direction": pos.direction,
            "pnl": pos.closed_pnl, "result": result, "tp_level": tp_level,
        })
        save_state(self)

    def _paper_tick(self, symbol: str, current_price: float):
        """Check open paper position and act on TP/SL hits."""
        pos = self._paper_positions.get(symbol)
        if not pos:
            return

        actions = self.strategy.check_position(pos, current_price)
        if not actions:
            return

        for action in actions:
            act = action["action"]

            if act == "close_all":
                reason     = action.get("reason", "")
                tp_level   = action.get("tp_level", 0)
                exit_price = action.get("exit_price", current_price)
                self._paper_close(symbol, exit_price, reason, tp_level)
                return

            elif act == "notify_tp1":
                self.notifier.paper_tp1_alert(pos, action.get("exit_price", current_price), tp_level=1)

            elif act == "notify_tp2":
                self.notifier.paper_tp1_alert(pos, action.get("exit_price", current_price), tp_level=2)

            elif act == "close_partial":
                close_size    = action["size"]                   # absolute size (1/3 of original)
                tp_level      = action.get("tp_level", 0)
                partial_price = action.get("exit_price", current_price)

                # Guard: don't close more than remaining
                close_size = min(close_size, pos.size_remaining)
                if close_size <= 0:
                    continue

                pnl = self._calc_pnl(pos, partial_price, close_size)
                pos.size_remaining -= close_size
                pos.closed_pnl     += pnl
                self.paper_balance  += pnl

                # Track per-strategy TP2
                if tp_level == 2:
                    sn = pos.strategy_name or "Unknown"
                    if sn not in self._strategy_stats:
                        self._strategy_stats[sn] = {"tp2": 0, "whale": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}
                    self._strategy_stats[sn]["tp2"] += 1

                logger.info(
                    f"[PAPER] TP{tp_level} {symbol} | "
                    f"closed {close_size:.4f} units @ {partial_price:.4f} | "
                    f"PnL={pnl:+.2f} | Balance=${self.paper_balance:.2f}"
                )
                self.notifier.paper_tp_hit(pos, tp_level, partial_price, pnl, self.paper_balance)
                save_state(self)

            elif act == "move_sl":
                new_sl        = action["new_sl"]
                pos.stop_loss = new_sl
                be_note = " → Break-Even" if new_sl == pos.entry_price else f" → {new_sl:.4f}"
                logger.info(f"[PAPER] SL moved{be_note} for {symbol}")
                # Move SL to break-even on MEXC live position
                if self.bybit.enabled:
                    self.bybit.move_sl_to_breakeven(symbol, pos.direction, pos.entry_price)

    # ------------------------------------------------------------------
    # Forex paper trading (parallel tracker → forex channel)
    # ------------------------------------------------------------------

    def _forex_paper_open(self, signal, strategy_name: str = "", live_price: float = None):
        """Open a forex paper position mirroring the confirmed signal using live MEXC price."""
        if not self.notifier.forex_enabled:
            return
        symbol = signal.symbol if hasattr(signal, "symbol") else signal.get("symbol", "")
        if not symbol or symbol in self._forex_positions:
            return

        sig_entry = signal.entry_price if hasattr(signal, "entry_price") else signal["entry"]
        sig_sl    = signal.stop_loss   if hasattr(signal, "stop_loss")   else signal["sl"]
        sig_tp1   = signal.tp1         if hasattr(signal, "tp1")         else signal["tp1"]
        sig_tp2   = signal.tp2         if hasattr(signal, "tp2")         else signal["tp2"]
        sig_tp3   = signal.tp3         if hasattr(signal, "tp3")         else signal["tp3"]
        direction = signal.direction   if hasattr(signal, "direction")   else signal["direction"]

        entry   = live_price if live_price else sig_entry
        sl_dist = abs(sig_entry - sig_sl)
        if sl_dist == 0:
            return

        offset = entry - sig_entry
        sl  = sig_sl  + offset
        tp1 = sig_tp1 + offset
        tp2 = sig_tp2 + offset
        tp3 = sig_tp3 + offset

        risk_amount = round(min(self.forex_paper_balance * self.risk_pct, 10.0), 2)
        size        = round(risk_amount / sl_dist, 6)
        if size <= 0:
            return

        self.forex_paper_balance -= risk_amount

        pos = Position(
            symbol=symbol, direction=direction,
            entry_price=entry, stop_loss=sl,
            tp1=tp1, tp2=tp2, tp3=tp3,
            size=size, size_remaining=size,
            margin_locked=risk_amount,
            strategy_name=strategy_name,
        )
        self._forex_positions[symbol] = pos
        logger.info(f"[FOREX PAPER] OPENED {direction.upper()} {symbol} @ {entry:.5f} (live) | Risk=${risk_amount:.2f}")

    def _forex_paper_tick(self, symbol: str, current_price: float):
        """Check forex paper position and send updates to forex channel."""
        pos = self._forex_positions.get(symbol)
        if not pos:
            return

        actions = self.strategy.check_position(pos, current_price)
        for action in actions:
            act = action["action"]

            if act == "close_all":
                reason   = action.get("reason", "")
                tp_level = action.get("tp_level", 0)
                exit_price = pos.stop_loss if reason == "SL hit" else (pos.tp2 if tp_level == 2 else current_price)
                pnl = self._calc_pnl(pos, exit_price, pos.size_remaining)
                pos.closed_pnl += pnl
                self.forex_paper_balance += pos.margin_locked + pnl

                result = "tp2" if tp_level == 2 else ("sl" if reason == "SL hit" else "other")
                self._forex_stats["total"] += 1
                if tp_level == 2:        self._forex_stats["tp2"] += 1
                elif reason == "SL hit": self._forex_stats["sl"]  += 1
                if pos.closed_pnl > 0:   self._forex_stats["wins"] += 1

                del self._forex_positions[symbol]
                logger.info(
                    f"[FOREX PAPER] CLOSED {symbol} | {reason} @ {exit_price:.5f} "
                    f"| PnL={pnl:+.2f} | Balance={self.forex_paper_balance:.2f} "
                    f"| W:{self._forex_stats['wins']} TP2:{self._forex_stats['tp2']} SL:{self._forex_stats['sl']}"
                )
                return

            # TP hits — tracked silently, no forex channel notification

    def _bybit_order(self, sig, symbol: str = ""):
        """Place order on MEXC. Accepts Signal dataclass or dict."""
        if not self.bybit.enabled:
            return
        if hasattr(sig, "entry_price"):   # Signal dataclass
            d         = {"symbol": sig.symbol, "direction": sig.direction,
                         "entry": sig.entry_price, "sl": sig.stop_loss, "tp3": sig.tp3}
            direction = sig.direction
            sym       = sig.symbol
        else:
            # Dict signals don't include symbol — inject it from caller
            d         = {**sig, "symbol": symbol}
            direction = sig.get("direction", "")
            sym       = symbol
        result = self.bybit.place_order(d)
        if result and result.get("order_id"):
            self._mexc_positions[sym] = {
                "direction":   direction,
                "entry_price": float(d.get("entry", 0)),
            }
            logger.info(f"[MEXC] Tracking live position: {sym} {direction} @ {d.get('entry')}")

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
        self._check_session_resume()
        symbols   = self.pair_selector.get_symbols()
        now       = datetime.utcnow().strftime("%H:%M:%S")
        open_pos  = len(self._paper_positions)
        paper_bal = f"${self.paper_balance:.2f}" if self.paper_enabled else ""
        paper_info = f" | Paper: {paper_bal} | Positions: {open_pos}" if self.paper_enabled else ""

        # ── Elite regime detection (once per tick using BTC data) ────────────
        # Reset daily signal count at UTC midnight
        today = datetime.utcnow().date()
        if self._daily_elite_date != today:
            self._daily_elite_count = 0
            self._daily_elite_date  = today

        # Lift elite pause if cooldown expired
        if self._elite_paused and self._elite_resume_at and datetime.utcnow() >= self._elite_resume_at:
            self._elite_paused    = False
            self._elite_resume_at = None
            logger.info("[ELITE] Pause lifted — resuming signal scanning")

        # BTC data for regime + enrich with elite strategy indicators
        btc_4h_raw    = self._fetch_ohlcv("BTC/USDT:USDT", "4h")
        btc_daily_raw = self._fetch_ohlcv("BTC/USDT:USDT", "1d")
        btc_4h_df   = self.elite_strategy.enrich(btc_4h_raw.copy())    if btc_4h_raw    is not None and len(btc_4h_raw)    >= 10 else None
        btc_daily_df = self.elite_strategy.enrich(btc_daily_raw.copy()) if btc_daily_raw is not None and len(btc_daily_raw) >= 10 else None
        regime_info  = detect_regime(btc_4h_df, btc_daily_df)
        regime_lbl   = {"bull": "🟢 Bull", "bear": "🔴 Bear", "neutral": "⚪ Neutral"}.get(
            regime_info["regime"], "⚪ Neutral"
        )
        logger.info(f">>> Scanning {len(symbols)} pairs @ {now} UTC{paper_info} | Regime: {regime_lbl} | {regime_info['label']}")

        signals_found = 0
        for symbol in symbols:
            try:
                htf_raw = self._fetch_ohlcv(symbol, self.tf_trend)
                if htf_raw is None:
                    continue

                htf_df = self.elite_strategy.enrich(htf_raw.copy())

                if len(htf_df) < 60:
                    logger.debug(f"Skipping {symbol}: insufficient 4H history ({len(htf_df)} candles)")
                    continue

                # Weekly data for structural TP and T1 scoring
                weekly_raw = self._fetch_ohlcv(symbol, "1w")
                weekly_df  = (
                    self.elite_strategy.enrich(weekly_raw.copy())
                    if weekly_raw is not None and len(weekly_raw) >= 5
                    else None
                )

                # Live price for paper trade monitoring and entry
                live_price    = self._fetch_live_price(symbol)
                current_price = live_price if live_price else float(htf_df.iloc[-2]["close"])

                # Elite trailing stop — runs before _paper_tick so SL is updated first
                if symbol in self._elite_trail_state:
                    self._elite_manage_trail(symbol, current_price, htf_df)

                # Paper position management
                if self.paper_enabled and symbol in self._paper_positions:
                    self._paper_tick(symbol, current_price)
                if symbol in self._forex_positions:
                    self._forex_paper_tick(symbol, current_price)

                in_paper = self.paper_enabled and symbol in self._paper_positions

                # ── Elite 4H institutional strategy ──────────────────────────
                if not in_paper:
                    if self._elite_tick(
                        symbol=symbol,
                        weekly_df=weekly_df,
                        h4_df=htf_df,
                        regime=regime_info,
                        current_price=current_price,
                        live_price=live_price,
                    ):
                        signals_found += 1

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}\n{traceback.format_exc()}")
                self.notifier.error_alert(f"Scanning {symbol}", str(e)[:200])

        signal_note = f" | {signals_found} signal(s) fired" if signals_found > 0 else " | No signals"
        logger.info(f"<<< Scan complete{signal_note} | Next scan in {self.poll_interval}s")
        logger.info(f"    Pairs: {' | '.join(s.split('/')[0] for s in symbols)}")

        self._maybe_send_daily_summary()
        self._maybe_send_positions_report()

    # ------------------------------------------------------------------
    # Elite 4H system methods
    # ------------------------------------------------------------------

    def _elite_tick(
        self,
        symbol: str,
        weekly_df,
        h4_df,
        regime: dict,
        current_price: float,
        live_price: float = None,
    ) -> bool:
        """
        Run the Elite 4H strategy for one symbol.
        Returns True if a signal was queued for admin approval.
        """
        # Daily signal cap
        if self._daily_elite_count >= self._daily_sig_limit:
            return False

        # Consecutive-SL cooldown
        if self._elite_paused:
            return False

        # Max concurrent positions
        concurrent = len(self._paper_positions) + len(self._mexc_positions)
        if concurrent >= self._max_concurrent:
            return False

        # Already pending approval for this symbol — don't spam admin
        if any(v["symbol"] == symbol for v in self._pending_signals.values()):
            return False

        # Standard signal cooldown
        if self._is_on_cooldown(symbol, "elite", 2):
            return False

        try:
            sig = self.elite_strategy.generate_signal(
                symbol=symbol,
                weekly_df=weekly_df,
                h4_df=h4_df,
                regime=regime,
                exchange=self.exchange if self.bybit.enabled else None,
            )
        except Exception as e:
            logger.warning(f"[ELITE] {symbol} generate_signal error: {e}")
            return False

        if not sig:
            return False

        # Queue for admin approval
        pid = self._next_pending_id
        self._next_pending_id += 1
        self._pending_signals[pid] = {
            "id":         pid,
            "symbol":     symbol,
            "signal":     sig,
            "live_price": live_price or current_price,
            "sent_at":    datetime.utcnow(),
        }
        self._send_signal_approval(pid, sig, regime)
        self._mark_sent(symbol, "elite", 2)
        self._daily_alerts.append({"stage": 2, "direction": sig["direction"], "symbol": symbol})

        logger.info(
            f"[ELITE] {sig['direction'].upper()} {symbol} "
            f"@ {sig['entry']:.6g} | Score={sig['score']}/20 | Pending #{pid}"
        )
        return True

    def _send_signal_approval(self, pid: int, sig: dict, regime: dict):
        """Send signal to admin bot with Approve / Skip / Wait buttons (20-point format)."""
        def f(v):
            if v is None:      return "N/A"
            if abs(v) >= 1000: return f"{v:,.2f}"
            if abs(v) >= 10:   return f"{v:.4f}"
            return f"{v:.6f}"

        direction  = sig["direction"]
        symbol     = sig["symbol"]
        score      = sig.get("score", 0)
        t_score    = sig.get("tech_score", 0)
        s_score    = sig.get("sent_score", 0)
        adv_score  = sig.get("adv_score",  0)
        tp_rr      = sig.get("tp_rr", 0)
        risk_usdt  = sig.get("risk_usdt", 5.0)
        reward     = risk_usdt * tp_rr

        dir_icon   = "🟢 LONG" if direction == "long" else "🔴 SHORT"
        regime_lbl = {"bull": "🟢 Bull", "bear": "🔴 Bear", "neutral": "⚪ Neutral"}.get(
            regime.get("regime", "neutral"), "⚪ Neutral"
        )
        conf_label = (
            "ELITE 🏆" if score >= 15 else
            "Strong ⚡" if score >= 12 else
            "Medium"   if score >= 8  else "Base"
        )

        # Score bars (out of 5 each for tech/sent, out of 10 for adv)
        bar_t   = "█" * t_score  + "░" * (5  - t_score)
        bar_s   = "█" * s_score  + "░" * (5  - s_score)
        bar_adv = "█" * min(adv_score, 10) + "░" * (10 - min(adv_score, 10))

        # Advanced factor lines (pull from adv_detail if present)
        adv   = sig.get("adv_detail", {})
        lines = []
        kz_lbl   = sig.get("kz_label",   "")
        wyck_lbl = sig.get("wyck_label",  "")
        mmm_lbl  = sig.get("mmm_label",   "")
        vsa_lbl  = sig.get("vsa_label",   "")
        im_lbl   = (sig.get("im_label",   "") or "").split("\n")[0]  # first line only

        if kz_lbl:   lines.append(kz_lbl)
        if wyck_lbl: lines.append(wyck_lbl)
        if mmm_lbl:  lines.append(mmm_lbl)
        if vsa_lbl:  lines.append(vsa_lbl)
        if im_lbl:   lines.append(im_lbl)

        adv_block = "\n".join(f"  {l}" for l in lines) if lines else "  N/A"

        text = (
            f"🚨 <b>Signal #{pid:03d} — Pending Approval</b>\n\n"
            f"{dir_icon}  •  <b>{symbol}</b>\n"
            f"Timeframe: 4H  |  {regime_lbl}\n\n"
            f"📌 Entry   <code>{f(sig['entry'])}</code>\n"
            f"🛑 SL      <code>{f(sig['sl'])}</code>\n"
            f"🎯 TP      <code>{f(sig['tp1'])}</code>  ({tp_rr:.1f}:1 RR)\n"
            f"💵 Risk    <b>${risk_usdt:.0f}</b>  →  Reward <b>${reward:.0f}</b>\n"
            f"🔁 Trail   activates at <code>{f(sig.get('trail_activate', 0))}</code>\n\n"
            f"📊 Score  <b>{score}/20</b>  [{conf_label}]\n"
            f"  Tech [{bar_t}] {t_score}/5\n"
            f"  Sent [{bar_s}] {s_score}/5\n"
            f"  Adv  [{bar_adv}] {adv_score}/10\n\n"
            f"<b>Advanced Confluence:</b>\n{adv_block}\n\n"
            f"<i>{sig.get('reason', '')[:180]}</i>"
        )
        markup = {
            "inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": f"elite_approve_{pid}"},
                {"text": "⏭ Skip",    "callback_data": f"elite_skip_{pid}"},
                {"text": "⏳ Wait",   "callback_data": f"elite_wait_{pid}"},
            ]]
        }
        self._admin_send(text, markup=markup)

    def _handle_signal_approve(self, pid: int):
        """
        Execute an approved elite signal:
          1. Post to public Telegram channel
          2. Place MEXC order
          3. Open paper position
        """
        entry_data = self._pending_signals.pop(pid, None)
        if not entry_data:
            logger.warning(f"[ELITE] Approved #{pid} not found in pending dict")
            return

        sig    = entry_data["signal"]
        symbol = entry_data["symbol"]
        live_p = entry_data.get("live_price", sig.get("entry", 0))
        self._daily_elite_count += 1

        direction  = sig["direction"]
        score      = sig.get("score", 0)
        dir_tag    = "🟢 LONG" if direction == "long" else "🔴 SHORT"
        base       = symbol.split("/")[0]
        tp_rr      = sig.get("tp_rr", 0)
        risk_usdt  = sig.get("risk_usdt", 5.0)
        reward     = risk_usdt * tp_rr
        conf_label = (
            "ELITE 🏆" if score >= 15 else
            "Strong ⚡" if score >= 12 else
            "Medium"   if score >= 8  else "Base"
        )
        from datetime import datetime as _dt
        now_utc = _dt.utcnow()
        kz_line = sig.get("kz_label", "")

        def f(v):
            if v is None:      return "N/A"
            if abs(v) >= 1000: return f"${v:,.2f}"
            if abs(v) >= 10:   return f"${v:.4f}"
            return f"${v:.6f}"

        # Advanced factor summary for channel
        adv_lines = []
        for lbl_key in ("kz_label", "wyck_label", "mmm_label", "vsa_label"):
            lbl = sig.get(lbl_key, "")
            if lbl:
                adv_lines.append(lbl)
        im_line = (sig.get("im_label", "") or "").split("\n")[0]
        if im_line:
            adv_lines.append(im_line)
        adv_block = "\n".join(adv_lines) if adv_lines else ""

        pub_text = (
            f"🚨 <b>ELITE SIGNAL</b>\n\n"
            f"{dir_tag}  •  <b>{base}</b>  |  4H\n"
            f"{kz_line}\n\n"
            f"📌 Entry   <code>{f(sig['entry'])}</code>\n"
            f"🛑 SL      <code>{f(sig['sl'])}</code>\n"
            f"🎯 TP      <code>{f(sig['tp1'])}</code>  ({tp_rr:.1f}:1)\n"
            f"💵 Risk ${risk_usdt:.0f}  →  Reward ${reward:.0f}\n"
            f"🔁 Trail   <code>{f(sig.get('trail_activate', 0))}</code>\n\n"
            f"Score: <b>{score}/20</b>  [{conf_label}]\n"
            + (f"\n{adv_block}\n" if adv_block else "")
        )
        self.notifier.send(pub_text)

        # MEXC live order
        self._bybit_order(sig, symbol)

        # Paper position
        if self.paper_enabled and symbol not in self._paper_positions:
            from src.strategy import Signal as Sig
            dummy = Sig(
                stage=2, direction=direction, symbol=symbol,
                entry_price=sig["entry"], stop_loss=sig["sl"],
                tp1=sig["tp1"], tp2=sig["tp2"], tp3=sig["tp3"],
                atr=sig.get("atr", 0), rsi=sig.get("rsi", 50),
                volume_ratio=sig.get("vol_ratio", 0),
                reason=sig.get("reason", ""),
            )
            self._paper_open(dummy, "Elite 4H BOS", live_price=live_p)

        # Register trailing stop state — managed each tick by _elite_manage_trail
        sl_dist = sig.get("sl_dist", abs(sig["entry"] - sig["sl"]))
        if sl_dist > 0:
            self._elite_trail_state[symbol] = {
                "direction":      direction,
                "entry":          sig["entry"],
                "sl_dist":        sl_dist,
                "trail_activate": sig.get(
                    "trail_activate",
                    sig["entry"] + (self.elite_strategy.trail_rr * sl_dist
                                    if direction == "long"
                                    else -self.elite_strategy.trail_rr * sl_dist),
                ),
                "current_sl":     sig["sl"],
                "tp":             sig["tp1"],
                "atr":            sig.get("atr", 0),
                "activated":      False,
            }
            logger.info(
                f"[ELITE] Trail state registered for {symbol} | "
                f"activates @ {self._elite_trail_state[symbol]['trail_activate']:.6g}"
            )

        logger.info(
            f"[ELITE] #{pid} approved — {direction.upper()} {symbol} "
            f"posted to channel + MEXC order placed"
        )

    # ------------------------------------------------------------------
    # Elite trailing stop management
    # ------------------------------------------------------------------

    def _send_trail_notification(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl_dist: float,
        new_sl: float,
        current_price: float,
        tp: float,
    ):
        """Send Telegram notification when the trailing stop moves."""
        base = symbol.split("/")[0]

        if direction == "long":
            running_rr = (current_price - entry) / sl_dist
            locked_rr  = (new_sl - entry) / sl_dist
        else:
            running_rr = (entry - current_price) / sl_dist
            locked_rr  = (entry - new_sl) / sl_dist

        # Try to get locked $ pnl from the paper position if one is open
        locked_str = ""
        pos = self._paper_positions.get(symbol)
        if pos:
            locked_pnl = self._calc_pnl(pos, new_sl, pos.size_remaining)
            locked_str = f"\nLocked profit: <b>${locked_pnl:+.2f}</b>"
        elif locked_rr != 0:
            locked_str = f"\nLocked profit: <b>{locked_rr:+.2f}R</b>"

        def fmtp(v: float) -> str:
            if abs(v) >= 1_000:  return f"${v:,.2f}"
            if abs(v) >= 10:     return f"${v:.4f}"
            return f"${v:.6f}"

        text = (
            f"🔒 <b>TRAIL STOP MOVED</b>\n"
            f"Pair: <b>{base}</b>\n"
            f"New SL: <code>{fmtp(new_sl)}</code>{locked_str}\n"
            f"Running RR: <b>{running_rr:.1f}:1</b> and climbing 📈"
        )
        self.notifier.send(text)

    def _elite_manage_trail(
        self,
        symbol: str,
        current_price: float,
        h4_df=None,
    ):
        """
        Manage trailing stop for an active elite position.

        Logic:
          - Before activation: watches for price to reach trail_activate (default 2:1).
          - After activation: trails price by 1 ATR; only moves in profit direction;
            never moves backwards.
          - Updates paper position's stop_loss so _paper_tick() handles the exit.
          - Calls update_trail_sl() on MEXC to update the live exchange stop.
        """
        state = self._elite_trail_state.get(symbol)
        if not state:
            return

        direction      = state["direction"]
        entry          = state["entry"]
        sl_dist        = state["sl_dist"]
        trail_activate = state["trail_activate"]
        current_sl     = state["current_sl"]
        tp             = state["tp"]
        atr            = state["atr"]

        # Refresh ATR from fresh 4H data when available
        if h4_df is not None and len(h4_df) >= 2:
            try:
                fresh_atr = float(h4_df.iloc[-2].get("atr", atr))
                if not pd.isna(fresh_atr) and fresh_atr > 0:
                    atr = fresh_atr
                    state["atr"] = atr
            except Exception:
                pass

        if atr <= 0:
            return

        def _apply_new_sl(new_sl: float):
            """Persist new SL to trail state, paper position, MEXC stop."""
            state["current_sl"] = new_sl
            # Update paper position SL so _paper_tick() triggers exit correctly
            pos = self._paper_positions.get(symbol)
            if pos:
                pos.stop_loss = new_sl
            # Update MEXC live stop order
            if self.bybit.enabled and symbol in self._mexc_positions:
                self.bybit.update_trail_sl(symbol, direction, new_sl, tp)
            self._send_trail_notification(
                symbol, direction, entry, sl_dist, new_sl, current_price, tp
            )

        if direction == "long":
            candidate_sl = current_price - atr

            if not state["activated"]:
                if current_price >= trail_activate:
                    state["activated"] = True
                    logger.info(
                        f"[TRAIL] {symbol} LONG trail ACTIVATED @ {current_price:.6g} "
                        f"(trail_activate={trail_activate:.6g})"
                    )
                    if candidate_sl > current_sl:
                        _apply_new_sl(candidate_sl)
                    else:
                        state["current_sl"] = candidate_sl
            else:
                # Already activated — only move up
                if candidate_sl > state["current_sl"]:
                    _apply_new_sl(candidate_sl)

        else:  # short
            candidate_sl = current_price + atr

            if not state["activated"]:
                if current_price <= trail_activate:
                    state["activated"] = True
                    logger.info(
                        f"[TRAIL] {symbol} SHORT trail ACTIVATED @ {current_price:.6g} "
                        f"(trail_activate={trail_activate:.6g})"
                    )
                    if candidate_sl < current_sl:
                        _apply_new_sl(candidate_sl)
                    else:
                        state["current_sl"] = candidate_sl
            else:
                if candidate_sl < state["current_sl"]:
                    _apply_new_sl(candidate_sl)

    def _send_session_summary(self):
        trades = self._session_trades
        if not trades:
            return
        wins      = [t for t in trades if t["pnl"] > 0]
        losses    = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        win_pct   = len(wins) / len(trades) * 100 if trades else 0
        self.notifier.paper_session_summary(
            total=len(trades), wins=len(wins), losses=len(losses),
            total_pnl=total_pnl, win_pct=win_pct,
            start_balance=self._session_start_bal,
            current_balance=self.paper_balance,
            stats=self._trade_stats,
            strategy_stats=self._strategy_stats,
        )

    def _maybe_send_positions_report(self):
        """Send open positions summary to Telegram every 60 minutes."""
        if not self.paper_enabled:
            return
        if (datetime.utcnow() - self._last_positions_report).total_seconds() < 3600:
            return
        self._last_positions_report = datetime.utcnow()
        self.notifier.paper_positions_update(self._paper_positions, self.paper_balance, self.paper_start_balance)

    # ------------------------------------------------------------------
    # Telegram command listener (admin only)
    # ------------------------------------------------------------------

    def _check_mode_recommendation(self):
        """
        Each tick: check BTC 1h ADX + active session window.
        Alert admin when conditions flip between scalp-friendly and swing-friendly.
        London open: 07-11 UTC | NY open: 13-17 UTC = scalp windows (high liquidity).
        Outside those windows / ADX < 20 = swing conditions.
        Sends at most once per 2 hours to avoid spam.
        """
        if not self._admin_token or not self._admin_id:
            return
        try:
            import pandas as pd
            now_utc = datetime.utcnow()
            hour    = now_utc.hour

            # Session windows (UTC)
            in_london  = 7 <= hour < 11
            in_ny      = 13 <= hour < 17
            in_active  = in_london or in_ny

            # BTC 1h conditions
            btc_raw = self._fetch_ohlcv("BTC/USDT:USDT", "1h")
            if btc_raw is None or len(btc_raw) < 40:
                return
            btc_df       = self.elite_strategy.enrich(btc_raw.copy())
            row          = btc_df.iloc[-2]
            adx          = float(row.get("adx", 20))
            atr          = float(row.get("atr", 0))
            atr_sma      = btc_df["atr"].rolling(20).mean().iloc[-2]
            atr_expanding = (not pd.isna(atr_sma)) and atr > float(atr_sma)
            trending      = (not pd.isna(adx)) and adx >= 25

            # Determine recommendation
            if in_active and (trending or atr_expanding):
                rec = "scalp"
            elif not in_active or (not trending and not atr_expanding):
                rec = "swing"
            else:
                return  # ambiguous conditions — stay silent

            # Only alert on change, and at most once per 2 hours
            if rec == self._mode_rec:
                return
            if self._last_mode_alert and (now_utc - self._last_mode_alert).total_seconds() < 7200:
                return

            self._mode_rec        = rec
            self._last_mode_alert = now_utc

            mode_buttons = {
                "inline_keyboard": [[
                    {"text": "⚡ Switch to Scalp", "callback_data": "cmd_scalp"},
                    {"text": "📈 Switch to Swing", "callback_data": "cmd_swing"},
                ]]
            }

            if rec == "scalp":
                session_label = "London" if in_london else "NY"
                atr_label     = "↑ expanding" if atr_expanding else "normal"
                self._mode_rec_reason = f"{session_label} session open | ADX 1h: {adx:.0f} | ATR: {atr_label}"
                msg = (
                    f"⚡ <b>Scalp Conditions Active</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Session: <b>{session_label} open</b> | ADX 1h: <b>{adx:.0f}</b> | ATR: {atr_label}\n"
                    f"Current mode: <b>{self.mode.upper()}</b>\n\n"
                    f"Liquidity is high — faster setups on <b>15m/30m</b> work well now."
                )
            else:
                session_label = "Asian session" if not in_active else "Low volatility"
                self._mode_rec_reason = f"{session_label} | ADX 1h: {adx:.0f}"
                msg = (
                    f"📈 <b>Swing Conditions Active</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Conditions: <b>{session_label}</b> | ADX 1h: <b>{adx:.0f}</b>\n"
                    f"Current mode: <b>{self.mode.upper()}</b>\n\n"
                    f"Low volatility / ranging — <b>4h/1h</b> swing setups preferred."
                )
            self._admin_send(msg, markup=mode_buttons)
            logger.info(f"[MODE REC] {rec.upper()} recommended (ADX={adx:.0f}, session={in_active}, atr_exp={atr_expanding})")
        except Exception as e:
            logger.warning(f"[MODE REC] check error: {e}")

    def _admin_send(self, text: str, markup: dict = None):
        """Send a message via the dedicated admin bot."""
        if not self._admin_token or not self._admin_id:
            return
        try:
            payload = {"chat_id": self._admin_id, "text": text, "parse_mode": "HTML"}
            if markup:
                payload["reply_markup"] = markup
            requests.post(
                f"https://api.telegram.org/bot{self._admin_token}/sendMessage",
                json=payload, timeout=10,
            )
        except Exception as e:
            logger.warning(f"[CMD] Admin send failed: {e}")

    def _control_panel_markup(self) -> dict:
        """Inline keyboard for the admin control panel."""
        bybit_btn    = "⏹ Stop Trading" if self.bybit.enabled else "▶️ Start Trading"
        bybit_cb     = "cmd_stop"        if self.bybit.enabled else "cmd_start"
        scalp_btn    = "⚡ Scalp ✓"     if self.mode == "scalp" else "⚡ Scalp"
        swing_btn    = "📈 Swing ✓"     if self.mode == "swing" else "📈 Swing"
        elite_pause_btn = "▶️ Resume Elite" if self._elite_paused else "⏸ Pause Elite"
        elite_pause_cb  = "cmd_resume_elite" if self._elite_paused else "cmd_pause_elite"
        pending_n    = len(self._pending_signals)
        return {
            "inline_keyboard": [
                # Row 1: MEXC toggle
                [{"text": bybit_btn, "callback_data": bybit_cb}],
                # Row 2: Mode switch
                [
                    {"text": scalp_btn, "callback_data": "cmd_scalp"},
                    {"text": swing_btn, "callback_data": "cmd_swing"},
                ],
                # Row 3: Elite controls
                [
                    {"text": elite_pause_btn, "callback_data": elite_pause_cb},
                    {"text": "🔄 Reset Daily", "callback_data": "cmd_reset_daily"},
                ],
                # Row 4: Pending signals
                [
                    {"text": f"⏳ Pending: {pending_n}", "callback_data": "cmd_status"},
                    {"text": f"📊 Today: {self._daily_elite_count}/{self._daily_sig_limit}", "callback_data": "cmd_status"},
                ],
                # Row 5: Status
                [{"text": "📊 Status", "callback_data": "cmd_status"}],
            ]
        }

    def _send_admin(self, text: str, markup: dict = None):
        """Send via admin bot (kept for compatibility)."""
        self._admin_send(text, markup)

    def _answer_callback(self, callback_id: str, text: str = ""):
        """Acknowledge a button press so Telegram removes the loading spinner."""
        if not self._admin_token:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._admin_token}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": text},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"[CMD] answerCallbackQuery failed: {e}")

    def _send_control_panel(self, *_):
        """Send the control panel with current state and action buttons."""
        bybit_state  = "🟢 ON"   if self.bybit.enabled  else "🔴 OFF"
        paper_state  = "🟢 ON"   if self.paper_enabled  else "🔴 OFF"
        elite_state  = "🔴 PAUSED" if self._elite_paused else "🟢 ACTIVE"
        resume_note  = ""
        if self._elite_paused and self._elite_resume_at:
            resume_note = f" (resumes {self._elite_resume_at.strftime('%H:%M UTC')})"
        open_pos     = len(self._paper_positions)
        balance      = f"${self.paper_balance:.2f}" if self.paper_enabled else "N/A"
        pending_n    = len(self._pending_signals)
        text = (
            f"🤖 <b>Bot Control Panel</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"MEXC execution:   {bybit_state}\n"
            f"Paper trading:    {paper_state} ({balance})\n"
            f"Open positions:   {open_pos}\n"
            f"Mode:             {self.mode.upper()}\n"
            f"Elite scanning:   {elite_state}{resume_note}\n"
            f"Signals today:    {self._daily_elite_count}/{self._daily_sig_limit}\n"
            f"Pending approval: {pending_n}"
        )
        self._admin_send(text, markup=self._control_panel_markup())

    def _poll_commands(self):
        """Check Telegram for admin commands once per tick. No threads — runs in main loop."""
        if not self._admin_token or not self._admin_id:
            return
        token = self._admin_token
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": self._cmd_offset, "timeout": 0, "allowed_updates": ["message", "callback_query"]},
                timeout=10,
            )
            data = resp.json()
            for update in data.get("result", []):
                self._cmd_offset = update["update_id"] + 1

                # ── Inline button press ──────────────────────────────────
                cb = update.get("callback_query")
                if cb:
                    cb_chat = str(cb.get("from", {}).get("id", ""))
                    if cb_chat != self._admin_id:
                        continue
                    cb_data = cb.get("data", "")
                    cb_id   = cb["id"]
                    if cb_data == "cmd_stop":
                        self.bybit.enabled = False
                        logger.info("[CMD] MEXC execution DISABLED via button")
                        self._answer_callback(cb_id, "⏹ Trading stopped")
                        self._send_control_panel()
                    elif cb_data == "cmd_start":
                        self.bybit.enabled = True
                        logger.info("[CMD] MEXC execution ENABLED via button")
                        self._answer_callback(cb_id, "▶️ Trading started")
                        self._send_control_panel()
                    elif cb_data == "cmd_status":
                        self._answer_callback(cb_id)
                        self._send_control_panel()
                    elif cb_data == "cmd_scalp":
                        self.mode         = "scalp"
                        self.tf_trend     = "30m"
                        self.tf_entry     = "15m"
                        self.tf_precision = "15m"
                        logger.info("[CMD] Switched to SCALP mode (30m trend / 15m structure / 15m precision)")
                        self._answer_callback(cb_id, "⚡ Scalp mode active")
                        self._send_control_panel()
                        reason = self._mode_rec_reason or "London/NY session open — high liquidity"
                        self.notifier.send(
                            f"⚡ <b>Mode switched to SCALP</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Reason: {reason}\n"
                            f"Trend: <b>30m</b>  →  Structure: <b>15m</b>  →  Precision: <b>5m</b>"
                        )
                    elif cb_data == "cmd_swing":
                        self.mode         = "swing"
                        self.tf_trend     = "4h"
                        self.tf_entry     = "1h"
                        self.tf_precision = "15m"
                        logger.info("[CMD] Switched to SWING mode (4h trend / 1h structure / 15m precision)")
                        self._answer_callback(cb_id, "📈 Swing mode active")
                        self._send_control_panel()
                        reason = self._mode_rec_reason or "Asian session / low volatility — ranging market"
                        self.notifier.send(
                            f"📈 <b>Mode switched to SWING</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Reason: {reason}\n"
                            f"Trend: <b>4h</b>  →  Structure: <b>1h</b>  →  Precision: <b>15m</b>"
                        )
                    elif cb_data == "cmd_conf_toggle":
                        self._confluence_enabled = not self._confluence_enabled
                        state = "ON" if self._confluence_enabled else "OFF"
                        logger.info(f"[CMD] Confluence filtering {state}")
                        self._answer_callback(cb_id, f"🎯 Confluence {state}")
                        self._send_control_panel()
                    elif cb_data.startswith("cmd_conf_"):
                        try:
                            new_min = int(cb_data.split("_")[-1])
                            if 1 <= new_min <= 5:
                                self._confluence_min = new_min
                                logger.info(f"[CMD] Confluence min set to {new_min}")
                                self._answer_callback(cb_id, f"🎯 Confluence min: {new_min}/5")
                                self._send_control_panel()
                        except (ValueError, IndexError):
                            pass

                    # ── Elite signal approval buttons ────────────────────
                    elif cb_data.startswith("elite_approve_"):
                        try:
                            pid = int(cb_data.split("elite_approve_")[1])
                            self._handle_signal_approve(pid)
                            self._answer_callback(cb_id, "✅ Signal approved & posted")
                        except (ValueError, IndexError, Exception) as e:
                            logger.warning(f"[CMD] elite_approve error: {e}")
                            self._answer_callback(cb_id, "Error approving signal")

                    elif cb_data.startswith("elite_skip_"):
                        try:
                            pid  = int(cb_data.split("elite_skip_")[1])
                            item = self._pending_signals.pop(pid, None)
                            sym  = item["symbol"] if item else "unknown"
                            self._answer_callback(cb_id, "⏭ Signal skipped")
                            logger.info(f"[ELITE] Signal #{pid} ({sym}) skipped by admin")
                        except (ValueError, IndexError):
                            pass

                    elif cb_data.startswith("elite_wait_"):
                        try:
                            pid = int(cb_data.split("elite_wait_")[1])
                            self._answer_callback(cb_id, "⏳ Signal held for re-evaluation")
                            logger.info(f"[ELITE] Signal #{pid} held by admin")
                        except (ValueError, IndexError):
                            pass

                    # ── Elite control buttons ─────────────────────────────
                    elif cb_data == "cmd_pause_elite":
                        self._elite_paused    = True
                        self._elite_resume_at = None
                        logger.info("[CMD] Elite scanning PAUSED by admin")
                        self._answer_callback(cb_id, "⏸ Elite scanning paused")
                        self._send_control_panel()

                    elif cb_data == "cmd_resume_elite":
                        self._elite_paused    = False
                        self._elite_resume_at = None
                        self._consecutive_sl  = 0
                        logger.info("[CMD] Elite scanning RESUMED by admin")
                        self._answer_callback(cb_id, "▶️ Elite scanning resumed")
                        self._send_control_panel()

                    elif cb_data == "cmd_reset_daily":
                        self._daily_elite_count = 0
                        logger.info("[CMD] Daily signal count reset")
                        self._answer_callback(cb_id, "🔄 Daily count reset to 0")
                        self._send_control_panel()

                    continue

                # ── Text command ─────────────────────────────────────────
                msg  = update.get("message", {})
                chat = str(msg.get("chat", {}).get("id", ""))
                text = msg.get("text", "").strip().lower()
                if not text:
                    continue
                logger.info(f"[CMD] Message from chat_id={chat} text='{text}' (admin={self._admin_id})")
                if chat != self._admin_id:
                    continue
                if text in ("/start", "/panel", "/help"):
                    self._send_control_panel()
                elif text == "/stop":
                    self.bybit.enabled = False
                    self._send_control_panel()
        except Exception as e:
            logger.warning(f"[CMD] Poll error: {e}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        paper_note = (
            f" | Paper: ON (balance={self.paper_balance:.0f} USDT, risk={self.risk_pct*100:.0f}% per trade)"
            if self.paper_enabled else " | Paper: OFF"
        )
        dp_cfg    = self.cfg.get("dynamic_pairs", {})
        dp_note   = f"Dynamic pairs: TOP {dp_cfg.get('top_n', 30)} by 24h volume | Refresh: every {dp_cfg.get('refresh_hours', 4)}h"
        logger.info("=" * 60)
        logger.info(f"Scanner started — {dp_note}{paper_note} | Mode: {self.mode.upper()}")
        logger.info(f"Trend TF: {self.tf_trend} | Structure TF: {self.tf_entry} | Precision TF: {self.tf_precision} | SR TF: {self.tf_sr}")
        logger.info(f"Cooldown: {self.cooldown_min}min | Summary: {self.daily_summary_hour:02d}:00 UTC")
        logger.info(f"Elite 4H system: daily_limit={self._daily_sig_limit} | max_concurrent={self._max_concurrent}")
        logger.info("=" * 60)

        try:
            symbols = self.pair_selector.get_symbols()
        except Exception as e:
            logger.error(f"Pair selector failed on startup: {e} — using fallback list")
            symbols = self.cfg.get("symbols", ["BTC/USDT:USDT", "ETH/USDT:USDT"])

        bybit_note = f" | MEXC: {'ON' if self.bybit.enabled else 'OFF'}"
        self.notifier.scanner_started(
            symbols, self.tf_trend, self.tf_entry,
            self.cooldown_min, self.paper_enabled, self.paper_balance,
            strategies=["Elite 4H BOS (10-point confluence scoring)"],
            label=f"Elite Crypto Futures Scanner{bybit_note}",
            mode=self.mode,
            confluence_min=self._confluence_min,
            tf_precision=self.tf_precision,
        )
        # Forex startup alert intentionally removed — forex bot runs as a separate service

        if self._admin_token and self._admin_id:
            logger.info(f"[CMD] Admin commands active (admin_id={self._admin_id}) — polling each tick")
            self._send_control_panel()
        else:
            logger.info("[CMD] ADMIN_BOT_TOKEN or TELEGRAM_ADMIN_ID not set — admin commands disabled")

        while self._running:
            try:
                self._poll_commands()
                self._check_mode_recommendation()
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
