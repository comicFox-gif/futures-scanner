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
from src.strategies.sr_bounce import SRBounceStrategy
from src.strategies.bollinger_breakout import BollingerBreakoutStrategy
from src.strategies.structure_break import StructureBreakStrategy
from src.strategies.macd_zero_cross import MACDZeroCrossStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.whale_momentum import WhaleMomentumStrategy
from src.strategies.vwap_pullback import VWAPPullbackStrategy
from src.pair_selector import PairSelector
from src.notifier import Notifier
from src.bybit_executor import BybitExecutor
from src.state_manager import save_state, load_state

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
        self.risk_pct: float = paper_cfg.get("risk_pct", 0.03)  # 3% of balance per trade
        self.paper_start_balance: float = self.paper_balance

        self.tf_sr: str = cfg.get("timeframe_sr", "4h")

        self.strategy    = Strategy(cfg)
        self.sr_strategy = SRBounceStrategy(cfg)
        self.bb_strategy = BollingerBreakoutStrategy(cfg)
        self.vp_strategy  = StructureBreakStrategy(cfg)
        self.mz_strategy  = MACDZeroCrossStrategy(cfg)
        self.rd_strategy = RSIDivergenceStrategy(cfg)
        self.wm_strategy   = WhaleMomentumStrategy(cfg)
        self.vwap_strategy = VWAPPullbackStrategy(cfg)
        self.notifier    = Notifier(
            channel_name=cfg.get("channel_name", ""),
            forex_symbols=set(cfg.get("forex_symbols", [])),
        )
        self.exchange    = self._init_exchange(cfg, env)
        self.bybit       = BybitExecutor(
            api_key       = env.get("BYBIT_KEY", ""),
            api_secret    = env.get("BYBIT_SECRET", ""),
            demo          = env.get("BYBIT_DEMO", "false").lower() == "true",
            testnet       = env.get("BYBIT_TESTNET", "false").lower() == "true",
            leverage      = int(env.get("BYBIT_LEVERAGE", "10")),
            risk_pct      = self.risk_pct,
            max_positions = 10,
            max_risk_usdt = float(env.get("BYBIT_MAX_RISK", "10")),
        )
        self.pair_selector = PairSelector(self.exchange, cfg)

        # Signal cooldown: (symbol, direction, stage) -> last alert time
        self._last_alert: dict[tuple, datetime] = {}

        # Paper positions: symbol -> Position
        self._paper_positions: dict[str, Position] = {}

        # Forex paper trading — parallel tracker, sends to forex channel
        forex_cfg = cfg.get("forex_paper", {})
        self.forex_paper_balance: float = forex_cfg.get("balance", 1000.0)
        self.forex_paper_start:   float = self.forex_paper_balance
        self._forex_positions:    dict[str, Position] = {}
        self._forex_stats = {"tp3": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}

        # Session tracking: 10 trades opened → 3h pause → reset
        self._session_count     = 0          # trades opened this session
        self._session_paused    = False
        self._resume_at         = None       # datetime when pause ends
        self._session_start_bal = self.paper_balance
        self._session_trades: list[dict] = []   # all closed trades this session

        # Lifetime trade stats (reset each session)
        self._trade_stats    = {"sl": 0, "tp3": 0, "be_sl": 0, "total": 0, "wins": 0}
        self._strategy_stats: dict[str, dict] = {}

        # Stats
        self._daily_alerts: list[dict] = []
        self._paper_trades: list[dict] = []
        self._last_summary_date         = None
        self._last_positions_report: datetime = datetime.utcnow()
        self._running = False

        # Admin command polling — uses dedicated ADMIN_BOT_TOKEN (separate from signal bot)
        self._admin_token  = os.getenv("ADMIN_BOT_TOKEN", "").strip()
        self._admin_id     = os.getenv("TELEGRAM_ADMIN_ID", "").strip()
        self._cmd_offset   = 0
        logger.info(f"[CMD] ADMIN_BOT_TOKEN={'SET' if self._admin_token else 'MISSING'} | TELEGRAM_ADMIN_ID={'SET' if self._admin_id else 'MISSING'}")

        # Restore paper state from previous session (survives redeploy)
        load_state(self)

    # ------------------------------------------------------------------
    # Exchange init
    # ------------------------------------------------------------------

    def _init_exchange(self, cfg: dict, env: dict):
        exchange_id = env.get("EXCHANGE", cfg.get("exchange", "bybit"))
        # Bybit uses "linear" for USDT perps; OKX uses "swap"
        type_map    = {"bybit": "linear", "okx": "swap"}
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
        """Fetch the current live mark/last price from Bybit via ccxt."""
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
            self._trade_stats       = {"sl": 0, "tp3": 0, "be_sl": 0, "total": 0, "wins": 0}
            self._strategy_stats    = {}
            self.notifier.send(
                f"🔄 <b>New Session Started</b>\n"
                f"Balance reset to <code>${self.paper_balance:.0f}</code> — scanning for signals..."
            )
            logger.info("[PAPER] Session reset — new 10-trade cycle started")

    def _paper_open(self, signal, strategy_name: str = "", live_price: float = None):
        """
        Open a simulated position from a confirmed signal.
        live_price: live Bybit ticker price — used as the actual entry price.
                    Falls back to signal.entry_price if not provided.
        """
        self._check_session_resume()

        if self._session_paused:
            return

        symbol = signal.symbol if hasattr(signal, "symbol") else signal["symbol"]
        if symbol in self._paper_positions:
            logger.debug(f"[PAPER] Already in position for {symbol}, skipping")
            return

        if self._session_count >= 50:
            return

        sig_entry = signal.entry_price if hasattr(signal, "entry_price") else signal["entry"]
        sig_sl    = signal.stop_loss   if hasattr(signal, "stop_loss")   else signal["sl"]
        sig_tp1   = signal.tp1         if hasattr(signal, "tp1")         else signal["tp1"]
        sig_tp2   = signal.tp2         if hasattr(signal, "tp2")         else signal["tp2"]
        sig_tp3   = signal.tp3         if hasattr(signal, "tp3")         else signal["tp3"]
        direction = signal.direction   if hasattr(signal, "direction")   else signal["direction"]

        # Use live Bybit price as entry; recalculate SL/TP offsets from it
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

        # Hard cap: max $5 SL exposure per paper trade
        risk_amount = round(min(self.paper_balance * self.risk_pct, 5.0), 2)
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
            f"@ {entry:.4f} (live) | Risk=${risk_amount:.2f} | Size={size:.4f} | "
            f"SL={sl:.4f} (-{sl_pct:.2f}%) | Available=${self.paper_balance:.2f} | "
            f"Session: {self._session_count}/10"
        )
        self.notifier.paper_opened(pos, self.paper_balance, open_count, self._session_count)
        save_state(self)

        # Stop opening new trades once 50 have been opened
        if self._session_count >= 10:
            self._session_paused = True  # no new entries — wait for all to close

    def _paper_close(self, symbol: str, exit_price: float, reason: str, tp_level: int = 0):
        """Close a paper position and send the closed alert. Used by tick and distribution exit."""
        pos = self._paper_positions.get(symbol)
        if not pos:
            return

        pnl = self._calc_pnl(pos, exit_price, pos.size_remaining)
        pos.closed_pnl += pnl
        self.paper_balance += pos.margin_locked + pnl

        if reason == "SL hit" and pos.be_activated:
            result = "be_sl";  self._trade_stats["be_sl"] += 1
        elif reason == "SL hit":
            result = "sl";     self._trade_stats["sl"]    += 1
        elif tp_level >= 2:
            result = "tp3";    self._trade_stats["tp3"] += 1
        else:
            result = "other"
        self._trade_stats["total"] += 1
        if pos.closed_pnl > 0:
            self._trade_stats["wins"] += 1

        sn = pos.strategy_name or "Unknown"
        if sn not in self._strategy_stats:
            self._strategy_stats[sn] = {"tp3": 0, "tp2": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}
        ss = self._strategy_stats[sn]
        ss["total"] += 1
        if result == "tp3":    ss["tp3"]   += 1
        elif result == "be_sl": ss["be_sl"] += 1
        elif result == "sl":    ss["sl"]    += 1
        if pos.closed_pnl > 0: ss["wins"]  += 1

        self._session_trades.append({"pnl": pos.closed_pnl, "result": result, "strategy": sn})
        del self._paper_positions[symbol]
        open_count = len(self._paper_positions)

        if self._session_count >= 10 and open_count == 0:
            self._send_session_summary()
            self._resume_at = datetime.utcnow() + timedelta(hours=3)
            logger.info("[PAPER] Session complete — 5h pause started")

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

    def _check_distribution(self, symbol: str, entry_df: pd.DataFrame, current_price: float):
        """
        After the whale entry, watch for exit signs.
        Longs: detect_distribution fires when whales are selling into the crowd.
        Shorts: detect_short_covering fires when whales are buying back (covering shorts).
        Sends a Telegram warning so subscribers can take profits. Fires once per position.
        """
        from src.indicators import detect_distribution, detect_short_covering
        pos = self._paper_positions.get(symbol)
        if not pos:
            return
        if getattr(pos, "_distribution_warned", False):
            return

        if pos.direction == "long":
            if current_price <= pos.entry_price:
                return
            signal = detect_distribution(entry_df)
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            exit_label = "distributing (selling into crowd)"
            direction_icon = "📉"
        else:  # short
            if current_price >= pos.entry_price:
                return
            signal = detect_short_covering(entry_df)
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
            exit_label = "covering shorts (buying back)"
            direction_icon = "📈"

        if signal:
            pos._distribution_warned = True
            logger.info(
                f"[WHALE EXIT] {pos.direction.upper()} exit signal on {symbol} | "
                f"PnL={pnl_pct:+.1f}% | {signal['reason_str']}"
            )
            self.notifier.send(
                f"⚠️ <b>Whale Exit — {symbol}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{direction_icon} Institutions appear to be <b>{exit_label}</b>\n"
                f"💰 Closing at: <b>{pnl_pct:+.1f}%</b>\n"
                f"──────────────────────────────\n"
                f"<i>{signal['reason_str']}</i>"
            )
            # Auto-close paper position
            if self.paper_enabled and symbol in self._paper_positions:
                self._paper_close(symbol, current_price, "Whale exit")
            # Auto-close Bybit live position
            if self.bybit.enabled:
                self.bybit.close_position(symbol, pos.direction)

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
                reason   = action.get("reason", "")
                tp_level = action.get("tp_level", 0)
                self._paper_close(symbol, current_price, reason, tp_level)
                return

            elif act == "notify_tp1":
                self.notifier.paper_tp1_alert(pos, current_price, tp_level=1)

            elif act == "notify_tp2":
                self.notifier.paper_tp1_alert(pos, current_price, tp_level=2)

            elif act == "close_partial":
                pct        = action["pct"]
                tp_level   = action.get("tp_level", 0)
                close_size = round(pos.size_remaining * pct, 6)
                if tp_level == 1:
                    partial_price = pos.tp1
                elif tp_level == 2:
                    partial_price = pos.tp2
                else:
                    partial_price = current_price
                pnl        = self._calc_pnl(pos, partial_price, close_size)
                pos.size_remaining -= close_size
                pos.closed_pnl += pnl
                self.paper_balance += pnl

                # Track TP2 per strategy
                if tp_level == 2:
                    sn = pos.strategy_name or "Unknown"
                    if sn not in self._strategy_stats:
                        self._strategy_stats[sn] = {"tp3": 0, "tp2": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}
                    self._strategy_stats[sn]["tp2"] += 1

                logger.info(
                    f"[PAPER] TP{tp_level} {symbol} | "
                    f"{pct*100:.0f}% closed @ {partial_price:.4f} | "
                    f"PnL={pnl:+.2f} | Balance={self.paper_balance:.2f}"
                )
                self.notifier.paper_tp_hit(pos, tp_level, partial_price, pnl, self.paper_balance)
                save_state(self)

            elif act == "move_sl":
                new_sl    = action["new_sl"]
                pos.stop_loss = new_sl
                be_note   = " → Break-Even" if new_sl == pos.entry_price else f" → {new_sl:.4f}"
                logger.info(f"[PAPER] SL moved{be_note} for {symbol}")
                self.notifier.paper_tp_hit(pos, 2, pos.tp2, 0, self.paper_balance)
                # Move SL to break-even on Bybit
                if self.bybit.enabled:
                    self.bybit.move_sl_to_breakeven(symbol, pos.direction, pos.entry_price)

    # ------------------------------------------------------------------
    # Forex paper trading (parallel tracker → forex channel)
    # ------------------------------------------------------------------

    def _forex_paper_open(self, signal, strategy_name: str = "", live_price: float = None):
        """Open a forex paper position mirroring the confirmed signal using live Bybit price."""
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

        risk_amount = round(min(self.forex_paper_balance * self.risk_pct, 5.0), 2)
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
                exit_price = pos.stop_loss if reason == "SL hit" else (pos.tp3 if tp_level == 3 else current_price)
                pnl = self._calc_pnl(pos, exit_price, pos.size_remaining)
                pos.closed_pnl += pnl
                self.forex_paper_balance += pos.margin_locked + pnl

                result = "tp3" if tp_level == 3 else ("sl" if reason == "SL hit" else "other")
                self._forex_stats["total"] += 1
                if tp_level == 3:        self._forex_stats["tp3"] += 1
                elif reason == "SL hit": self._forex_stats["sl"]  += 1
                if pos.closed_pnl > 0:   self._forex_stats["wins"] += 1

                del self._forex_positions[symbol]
                logger.info(
                    f"[FOREX PAPER] CLOSED {symbol} | {reason} @ {exit_price:.5f} "
                    f"| PnL={pnl:+.2f} | Balance={self.forex_paper_balance:.2f} "
                    f"| W:{self._forex_stats['wins']} TP3:{self._forex_stats['tp3']} SL:{self._forex_stats['sl']}"
                )
                return

            # TP hits — tracked silently, no forex channel notification

    def _bybit_order(self, sig, symbol: str = ""):
        """Place order on Bybit. Accepts Signal dataclass or dict."""
        if not self.bybit.enabled or self._session_paused:
            return
        if hasattr(sig, "entry_price"):   # Signal dataclass
            d = {"symbol": sig.symbol, "direction": sig.direction,
                 "entry": sig.entry_price, "sl": sig.stop_loss, "tp3": sig.tp3}
        else:
            # Dict signals don't include symbol — inject it from caller
            d = {**sig, "symbol": symbol}
        self.bybit.place_order(d)

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

    def _strategy_label(self, base_name: str, signal) -> str:
        """
        If the signal reason contains 'Bull Trap', prefix the label so Telegram
        subscribers know it's a trap fade, not a normal signal.
        """
        reason = signal.get("reason", "") if isinstance(signal, dict) else getattr(signal, "reason", "")
        if "Bull Trap" in reason:
            return f"🪤 Bull Trap Fade ({base_name})"
        return base_name

    def _bias_blocks(self, direction: str, bias: str, is_bull_trap: bool = False) -> bool:
        """
        Returns True if the market bias is strongly against this trade direction.
        Bearish market → block longs. Bullish market → block shorts.
        Neutral → allow both.
        Bull trap shorts are never blocked — fading a pump in a bullish market is the point.
        """
        if is_bull_trap:
            return False
        if bias == "bearish" and direction == "long":
            return True
        if bias == "bullish" and direction == "short":
            return True
        return False

    def _market_bias(self) -> str:
        """
        Determine overall crypto market direction using BTC as the benchmark.
        Returns 'bearish', 'bullish', or 'neutral'.

        Logic (3 votes — majority wins):
          Vote 1: BTC price vs EMA50 on HTF
          Vote 2: BTC price vs EMA200 on HTF
          Vote 3: BTC MACD line vs zero
        """
        try:
            btc_raw = self._fetch_ohlcv("BTC/USDT:USDT", self.tf_trend)
            if btc_raw is None or len(btc_raw) < 60:
                return "neutral"
            btc_df  = self.strategy.enrich(btc_raw.copy())
            if len(btc_df) < 5:
                return "neutral"
            row      = btc_df.iloc[-2]
            price    = float(row["close"])
            ema50    = float(row.get(f"ema_{self.cfg['strategy']['ema_slow']}", float("nan")))
            ema200   = float(row.get(f"ema_{self.cfg['strategy']['ema_trend']}", float("nan")))
            macd     = float(row.get("macd", float("nan")))
            if any(pd.isna(v) for v in [ema50, ema200, macd]):
                return "neutral"
            bull_votes = sum([price > ema50, price > ema200, macd > 0])
            bear_votes = sum([price < ema50, price < ema200, macd < 0])
            if bull_votes >= 2:
                return "bullish"
            if bear_votes >= 2:
                return "bearish"
            return "neutral"
        except Exception:
            return "neutral"

    def _check_early_whale(self, symbol: str, htf_df: "pd.DataFrame", current_price: float, market_bias: str):
        """
        Detect whale momentum on the fast confirmation TF before it shows on the entry TF.
        Scalp: 5m   |  Swing: 15m
        Fires an alert + paper/bybit order if institutional footprint detected early.
        """
        from src.indicators import detect_whale_entry, detect_whale_sell
        tf_fast = "5m" if self.mode == "scalp" else "15m"
        cooldown_key = symbol + "_early_whale"

        if self._is_on_cooldown(symbol, "early_whale", 2):
            return

        try:
            fast_raw = self._fetch_ohlcv(symbol, tf_fast)
            if fast_raw is None or len(fast_raw) < 30:
                return
            fast_df = self.strategy.enrich(fast_raw.copy())
            if len(fast_df) < 20:
                return

            cfg_wm  = self.cfg.get("whale_momentum", {})
            vol_mult   = cfg_wm.get("volume_multiplier", 2.5)
            body_min   = cfg_wm.get("body_min", 0.55)
            delta_mult = cfg_wm.get("delta_mult", 1.8)
            rsi_max    = cfg_wm.get("rsi_max", 62)
            rsi_min    = cfg_wm.get("rsi_min", 38)

            # HTF trend filter
            htf_row  = htf_df.iloc[-2]
            ema50    = float(htf_row.get(f"ema_{self.cfg['strategy']['ema_slow']}", float("nan")))
            ema200   = float(htf_row.get(f"ema_{self.cfg['strategy']['ema_trend']}", float("nan")))
            htf_bull = not pd.isna(ema50) and float(htf_row["close"]) > ema50
            htf_bear = not pd.isna(ema50) and float(htf_row["close"]) < ema50

            fast_row = fast_df.iloc[-2]
            rsi      = float(fast_row.get("rsi", 50))
            atr      = float(fast_row.get("atr", current_price * 0.005))

            # ── Long: bullish HTF + whale entry on fast TF ───────────────
            whale_long = detect_whale_entry(fast_df, vol_mult, body_min, delta_mult)
            if whale_long and htf_bull and rsi <= rsi_max and not self._bias_blocks("long", market_bias):
                sl  = current_price - atr * self.cfg["signal"].get("atr_sl_multiplier", 1.5)
                tp1 = current_price + atr * self.cfg["signal"].get("tp1_rr", 1.0) * self.cfg["signal"].get("atr_sl_multiplier", 1.5)
                tp2 = current_price + atr * self.cfg["signal"].get("tp2_rr", 2.0) * self.cfg["signal"].get("atr_sl_multiplier", 1.5)
                sig = {"symbol": symbol, "direction": "long", "entry": current_price,
                       "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp2,
                       "atr": atr, "rsi": rsi, "vol_ratio": whale_long["vol_ratio"],
                       "reason": f"⚡ Early Whale Long | {tf_fast} spike | {whale_long['reason']}", "stage": 2}
                logger.info(f"[EARLY🐋] LONG {symbol} @ {current_price:.4f} | {tf_fast} whale spike | RSI={rsi:.0f}")
                self.notifier.confirmed_signal(sig, f"⚡ Early 🐋 Whale ({tf_fast})", 5)
                self._bybit_order(sig, symbol)
                if self.paper_enabled and symbol not in self._paper_positions:
                    from src.strategy import Signal as Sig
                    dummy = Sig(stage=2, direction="long", symbol=symbol,
                                entry_price=current_price, stop_loss=sl,
                                tp1=tp1, tp2=tp2, tp3=tp2,
                                atr=atr, rsi=rsi, volume_ratio=whale_long["vol_ratio"],
                                reason=sig["reason"])
                    self._paper_open(dummy, "Early Whale", live_price=current_price)
                self._mark_sent(symbol, "early_whale", 2)
                return

            # ── Short: bearish HTF + whale sell on fast TF ───────────────
            whale_short = detect_whale_sell(fast_df, vol_mult, body_min, delta_mult)
            if whale_short and htf_bear and rsi >= rsi_min and not self._bias_blocks("short", market_bias):
                sl  = current_price + atr * self.cfg["signal"].get("atr_sl_multiplier", 1.5)
                tp1 = current_price - atr * self.cfg["signal"].get("tp1_rr", 1.0) * self.cfg["signal"].get("atr_sl_multiplier", 1.5)
                tp2 = current_price - atr * self.cfg["signal"].get("tp2_rr", 2.0) * self.cfg["signal"].get("atr_sl_multiplier", 1.5)
                sig = {"symbol": symbol, "direction": "short", "entry": current_price,
                       "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp2,
                       "atr": atr, "rsi": rsi, "vol_ratio": whale_short["vol_ratio"],
                       "reason": f"⚡ Early Whale Short | {tf_fast} spike | {whale_short['reason']}", "stage": 2}
                logger.info(f"[EARLY🐋] SHORT {symbol} @ {current_price:.4f} | {tf_fast} whale spike | RSI={rsi:.0f}")
                self.notifier.confirmed_signal(sig, f"⚡ Early 🐋 Whale ({tf_fast})", 5)
                self._bybit_order(sig, symbol)
                if self.paper_enabled and symbol not in self._paper_positions:
                    from src.strategy import Signal as Sig
                    dummy = Sig(stage=2, direction="short", symbol=symbol,
                                entry_price=current_price, stop_loss=sl,
                                tp1=tp1, tp2=tp2, tp3=tp2,
                                atr=atr, rsi=rsi, volume_ratio=whale_short["vol_ratio"],
                                reason=sig["reason"])
                    self._paper_open(dummy, "Early Whale", live_price=current_price)
                self._mark_sent(symbol, "early_whale", 2)

        except Exception as e:
            logger.warning(f"[EARLY🐋] {symbol} error: {e}")

    def _mtf_confirm(self, symbol: str, direction: str) -> bool:
        """
        3rd timeframe confirmation before entering a signal.
        Scalp: confirm on 5m  — trend (EMA50) + rejection candle.
        Swing: confirm on 15m — trend (EMA50) + rejection candle.
        Returns True if confirmed, False to skip.
        """
        tf_confirm = "5m" if self.mode == "scalp" else "15m"
        try:
            raw = self._fetch_ohlcv(symbol, tf_confirm)
            if raw is None or len(raw) < 20:
                return True  # can't confirm — don't block the signal
            df  = self.strategy.enrich(raw.copy())
            if len(df) < 5:
                return True
            row      = df.iloc[-2]
            close    = float(row["close"])
            open_    = float(row["open"])
            ema50_col = f"ema_{self.cfg['strategy']['ema_slow']}"
            ema50    = float(row.get(ema50_col, float("nan")))
            if pd.isna(ema50):
                return True

            if direction == "short":
                trend_ok      = close < ema50          # price below EMA50 = downtrend
                rejection_ok  = close < open_          # bearish candle = rejection confirmed
            else:
                trend_ok      = close > ema50          # price above EMA50 = uptrend
                rejection_ok  = close > open_          # bullish candle = continuation confirmed

            confirmed = trend_ok and rejection_ok
            if not confirmed:
                logger.info(f"[MTF] {symbol} {direction.upper()} blocked on {tf_confirm} — trend_ok={trend_ok} rejection_ok={rejection_ok}")
            return confirmed
        except Exception as e:
            logger.warning(f"[MTF] {symbol} confirm error: {e} — allowing signal")
            return True

    def _tick(self):
        self._check_session_resume()
        symbols   = self.pair_selector.get_symbols()
        now       = datetime.utcnow().strftime("%H:%M:%S")
        open_pos  = len(self._paper_positions)
        paper_bal = f"${self.paper_balance:.2f}" if self.paper_enabled else ""
        paper_info = f" | Paper: {paper_bal} | Positions: {open_pos}" if self.paper_enabled else ""

        # Market regime — checked once per tick using BTC as benchmark
        market_bias = self._market_bias()
        bias_icon   = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(market_bias, "⚪")
        logger.info(f">>> Scanning {len(symbols)} pairs @ {now} UTC{paper_info} | Market: {bias_icon} {market_bias.upper()}")

        signals_found = 0
        for symbol in symbols:
            try:
                htf_raw   = self._fetch_ohlcv(symbol, self.tf_trend)
                entry_raw = self._fetch_ohlcv(symbol, self.tf_entry)
                if htf_raw is None or entry_raw is None:
                    continue

                # Fetch 4H data for S/R strategy
                sr_raw = self._fetch_ohlcv(symbol, self.tf_sr)

                htf_df   = self.strategy.enrich(htf_raw.copy())
                entry_df = self.strategy.enrich(entry_raw.copy())
                sr_df    = self.sr_strategy.enrich(sr_raw.copy()) if sr_raw is not None else None

                # Skip symbols without enough candle history.
                # OKX caps at 300 candles per call; after EMA200 dropna ~100 rows remain.
                # 60 is enough for all indicators (MACD needs 35, RSI needs 14).
                min_candles = 60
                if len(htf_df) < min_candles or len(entry_df) < min_candles:
                    logger.debug(f"Skipping {symbol}: insufficient candle history ({len(htf_df)} htf, {len(entry_df)} entry)")
                    continue

                # Use live Bybit price for paper trade monitoring and entry.
                # Falls back to last closed candle only if ticker fetch fails.
                live_price    = self._fetch_live_price(symbol)
                current_price = live_price if live_price else float(entry_df.iloc[-2]["close"])

                # Paper position management (always runs if position is open)
                if self.paper_enabled and symbol in self._paper_positions:
                    self._paper_tick(symbol, current_price)
                    # Distribution detection — warn subscribers when whales appear to be exiting
                    self._check_distribution(symbol, entry_df, current_price)
                if symbol in self._forex_positions:
                    self._forex_paper_tick(symbol, current_price)

                in_paper = self.paper_enabled and symbol in self._paper_positions

                # ── Strategy 1: EMA Momentum ──────────────────────────────
                if not in_paper:
                    signal = self.strategy.generate_signal(symbol, htf_df, entry_df)

                    if signal.stage == 2 and not self._is_on_cooldown(symbol, signal.direction + "_ema", signal.stage):
                        quality = 5  # EMA requires all 5 conditions
                        logger.info(
                            f"[EMA CONFIRMED] {signal.direction.upper()} {symbol} "
                            f"@ {signal.entry_price:.4f} | RSI={signal.rsi:.1f} | Q={quality} | {signal.reason}"
                        )
                        if not self._session_paused and quality >= 5 and not self._bias_blocks(signal.direction, market_bias) and self._mtf_confirm(symbol, signal.direction):
                            self.notifier.confirmed_signal(signal, self._strategy_label("EMA Momentum", signal), quality)
                            if self.paper_enabled:
                                self._paper_open(signal, "EMA Momentum", live_price=current_price)
                            self._forex_paper_open(signal, "EMA Momentum", live_price=current_price)
                            self._bybit_order(signal)
                        else:
                            logger.info(f"[EMA] Skipped {symbol} — quality {quality} < 5 or bias blocked ({market_bias})")

                        self._mark_sent(symbol, signal.direction + "_ema", signal.stage)
                        self._daily_alerts.append({"stage": signal.stage, "direction": signal.direction, "symbol": symbol})
                        signals_found += 1

                # ── Strategy 2: S/R Bounce ────────────────────────────────
                if sr_df is not None and not in_paper:
                    sr_sig = self.sr_strategy.generate_signal(symbol, sr_df, entry_df)

                    if sr_sig and sr_sig["stage"] == 2 and not self._is_on_cooldown(symbol, sr_sig["direction"] + "_sr", sr_sig["stage"]):
                        q = sr_sig.get("quality", 3)
                        logger.info(
                            f"[SR CONFIRMED] {sr_sig['direction'].upper()} {symbol} "
                            f"@ {sr_sig['entry']:.4f} | Q={q} | {sr_sig['reason']}"
                        )
                        if not self._session_paused and q >= 5 and not self._bias_blocks(sr_sig["direction"], market_bias) and self._mtf_confirm(symbol, sr_sig["direction"]):
                            if "Bull Trap" in sr_sig.get("reason", ""):
                                self.notifier.confirmed_signal(sr_sig, self._strategy_label("S/R Bounce", sr_sig), q)
                            else:
                                self.notifier.sr_confirmed_signal(sr_sig)
                            self._bybit_order(sr_sig, symbol)
                            if symbol not in self._forex_positions:
                                self._forex_paper_open(sr_sig, "S/R Bounce", live_price=current_price)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(
                                    stage=2, direction=sr_sig["direction"], symbol=symbol,
                                    entry_price=sr_sig["entry"], stop_loss=sr_sig["sl"],
                                    tp1=sr_sig["tp1"], tp2=sr_sig["tp2"], tp3=sr_sig["tp3"],
                                    atr=sr_sig["atr"], rsi=sr_sig["rsi"],
                                    volume_ratio=sr_sig.get("vol_ratio", 0),
                                    reason=sr_sig.get("reason", ""),
                                )
                                self._paper_open(dummy, "S/R Bounce", live_price=current_price)
                        else:
                            logger.info(f"[SR] Skipped {symbol} — quality {q} < 5")

                        self._mark_sent(symbol, sr_sig["direction"] + "_sr", sr_sig["stage"])
                        self._daily_alerts.append({"stage": sr_sig["stage"], "direction": sr_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 3: BB Squeeze Breakout ───────────────────────
                if not in_paper:
                    bb_sig = self.bb_strategy.generate_signal(symbol, htf_df, entry_df)
                    if bb_sig and bb_sig["stage"] == 2 and not self._is_on_cooldown(symbol, bb_sig["direction"] + "_bb", bb_sig["stage"]):
                        q = bb_sig.get("quality", 3)
                        logger.info(
                            f"[BB CONFIRMED] {bb_sig['direction'].upper()} {symbol} "
                            f"@ {bb_sig['entry']:.4f} | Q={q} | {bb_sig['reason']}"
                        )
                        bb_trap = "Bull Trap" in bb_sig.get("reason", "")
                        if not self._session_paused and q >= 5 and not self._bias_blocks(bb_sig["direction"], market_bias, is_bull_trap=bb_trap) and self._mtf_confirm(symbol, bb_sig["direction"]):
                            self.notifier.confirmed_signal(bb_sig, self._strategy_label("BB Breakout", bb_sig), q)
                            self._bybit_order(bb_sig, symbol)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=bb_sig["direction"], symbol=symbol,
                                            entry_price=bb_sig["entry"], stop_loss=bb_sig["sl"],
                                            tp1=bb_sig["tp1"], tp2=bb_sig["tp2"], tp3=bb_sig["tp3"],
                                            atr=bb_sig["atr"], rsi=bb_sig["rsi"],
                                            volume_ratio=bb_sig.get("vol_ratio", 0),
                                            reason=bb_sig.get("reason", ""))
                                self._paper_open(dummy, "BB Breakout", live_price=current_price)
                        else:
                            reason = "paused" if self._session_paused else ("bias blocked" if self._bias_blocks(bb_sig["direction"], market_bias) else f"quality {q} < 5")
                            logger.info(f"[BB] Skipped {symbol} — {reason}")
                        self._mark_sent(symbol, bb_sig["direction"] + "_bb", bb_sig["stage"])
                        self._daily_alerts.append({"stage": bb_sig["stage"], "direction": bb_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 4: Break of Structure ────────────────────────
                if not in_paper:
                    vp_sig = self.vp_strategy.generate_signal(symbol, htf_df, entry_df)
                    if vp_sig and vp_sig["stage"] == 2 and not self._is_on_cooldown(symbol, vp_sig["direction"] + "_vp", vp_sig["stage"]):
                        q = vp_sig.get("quality", 3)
                        logger.info(
                            f"[BOS CONFIRMED] {vp_sig['direction'].upper()} {symbol} "
                            f"@ {vp_sig['entry']:.4f} | Q={q} | {vp_sig['reason']}"
                        )
                        vp_trap = "Bull Trap" in vp_sig.get("reason", "")
                        if not self._session_paused and q >= 5 and not self._bias_blocks(vp_sig["direction"], market_bias, is_bull_trap=vp_trap) and self._mtf_confirm(symbol, vp_sig["direction"]):
                            self.notifier.confirmed_signal(vp_sig, self._strategy_label("Break of Structure", vp_sig), q)
                            self._bybit_order(vp_sig, symbol)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=vp_sig["direction"], symbol=symbol,
                                            entry_price=vp_sig["entry"], stop_loss=vp_sig["sl"],
                                            tp1=vp_sig["tp1"], tp2=vp_sig["tp2"], tp3=vp_sig["tp3"],
                                            atr=vp_sig["atr"], rsi=vp_sig["rsi"],
                                            volume_ratio=vp_sig.get("vol_ratio", 0),
                                            reason=vp_sig.get("reason", ""))
                                self._paper_open(dummy, "Break of Structure", live_price=current_price)
                        else:
                            reason = "paused" if self._session_paused else ("bias blocked" if self._bias_blocks(vp_sig["direction"], market_bias) else f"quality {q} < 5")
                            logger.info(f"[BOS] Skipped {symbol} — {reason}")
                        self._mark_sent(symbol, vp_sig["direction"] + "_vp", vp_sig["stage"])
                        self._daily_alerts.append({"stage": vp_sig["stage"], "direction": vp_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 5: RSI Divergence ────────────────────────────
                if not in_paper:
                    rd_sig = self.rd_strategy.generate_signal(symbol, htf_df, entry_df)
                    if rd_sig and rd_sig["stage"] == 2 and not self._is_on_cooldown(symbol, rd_sig["direction"] + "_rd", rd_sig["stage"]):
                        q = rd_sig.get("quality", 3)
                        logger.info(
                            f"[DIV CONFIRMED] {rd_sig['direction'].upper()} {symbol} "
                            f"@ {rd_sig['entry']:.4f} | Q={q} | {rd_sig['reason']}"
                        )
                        if not self._session_paused and q >= 5 and not self._bias_blocks(rd_sig["direction"], market_bias) and self._mtf_confirm(symbol, rd_sig["direction"]):
                            self.notifier.confirmed_signal(rd_sig, self._strategy_label("RSI Divergence", rd_sig), q)
                            self._bybit_order(rd_sig, symbol)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=rd_sig["direction"], symbol=symbol,
                                            entry_price=rd_sig["entry"], stop_loss=rd_sig["sl"],
                                            tp1=rd_sig["tp1"], tp2=rd_sig["tp2"], tp3=rd_sig["tp3"],
                                            atr=rd_sig["atr"], rsi=rd_sig["rsi"],
                                            volume_ratio=rd_sig.get("vol_ratio", 0),
                                            reason=rd_sig.get("reason", ""))
                                self._paper_open(dummy, "RSI Divergence", live_price=current_price)
                        else:
                            reason = "paused" if self._session_paused else ("bias blocked" if self._bias_blocks(rd_sig["direction"], market_bias) else f"quality {q} < 5")
                            logger.info(f"[DIV] Skipped {symbol} — {reason}")
                        self._mark_sent(symbol, rd_sig["direction"] + "_rd", rd_sig["stage"])
                        self._daily_alerts.append({"stage": rd_sig["stage"], "direction": rd_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 6: MACD Zero Cross ───────────────────────────
                if not in_paper:
                    mz_sig = self.mz_strategy.generate_signal(symbol, htf_df, entry_df)
                    if mz_sig and mz_sig["stage"] == 2 and not self._is_on_cooldown(symbol, mz_sig["direction"] + "_mz", mz_sig["stage"]):
                        q = mz_sig.get("quality", 3)
                        logger.info(
                            f"[MACD0 CONFIRMED] {mz_sig['direction'].upper()} {symbol} "
                            f"@ {mz_sig['entry']:.4f} | Q={q} | {mz_sig['reason']}"
                        )
                        if not self._session_paused and q >= 5 and not self._bias_blocks(mz_sig["direction"], market_bias) and self._mtf_confirm(symbol, mz_sig["direction"]):
                            self.notifier.confirmed_signal(mz_sig, self._strategy_label("MACD Zero Cross", mz_sig), q)
                            self._bybit_order(mz_sig, symbol)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=mz_sig["direction"], symbol=symbol,
                                            entry_price=mz_sig["entry"], stop_loss=mz_sig["sl"],
                                            tp1=mz_sig["tp1"], tp2=mz_sig["tp2"], tp3=mz_sig["tp3"],
                                            atr=mz_sig["atr"], rsi=mz_sig["rsi"],
                                            volume_ratio=mz_sig.get("vol_ratio", 0),
                                            reason=mz_sig.get("reason", ""))
                                self._paper_open(dummy, "MACD Zero Cross", live_price=current_price)
                        else:
                            reason = "paused" if self._session_paused else ("bias blocked" if self._bias_blocks(mz_sig["direction"], market_bias) else f"quality {q} < 5")
                            logger.info(f"[MACD0] Skipped {symbol} — {reason}")
                        self._mark_sent(symbol, mz_sig["direction"] + "_mz", mz_sig["stage"])
                        self._daily_alerts.append({"stage": mz_sig["stage"], "direction": mz_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 7: Whale Momentum ────────────────────────────
                if not in_paper:
                    wm_sig = self.wm_strategy.generate_signal(symbol, htf_df, entry_df)
                    if wm_sig and wm_sig["stage"] == 2 and not self._is_on_cooldown(symbol, wm_sig["direction"] + "_wm", wm_sig["stage"]):
                        q = wm_sig.get("quality", 5)
                        logger.info(
                            f"[WHALE CONFIRMED] {wm_sig['direction'].upper()} {symbol} "
                            f"@ {wm_sig['entry']:.4f} | Q={q} | {wm_sig['reason']}"
                        )
                        if not self._session_paused and q >= 5 and not self._bias_blocks(wm_sig["direction"], market_bias) and self._mtf_confirm(symbol, wm_sig["direction"]):
                            self.notifier.confirmed_signal(wm_sig, "🐋 Whale Momentum", q)
                            self._bybit_order(wm_sig, symbol)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=wm_sig["direction"], symbol=symbol,
                                            entry_price=wm_sig["entry"], stop_loss=wm_sig["sl"],
                                            tp1=wm_sig["tp1"], tp2=wm_sig["tp2"], tp3=wm_sig["tp3"],
                                            atr=wm_sig["atr"], rsi=wm_sig["rsi"],
                                            volume_ratio=wm_sig.get("vol_ratio", 0),
                                            reason=wm_sig.get("reason", ""))
                                self._paper_open(dummy, "Whale Momentum", live_price=current_price)
                        else:
                            reason = "paused" if self._session_paused else ("bias blocked" if self._bias_blocks(wm_sig["direction"], market_bias) else f"quality {q} < 5")
                            logger.info(f"[WHALE] Skipped {symbol} — {reason}")
                        self._mark_sent(symbol, wm_sig["direction"] + "_wm", wm_sig["stage"])
                        self._daily_alerts.append({"stage": wm_sig["stage"], "direction": wm_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 8: VWAP Pullback (swing only) ───────────────
                if not in_paper and self.mode != "scalp":
                    vwap_sig = self.vwap_strategy.generate_signal(symbol, htf_df, entry_df)
                    if vwap_sig and vwap_sig["stage"] == 2 and not self._is_on_cooldown(symbol, vwap_sig["direction"] + "_vwap", vwap_sig["stage"]):
                        q = vwap_sig.get("quality", 3)
                        logger.info(
                            f"[VWAP CONFIRMED] {vwap_sig['direction'].upper()} {symbol} "
                            f"@ {vwap_sig['entry']:.4f} | Q={q} | {vwap_sig['reason']}"
                        )
                        vwap_trap = "Bull Trap" in vwap_sig.get("reason", "")
                        if not self._session_paused and q >= 5 and not self._bias_blocks(vwap_sig["direction"], market_bias, is_bull_trap=vwap_trap) and self._mtf_confirm(symbol, vwap_sig["direction"]):
                            self.notifier.confirmed_signal(vwap_sig, self._strategy_label("VWAP Pullback", vwap_sig), q)
                            self._bybit_order(vwap_sig, symbol)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=vwap_sig["direction"], symbol=symbol,
                                            entry_price=vwap_sig["entry"], stop_loss=vwap_sig["sl"],
                                            tp1=vwap_sig["tp1"], tp2=vwap_sig["tp2"], tp3=vwap_sig["tp3"],
                                            atr=vwap_sig["atr"], rsi=vwap_sig["rsi"],
                                            volume_ratio=vwap_sig.get("vol_ratio", 0),
                                            reason=vwap_sig.get("reason", ""))
                                self._paper_open(dummy, "VWAP Pullback", live_price=current_price)
                        else:
                            reason = "paused" if self._session_paused else ("bias blocked" if self._bias_blocks(vwap_sig["direction"], market_bias, is_bull_trap=vwap_trap) else f"quality {q} < 5")
                            logger.info(f"[VWAP] Skipped {symbol} — {reason}")
                        self._mark_sent(symbol, vwap_sig["direction"] + "_vwap", vwap_sig["stage"])
                        self._daily_alerts.append({"stage": vwap_sig["stage"], "direction": vwap_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Early Whale Detection (fast TF) ──────────────────────
                if not in_paper:
                    self._check_early_whale(symbol, htf_df, current_price, market_bias)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}\n{traceback.format_exc()}")
                self.notifier.error_alert(f"Scanning {symbol}", str(e)[:200])

        signal_note = f" | {signals_found} signal(s) fired" if signals_found > 0 else " | No signals"
        logger.info(f"<<< Scan complete{signal_note} | Next scan in {self.poll_interval}s")
        logger.info(f"    Pairs: {' | '.join(s.split('/')[0] for s in symbols)}")

        self._maybe_send_daily_summary()
        self._maybe_send_positions_report()

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
        bybit_btn = "⏹ Stop Trading" if self.bybit.enabled else "▶️ Start Trading"
        bybit_cb  = "cmd_stop" if self.bybit.enabled else "cmd_start"
        return {
            "inline_keyboard": [
                [{"text": bybit_btn, "callback_data": bybit_cb}],
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
        bybit_state = "🟢 ON" if self.bybit.enabled else "🔴 OFF"
        paper_state = "🟢 ON" if self.paper_enabled else "🔴 OFF"
        open_pos    = len(self._paper_positions)
        balance     = f"${self.paper_balance:.2f}" if self.paper_enabled else "N/A"
        text = (
            f"🤖 <b>Bot Control Panel</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Bybit execution: {bybit_state}\n"
            f"Paper trading:   {paper_state} ({balance})\n"
            f"Open positions:  {open_pos}\n"
            f"Mode: {self.mode.upper()}"
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
                        logger.info("[CMD] Bybit execution DISABLED via button")
                        self._answer_callback(cb_id, "⏹ Trading stopped")
                        self._send_control_panel()
                    elif cb_data == "cmd_start":
                        self.bybit.enabled = True
                        logger.info("[CMD] Bybit execution ENABLED via button")
                        self._answer_callback(cb_id, "▶️ Trading started")
                        self._send_control_panel()
                    elif cb_data == "cmd_status":
                        self._answer_callback(cb_id)
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
        logger.info(f"Trend TF: {self.tf_trend} | Entry TF: {self.tf_entry} | SR TF: {self.tf_sr}")
        logger.info(f"Cooldown: {self.cooldown_min}min | Summary: {self.daily_summary_hour:02d}:00 UTC")
        logger.info("=" * 60)

        try:
            symbols = self.pair_selector.get_symbols()
        except Exception as e:
            logger.error(f"Pair selector failed on startup: {e} — using fallback list")
            symbols = self.cfg.get("symbols", ["BTC/USDT:USDT", "ETH/USDT:USDT"])

        bybit_note = f" | Bybit: {'ON' if self.bybit.enabled else 'OFF'}"
        strat_list = ["EMA Trend", "S/R Bounce", "BB Breakout", "BOS", "RSI Divergence", "MACD Zero Cross", "🐋 Whale Momentum"]
        if self.mode != "scalp":
            strat_list.append("VWAP Pullback")
        self.notifier.scanner_started(
            symbols, self.tf_trend, self.tf_entry,
            self.cooldown_min, self.paper_enabled, self.paper_balance,
            strategies=strat_list,
            label=f"Crypto Futures Scanner{bybit_note}",
            mode=self.mode,
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
