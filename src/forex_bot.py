"""
Forex Signal Scanner + Paper Trading
--------------------------------------
Scans 20 forex pairs every 60s across two strategies:
  Strategy 1: FX EMA Trend    — 4H/1H/15m multi-timeframe momentum
  Strategy 2: London Breakout — Asian range breakout at London open

Data source: yfinance (free, no API key, globally accessible)

Alert stages:
  Stage 1 WARNING   — setup forming, prepare
  Stage 2 CONFIRMED — enter manually
"""

from __future__ import annotations
import time
import logging
import traceback
from datetime import datetime, date, timedelta
from typing import Optional

from src.forex_data import fetch_ohlcv
from src.forex_pair_selector import ForexPairSelector
from src.strategies.forex_ema_trend import ForexEmaTrendStrategy
from src.strategies.london_breakout import LondonBreakoutStrategy
from src.strategies.fibonacci_pullback import FibonacciPullbackStrategy
from src.strategies.daily_level_break import DailyLevelBreakStrategy
from src.strategies.bb_mean_reversion import BBMeanReversionStrategy
from src.notifier import Notifier
from src.strategy import Position   # reuse Position dataclass

logger = logging.getLogger("forex_bot")

MIN_CANDLES = 210  # EMA200 + safety buffer


# ------------------------------------------------------------------
# Paper position management (self-contained, mirrors bot.py logic)
# ------------------------------------------------------------------

def _check_position(pos: Position, current_price: float) -> list[dict]:
    tp1_pct = 0.30
    tp2_pct = 0.30
    actions = []

    if pos.direction == "long":
        if current_price <= pos.stop_loss:
            actions.append({"action": "close_all", "reason": "SL hit"})
            return actions
        if not pos.tp3_hit and current_price >= pos.tp3:
            actions.append({"action": "close_all", "reason": "TP3 hit", "tp_level": 3})
            pos.tp3_hit = True
            return actions
        if not pos.tp2_hit and current_price >= pos.tp2:
            actions.append({"action": "move_sl", "new_sl": pos.entry_price, "reason": "SL to Break-Even"})
            pos.tp2_hit = True
            pos.be_activated = True
        elif not pos.tp1_hit and current_price >= pos.tp1:
            actions.append({"action": "notify_tp1"})
            pos.tp1_hit = True
    else:
        if current_price >= pos.stop_loss:
            actions.append({"action": "close_all", "reason": "SL hit"})
            return actions
        if not pos.tp3_hit and current_price <= pos.tp3:
            actions.append({"action": "close_all", "reason": "TP3 hit", "tp_level": 3})
            pos.tp3_hit = True
            return actions
        if not pos.tp2_hit and current_price <= pos.tp2:
            actions.append({"action": "move_sl", "new_sl": pos.entry_price, "reason": "SL to Break-Even"})
            pos.tp2_hit = True
            pos.be_activated = True
        elif not pos.tp1_hit and current_price <= pos.tp1:
            actions.append({"action": "notify_tp1"})
            pos.tp1_hit = True

    return actions


class ForexBot:
    def __init__(self, cfg: dict):
        # Mode switch: "scalp" swaps signal + filter params before strategies load
        self.mode = cfg.get("mode", "swing")
        self.send_warnings = (self.mode != "scalp")   # scalp = confirmed only
        if self.mode == "scalp":
            if "scalp_signal" in cfg:
                cfg = {**cfg, "signal": cfg["scalp_signal"]}
            if "scalp_filters" in cfg:
                cfg = {**cfg, "filters": cfg["scalp_filters"]}
        self.cfg             = cfg
        self.pair_selector   = ForexPairSelector(cfg)
        self.tf_htf: str     = cfg.get("timeframe_htf", "4h")
        self.tf_trend: str   = cfg["timeframe_trend"]     # 1h
        self.tf_entry: str   = cfg["timeframe_entry"]     # 15m
        self.lookback: int   = cfg["strategy"]["lookback_candles"]
        self.poll_interval   = cfg["bot"]["poll_interval_seconds"]
        self.daily_sum_hour  = cfg["bot"].get("daily_summary_utc_hour", 0)
        self.cooldown_min    = cfg["signal"].get("signal_cooldown_minutes", 240)

        # Paper trading
        paper_cfg = cfg.get("paper_trading", {})
        self.paper_enabled     = paper_cfg.get("enabled", False)
        self.paper_balance     = paper_cfg.get("balance", 1000.0)
        self.risk_pct  = paper_cfg.get("risk_pct", 0.03)  # 3% of balance per trade
        self.paper_start_bal   = self.paper_balance

        self.alerts_enabled = cfg.get("alerts_enabled", True)
        self.ema_strategy  = ForexEmaTrendStrategy(cfg)
        self.lb_strategy   = LondonBreakoutStrategy(cfg)
        self.fib_strategy  = FibonacciPullbackStrategy(cfg)
        self.dl_strategy   = DailyLevelBreakStrategy(cfg)
        self.bbmr_strategy = BBMeanReversionStrategy(cfg)
        self.notifier      = Notifier(channel_name=cfg.get("channel_name", ""))
        # Forex bot always sends to the forex channel.
        # Swap main token → forex token so send() and send_signal() both hit the forex channel.
        if self.notifier.forex_enabled:
            self.notifier.token   = self.notifier.forex_token
            self.notifier.chat_id = self.notifier.forex_chat_id
            self.notifier.enabled = True
        if not self.alerts_enabled:
            self.notifier.enabled = False

        self._last_alert: dict[tuple, datetime] = {}
        self._paper_positions: dict[str, Position] = {}
        self._session_count     = 0
        self._session_paused    = False
        self._resume_at         = None
        self._session_start_bal = self.paper_balance
        self._session_trades: list[dict] = []
        self._trade_stats    = {"sl": 0, "tp3": 0, "be_sl": 0, "total": 0, "wins": 0}
        self._strategy_stats: dict[str, dict] = {}
        self._daily_alerts: list[dict] = []
        self._paper_trades: list[dict] = []
        self._last_summary_date         = None
        self._last_positions_report: datetime = datetime.utcnow()
        self._running = False

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, pair: str, key: str, stage: int) -> bool:
        last = self._last_alert.get((pair, key, stage))
        if last is None:
            return False
        return datetime.utcnow() - last < timedelta(minutes=self.cooldown_min)

    def _mark_sent(self, pair: str, key: str, stage: int):
        self._last_alert[(pair, key, stage)] = datetime.utcnow()

    # ------------------------------------------------------------------
    # Paper trading
    # ------------------------------------------------------------------

    def _check_session_resume(self):
        if self._session_paused and self._resume_at and datetime.utcnow() >= self._resume_at:
            self._session_paused    = False
            self._resume_at         = None
            self._session_count     = 0
            self._session_trades    = []
            self._paper_positions   = {}
            self.paper_balance      = self.paper_start_bal
            self._session_start_bal = self.paper_balance
            self._trade_stats       = {"sl": 0, "tp3": 0, "be_sl": 0, "total": 0, "wins": 0}
            self._strategy_stats    = {}
            self.notifier.send(
                f"🔄 <b>New Session Started</b>\n"
                f"Balance reset to <code>${self.paper_balance:.0f}</code> — scanning for signals..."
            )
            logger.info("[PAPER-FX] Session reset — new 50-trade cycle started")

    def _paper_open(self, sig: dict, strategy_name: str = ""):
        self._check_session_resume()
        if self._session_paused:
            return
        pair = sig["symbol"]
        if pair in self._paper_positions:
            return
        if self._session_count >= 50:
            return
        sl_dist = abs(sig["entry"] - sig["sl"])
        if sl_dist == 0:
            return

        # Hard cap: max $5 SL exposure per paper trade
        risk_amount = round(min(self.paper_balance * self.risk_pct, 5.0), 2)
        size = round(risk_amount / sl_dist, 6)
        if size <= 0:
            return

        self.paper_balance -= risk_amount   # lock margin immediately

        pos = Position(
            symbol=pair,
            direction=sig["direction"],
            entry_price=sig["entry"],
            stop_loss=sig["sl"],
            tp1=sig["tp1"],
            tp2=sig["tp2"],
            tp3=sig["tp3"],
            size=size,
            size_remaining=size,
            margin_locked=risk_amount,
            strategy_name=strategy_name,
        )
        self._paper_positions[pair] = pos
        self._session_count += 1
        sl_pct = sl_dist / sig["entry"] * 100
        open_count = len(self._paper_positions)
        logger.info(
            f"[PAPER-FX] OPENED {sig['direction'].upper()} {pair} "
            f"@ {sig['entry']:.5f} | Risk=${risk_amount:.2f} | Size={size:.6f} | "
            f"SL={sig['sl']:.5f} (-{sl_pct:.2f}%) | Session: {self._session_count}/50"
        )
        self.notifier.paper_opened(pos, self.paper_balance, open_count, self._session_count)

        if self._session_count >= 50:
            self._session_paused = True

    def _paper_tick(self, pair: str, price: float):
        pos = self._paper_positions.get(pair)
        if not pos:
            return
        actions = _check_position(pos, price)
        for action in actions:
            act = action["action"]
            if act == "close_all":
                tp_level     = action.get("tp_level", 0)
                close_reason = action.get("reason", "")
                # Use live market price as exit — simulates real fill
                exit_price = price
                pnl = self._pnl(pos, exit_price, pos.size_remaining)
                pos.closed_pnl += pnl
                self.paper_balance += pos.margin_locked + pnl

                if tp_level == 3:
                    self._trade_stats["tp3"] += 1
                    result = "tp3"
                elif close_reason == "SL hit" and pos.be_activated:
                    self._trade_stats["be_sl"] += 1
                    result = "be_sl"
                elif close_reason == "SL hit":
                    self._trade_stats["sl"] += 1
                    result = "sl"
                else:
                    result = "other"
                self._trade_stats["total"] += 1
                if pos.closed_pnl > 0:
                    self._trade_stats["wins"] += 1

                # Per-strategy stats
                sn = pos.strategy_name or "Unknown"
                if sn not in self._strategy_stats:
                    self._strategy_stats[sn] = {"tp3": 0, "tp2": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}
                ss = self._strategy_stats[sn]
                ss["total"] += 1
                if result == "tp3":       ss["tp3"]   += 1
                elif result == "be_sl":   ss["be_sl"] += 1
                elif result == "sl":      ss["sl"]    += 1
                if pos.closed_pnl > 0:    ss["wins"]  += 1

                self._session_trades.append({"pnl": pos.closed_pnl, "result": result, "strategy": sn})

                del self._paper_positions[pair]
                open_count = len(self._paper_positions)

                if self._session_count >= 50 and open_count == 0:
                    self._send_session_summary()
                    self._resume_at = datetime.utcnow() + timedelta(hours=5)
                    logger.info("[PAPER-FX] Session complete — 5h pause started")

                logger.info(
                    f"[PAPER-FX] CLOSED {pair} | {close_reason} "
                    f"@ {exit_price:.5f} | PnL={pos.closed_pnl:+.2f} | Balance=${self.paper_balance:.2f} | Open: {open_count}"
                )
                self.notifier.paper_closed(pos, close_reason, exit_price,
                                           pos.closed_pnl, self.paper_balance, tp_level, self._trade_stats)
                self._paper_trades.append({
                    "symbol": pair, "direction": pos.direction,
                    "pnl": pos.closed_pnl, "result": result, "tp_level": tp_level,
                })
                return
            elif act == "notify_tp1":
                self.notifier.paper_tp1_alert(pos, price)

            elif act == "close_partial":
                pct  = action["pct"]
                tp_l = action.get("tp_level", 0)
                size = round(pos.size_remaining * pct, 6)
                partial_price = pos.tp2 if tp_l == 2 else price
                pnl  = self._pnl(pos, partial_price, size)
                pos.size_remaining -= size
                pos.closed_pnl     += pnl
                self.paper_balance += pnl

                if tp_l == 2:
                    sn = pos.strategy_name or "Unknown"
                    if sn not in self._strategy_stats:
                        self._strategy_stats[sn] = {"tp3": 0, "tp2": 0, "sl": 0, "be_sl": 0, "total": 0, "wins": 0}
                    self._strategy_stats[sn]["tp2"] += 1

                logger.info(
                    f"[PAPER-FX] TP{tp_l} {pair} | {pct*100:.0f}% closed "
                    f"@ {partial_price:.5f} | PnL={pnl:+.2f} | Balance={self.paper_balance:.2f}"
                )
                self.notifier.paper_tp_hit(pos, tp_l, partial_price, pnl, self.paper_balance)
            elif act == "move_sl":
                pos.stop_loss = action["new_sl"]
                note = " → Break-Even" if action["new_sl"] == pos.entry_price else f" → {action['new_sl']:.5f}"
                logger.info(f"[PAPER-FX] SL moved{note} for {pair}")

    def _pnl(self, pos: Position, exit_price: float, size: float) -> float:
        if pos.direction == "long":
            return (exit_price - pos.entry_price) * size
        return (pos.entry_price - exit_price) * size

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def _maybe_daily_summary(self):
        now   = datetime.utcnow()
        today = now.date()
        if now.hour != self.daily_sum_hour:
            return
        if self._last_summary_date == today:
            return
        self._last_summary_date = today

        confirmed = [a for a in self._daily_alerts if a["stage"] == 2]
        warnings  = [a for a in self._daily_alerts if a["stage"] == 1]
        longs     = [a for a in confirmed if a["direction"] == "long"]
        shorts    = [a for a in confirmed if a["direction"] == "short"]
        syms      = ", ".join({a["symbol"] for a in confirmed}) or "None"

        paper_section = ""
        if self.paper_enabled and self._paper_trades:
            wins      = [t for t in self._paper_trades if t["result"] == "win"]
            losses    = [t for t in self._paper_trades if t["result"] == "loss"]
            total_pnl = sum(t["pnl"] for t in self._paper_trades)
            win_rate  = len(wins) / len(self._paper_trades) * 100
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            paper_section = (
                f"\n─────────────────────────\n"
                f"📄 <b>Paper Trading</b>\n"
                f"Trades: <code>{len(self._paper_trades)}</code>  "
                f"(W: {len(wins)} / L: {len(losses)})  Win rate: <code>{win_rate:.0f}%</code>\n"
                f"{pnl_emoji} Day PnL: <code>{total_pnl:+.2f} USD</code>\n"
                f"Balance: <code>{self.paper_balance:.2f}</code>  "
                f"(started: {self.paper_start_bal:.2f})"
            )

        self.notifier.send(
            f"📊 <b>Forex Daily Summary — {today}</b>\n"
            f"─────────────────────────\n"
            f"Confirmed signals: <code>{len(confirmed)}</code>  "
            f"(🟢 {len(longs)} Long / 🔴 {len(shorts)} Short)\n"
            f"Warnings:          <code>{len(warnings)}</code>\n"
            f"Pairs triggered:   {syms}"
            f"{paper_section}\n"
            f"─────────────────────────\n"
            f"<i>Next summary {self.daily_sum_hour:02d}:00 UTC</i>"
        )
        self._daily_alerts = []
        self._paper_trades = []

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick(self):
        self._check_session_resume()
        now      = datetime.utcnow().strftime("%H:%M:%S")
        open_pos = len(self._paper_positions)
        bal_info = f" | Paper: ${self.paper_balance:.2f} | Pos: {open_pos}" if self.paper_enabled else ""
        pairs = self.pair_selector.get_pairs()
        logger.info(f">>> FX Scanning {len(pairs)} pairs @ {now} UTC{bal_info}")

        signals_found = 0
        for pair in pairs:
            try:
                # Fetch all timeframes
                htf_raw   = fetch_ohlcv(pair, self.tf_htf,   self.lookback)
                itf_raw   = fetch_ohlcv(pair, self.tf_trend,  self.lookback)
                entry_raw = fetch_ohlcv(pair, self.tf_entry,  self.lookback)

                if htf_raw is None or itf_raw is None or entry_raw is None:
                    continue

                # Enrich with indicators
                htf_df   = self.ema_strategy.enrich(htf_raw.copy())
                itf_df   = self.ema_strategy.enrich(itf_raw.copy())
                entry_df = self.ema_strategy.enrich(entry_raw.copy())
                lb_df    = self.lb_strategy.enrich(entry_raw.copy())

                # Skip pairs without enough history
                if len(htf_df) < MIN_CANDLES or len(itf_df) < MIN_CANDLES or len(entry_df) < MIN_CANDLES:
                    logger.debug(f"Skipping {pair}: insufficient history")
                    continue

                current_price = float(entry_df.iloc[-2]["close"])

                # Paper position management
                if self.paper_enabled and pair in self._paper_positions:
                    self._paper_tick(pair, current_price)

                in_paper = self.paper_enabled and pair in self._paper_positions

                # ── Strategy 1: FX EMA Trend ──────────────────────────────
                if not in_paper:
                    sig = self.ema_strategy.generate_signal(pair, htf_df, itf_df, entry_df)
                    if sig and not self._is_on_cooldown(pair, sig["direction"] + "_ema", sig["stage"]):
                        q = sig.get("quality", 3)
                        stage_label = "CONFIRMED" if sig["stage"] == 2 else "WARNING"
                        logger.info(f"[EMA {stage_label}] {sig['direction'].upper()} {pair} @ {sig['entry']:.5f} | Q={q}")
                        if sig["stage"] == 2 and not self._session_paused and q >= 4:
                            self.notifier.fx_confirmed_signal(sig, "FX EMA Trend", force_forex_channel=True)
                            if self.paper_enabled:
                                self._paper_open(sig, "FX EMA Trend")
                        elif sig["stage"] == 2:
                            logger.info(f"[EMA] Skipped {pair} — quality {q} < 4")
                        elif self.send_warnings:
                            self.notifier.fx_warning_signal(sig, "FX EMA Trend")
                        self._mark_sent(pair, sig["direction"] + "_ema", sig["stage"])
                        self._daily_alerts.append({"stage": sig["stage"], "direction": sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 2: London Breakout ───────────────────────────
                if not in_paper:
                    lb_sig = self.lb_strategy.generate_signal(pair, lb_df)
                    if lb_sig and not self._is_on_cooldown(pair, lb_sig["direction"] + "_lb", lb_sig["stage"]):
                        q = lb_sig.get("quality", 3)
                        stage_label = "CONFIRMED" if lb_sig["stage"] == 2 else "WARNING"
                        logger.info(f"[LB {stage_label}] {lb_sig['direction'].upper()} {pair} @ {lb_sig['entry']:.5f} | Q={q}")
                        if lb_sig["stage"] == 2 and not self._session_paused and q >= 4:
                            self.notifier.lb_confirmed_signal(lb_sig, force_forex_channel=True)
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(lb_sig, "London Breakout")
                        elif lb_sig["stage"] == 2:
                            logger.info(f"[LB] Skipped {pair} — quality {q} < 4")
                        elif self.send_warnings:
                            self.notifier.fx_warning_signal(lb_sig, "London Breakout")
                        self._mark_sent(pair, lb_sig["direction"] + "_lb", lb_sig["stage"])
                        self._daily_alerts.append({"stage": lb_sig["stage"], "direction": lb_sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 3: Fibonacci Pullback ───────────────────────
                if not in_paper:
                    fib_sig = self.fib_strategy.generate_signal(pair, htf_df, entry_df)
                    if fib_sig and not self._is_on_cooldown(pair, fib_sig["direction"] + "_fib", fib_sig["stage"]):
                        q = fib_sig.get("quality", 3)
                        logger.info(f"[FIB CONFIRMED] {fib_sig['direction'].upper()} {pair} @ {fib_sig['entry']:.5f} | Q={q}")
                        if fib_sig["stage"] == 2 and not self._session_paused and q >= 5:
                            self.notifier.fx_confirmed_signal(fib_sig, "Fib Pullback", force_forex_channel=True)
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(fib_sig, "Fib Pullback")
                        else:
                            logger.info(f"[FIB] Skipped {pair} — quality {q} < 5")
                        self._mark_sent(pair, fib_sig["direction"] + "_fib", fib_sig["stage"])
                        self._daily_alerts.append({"stage": fib_sig["stage"], "direction": fib_sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 4: Daily Level Break + Retest ───────────────
                if not in_paper:
                    dl_sig = self.dl_strategy.generate_signal(pair, htf_df, entry_df)
                    if dl_sig and not self._is_on_cooldown(pair, dl_sig["direction"] + "_dl", dl_sig["stage"]):
                        q = dl_sig.get("quality", 3)
                        logger.info(f"[DL CONFIRMED] {dl_sig['direction'].upper()} {pair} @ {dl_sig['entry']:.5f} | Q={q}")
                        if dl_sig["stage"] == 2 and not self._session_paused and q >= 5:
                            self.notifier.fx_confirmed_signal(dl_sig, "Daily Level Break", force_forex_channel=True)
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(dl_sig, "Daily Level Break")
                        else:
                            logger.info(f"[DL] Skipped {pair} — quality {q} < 5")
                        self._mark_sent(pair, dl_sig["direction"] + "_dl", dl_sig["stage"])
                        self._daily_alerts.append({"stage": dl_sig["stage"], "direction": dl_sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 5: BB Mean Reversion ────────────────────────
                if not in_paper:
                    bbmr_sig = self.bbmr_strategy.generate_signal(pair, htf_df, entry_df)
                    if bbmr_sig and not self._is_on_cooldown(pair, bbmr_sig["direction"] + "_bbmr", bbmr_sig["stage"]):
                        q = bbmr_sig.get("quality", 3)
                        logger.info(f"[BBMR CONFIRMED] {bbmr_sig['direction'].upper()} {pair} @ {bbmr_sig['entry']:.5f} | Q={q}")
                        if bbmr_sig["stage"] == 2 and not self._session_paused and q >= 5:
                            self.notifier.fx_confirmed_signal(bbmr_sig, "BB Mean Reversion", force_forex_channel=True)
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(bbmr_sig, "BB Mean Reversion")
                        else:
                            logger.info(f"[BBMR] Skipped {pair} — quality {q} < 5")
                        self._mark_sent(pair, bbmr_sig["direction"] + "_bbmr", bbmr_sig["stage"])
                        self._daily_alerts.append({"stage": bbmr_sig["stage"], "direction": bbmr_sig["direction"], "symbol": pair})
                        signals_found += 1

            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
                logger.debug(traceback.format_exc())

        if signals_found == 0:
            logger.info("No signals this tick")
        self._maybe_daily_summary()
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
        self.notifier.paper_positions_update(self._paper_positions, self.paper_balance, self.paper_start_bal)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        pairs = self.pair_selector.get_pairs()
        logger.info(
            f"Forex Scanner started | {len(pairs)} pairs | "
            f"FX EMA Trend + London Breakout + Fib Pullback + Daily Level Break + BB Mean Reversion"
        )
        self.notifier.scanner_started(
            self.pair_selector.get_pairs(),
            tf_trend=self.tf_trend,
            tf_entry=self.tf_entry,
            cooldown_min=self.cooldown_min,
            paper_enabled=self.paper_enabled,
            paper_balance=self.paper_balance,
            strategies=["FX EMA Trend", "London Breakout", "Fib Pullback", "Daily Level Break", "BB Mean Reversion"],
            label="Forex Scanner",
        )
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Tick error: {e}")
                logger.debug(traceback.format_exc())
                self.notifier.error_alert("ForexBot._tick", str(e))
            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False
