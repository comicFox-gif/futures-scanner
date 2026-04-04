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
from src.strategies.order_block import OrderBlockStrategy
from src.strategies.trendline_break import TrendlineBreakStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.rsi_macd_reversal import RSIMACDReversalStrategy
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
            actions.append({"action": "close_partial", "pct": tp2_pct, "tp_level": 2})
            actions.append({"action": "move_sl", "new_sl": pos.tp1})
            pos.tp2_hit = True
        elif not pos.tp1_hit and current_price >= pos.tp1:
            actions.append({"action": "close_partial", "pct": tp1_pct, "tp_level": 1})
            actions.append({"action": "move_sl", "new_sl": pos.entry_price})
            pos.tp1_hit = True
            pos.be_activated = True
    else:
        if current_price >= pos.stop_loss:
            actions.append({"action": "close_all", "reason": "SL hit"})
            return actions
        if not pos.tp3_hit and current_price <= pos.tp3:
            actions.append({"action": "close_all", "reason": "TP3 hit", "tp_level": 3})
            pos.tp3_hit = True
            return actions
        if not pos.tp2_hit and current_price <= pos.tp2:
            actions.append({"action": "close_partial", "pct": tp2_pct, "tp_level": 2})
            actions.append({"action": "move_sl", "new_sl": pos.tp1})
            pos.tp2_hit = True
        elif not pos.tp1_hit and current_price <= pos.tp1:
            actions.append({"action": "close_partial", "pct": tp1_pct, "tp_level": 1})
            actions.append({"action": "move_sl", "new_sl": pos.entry_price})
            pos.tp1_hit = True
            pos.be_activated = True

    return actions


class ForexBot:
    def __init__(self, cfg: dict):
        # Mode switch: "scalp" swaps signal + filter params before strategies load
        self.mode = cfg.get("mode", "swing")
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
        self.paper_risk_pct    = paper_cfg.get("risk_per_trade_pct", 1.0) / 100.0
        self.paper_start_bal   = self.paper_balance

        self.ema_strategy = ForexEmaTrendStrategy(cfg)
        self.lb_strategy  = LondonBreakoutStrategy(cfg)
        self.ob_strategy  = OrderBlockStrategy(cfg)
        self.tl_strategy  = TrendlineBreakStrategy(cfg)
        self.rd_strategy  = RSIDivergenceStrategy(cfg)
        self.rm_strategy  = RSIMACDReversalStrategy(cfg)
        self.notifier     = Notifier(channel_name=cfg.get("channel_name", ""))

        self._last_alert: dict[tuple, datetime] = {}
        self._paper_positions: dict[str, Position] = {}
        self._daily_alerts: list[dict] = []
        self._paper_trades: list[dict] = []
        self._last_summary_date: Optional[date] = None
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

    def _paper_open(self, sig: dict):
        pair = sig["symbol"]
        if pair in self._paper_positions:
            return
        sl_dist = abs(sig["entry"] - sig["sl"])
        if sl_dist == 0:
            return
        size = round(self.paper_balance * self.paper_risk_pct / sl_dist, 6)
        if size <= 0:
            return
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
        )
        self._paper_positions[pair] = pos
        sl_pct = sl_dist / sig["entry"] * 100
        logger.info(
            f"[PAPER-FX] OPENED {sig['direction'].upper()} {pair} "
            f"@ {sig['entry']:.5f} | Size={size:.6f} | "
            f"SL={sig['sl']:.5f} (-{sl_pct:.2f}%) | "
            f"TP3={sig['tp3']:.5f}"
        )
        self.notifier.paper_opened(pos, self.paper_balance)

    def _paper_tick(self, pair: str, price: float):
        pos = self._paper_positions.get(pair)
        if not pos:
            return
        actions = _check_position(pos, price)
        for action in actions:
            act = action["action"]
            if act == "close_all":
                pnl = self._pnl(pos, price, pos.size_remaining)
                pos.closed_pnl += pnl
                self.paper_balance += pnl
                tp_level = action.get("tp_level", 0)
                logger.info(
                    f"[PAPER-FX] CLOSED {pair} | {action.get('reason', '')} "
                    f"@ {price:.5f} | PnL={pnl:+.2f} | Balance={self.paper_balance:.2f}"
                )
                self.notifier.paper_closed(pos, action.get("reason", ""), price,
                                           pos.closed_pnl, self.paper_balance, tp_level)
                self._paper_trades.append({
                    "symbol": pair, "direction": pos.direction,
                    "pnl": pos.closed_pnl,
                    "result": "win" if pos.closed_pnl > 0 else "loss",
                    "tp_level": tp_level,
                })
                del self._paper_positions[pair]
                return
            elif act == "close_partial":
                pct  = action["pct"]
                tp_l = action.get("tp_level", 0)
                size = round(pos.size_remaining * pct, 6)
                pnl  = self._pnl(pos, price, size)
                pos.size_remaining -= size
                pos.closed_pnl     += pnl
                self.paper_balance += pnl
                logger.info(
                    f"[PAPER-FX] TP{tp_l} {pair} | {pct*100:.0f}% closed "
                    f"@ {price:.5f} | PnL={pnl:+.2f} | Balance={self.paper_balance:.2f}"
                )
                self.notifier.paper_tp_hit(pos, tp_l, price, pnl, self.paper_balance)
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
                        stage_label = "CONFIRMED" if sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[EMA {stage_label}] {sig['direction'].upper()} {pair} "
                            f"@ {sig['entry']:.5f} | RSI={sig['rsi']:.1f} | {sig['reason']}"
                        )
                        if sig["stage"] == 2:
                            self.notifier.fx_confirmed_signal(sig, "FX EMA Trend")
                            if self.paper_enabled:
                                self._paper_open(sig)
                        else:
                            self.notifier.fx_warning_signal(sig, "FX EMA Trend")
                        self._mark_sent(pair, sig["direction"] + "_ema", sig["stage"])
                        self._daily_alerts.append({"stage": sig["stage"], "direction": sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 2: London Breakout ───────────────────────────
                if not in_paper:
                    lb_sig = self.lb_strategy.generate_signal(pair, lb_df)
                    if lb_sig and not self._is_on_cooldown(pair, lb_sig["direction"] + "_lb", lb_sig["stage"]):
                        stage_label = "CONFIRMED" if lb_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[LB {stage_label}] {lb_sig['direction'].upper()} {pair} "
                            f"@ {lb_sig['entry']:.5f} | Range={lb_sig.get('range_pips', 0):.0f} pips | "
                            f"{lb_sig['reason']}"
                        )
                        if lb_sig["stage"] == 2:
                            self.notifier.lb_confirmed_signal(lb_sig)
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(lb_sig)
                        else:
                            self.notifier.fx_warning_signal(lb_sig, "London Breakout")
                        self._mark_sent(pair, lb_sig["direction"] + "_lb", lb_sig["stage"])
                        self._daily_alerts.append({"stage": lb_sig["stage"], "direction": lb_sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 3: Order Block ───────────────────────────────
                if not in_paper:
                    ob_sig = self.ob_strategy.generate_signal(pair, htf_df, entry_df)
                    if ob_sig and not self._is_on_cooldown(pair, ob_sig["direction"] + "_ob", ob_sig["stage"]):
                        stage_label = "CONFIRMED" if ob_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[OB {stage_label}] {ob_sig['direction'].upper()} {pair} "
                            f"@ {ob_sig['entry']:.5f} | {ob_sig['reason']}"
                        )
                        if ob_sig["stage"] == 2:
                            self.notifier.fx_confirmed_signal(ob_sig, "Order Block")
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(ob_sig)
                        else:
                            self.notifier.fx_warning_signal(ob_sig, "Order Block")
                        self._mark_sent(pair, ob_sig["direction"] + "_ob", ob_sig["stage"])
                        self._daily_alerts.append({"stage": ob_sig["stage"], "direction": ob_sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 4: Trendline ─────────────────────────────────
                if not in_paper:
                    tl_sig = self.tl_strategy.generate_signal(pair, itf_df, entry_df)
                    if tl_sig and not self._is_on_cooldown(pair, tl_sig["direction"] + "_tl", tl_sig["stage"]):
                        stage_label = "CONFIRMED" if tl_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[TL {stage_label}] {tl_sig['direction'].upper()} {pair} "
                            f"@ {tl_sig['entry']:.5f} | {tl_sig['reason']}"
                        )
                        if tl_sig["stage"] == 2:
                            self.notifier.fx_confirmed_signal(tl_sig, "Trendline")
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(tl_sig)
                        else:
                            self.notifier.fx_warning_signal(tl_sig, "Trendline")
                        self._mark_sent(pair, tl_sig["direction"] + "_tl", tl_sig["stage"])
                        self._daily_alerts.append({"stage": tl_sig["stage"], "direction": tl_sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 5: RSI Divergence ────────────────────────────
                if not in_paper:
                    rd_sig = self.rd_strategy.generate_signal(pair, itf_df, entry_df)
                    if rd_sig and not self._is_on_cooldown(pair, rd_sig["direction"] + "_rd", rd_sig["stage"]):
                        stage_label = "CONFIRMED" if rd_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[DIV {stage_label}] {rd_sig['direction'].upper()} {pair} "
                            f"@ {rd_sig['entry']:.5f} | {rd_sig['reason']}"
                        )
                        if rd_sig["stage"] == 2:
                            self.notifier.fx_confirmed_signal(rd_sig, "RSI Divergence")
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(rd_sig)
                        else:
                            self.notifier.fx_warning_signal(rd_sig, "RSI Divergence")
                        self._mark_sent(pair, rd_sig["direction"] + "_rd", rd_sig["stage"])
                        self._daily_alerts.append({"stage": rd_sig["stage"], "direction": rd_sig["direction"], "symbol": pair})
                        signals_found += 1

                # ── Strategy 6: RSI+MACD Reversal ────────────────────────
                if not in_paper:
                    rm_sig = self.rm_strategy.generate_signal(pair, entry_df)
                    if rm_sig and not self._is_on_cooldown(pair, rm_sig["direction"] + "_rm", rm_sig["stage"]):
                        stage_label = "CONFIRMED" if rm_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[RM {stage_label}] {rm_sig['direction'].upper()} {pair} "
                            f"@ {rm_sig['entry']:.5f} | RSI={rm_sig['rsi']:.1f} | {rm_sig['reason']}"
                        )
                        if rm_sig["stage"] == 2:
                            self.notifier.fx_confirmed_signal(rm_sig, "RSI+MACD Reversal")
                            if self.paper_enabled and pair not in self._paper_positions:
                                self._paper_open(rm_sig)
                        else:
                            self.notifier.fx_warning_signal(rm_sig, "RSI+MACD Reversal")
                        self._mark_sent(pair, rm_sig["direction"] + "_rm", rm_sig["stage"])
                        self._daily_alerts.append({"stage": rm_sig["stage"], "direction": rm_sig["direction"], "symbol": pair})
                        signals_found += 1

            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
                logger.debug(traceback.format_exc())

        if signals_found == 0:
            logger.info("No signals this tick")
        self._maybe_daily_summary()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        pairs = self.pair_selector.get_pairs()
        logger.info(
            f"Forex Scanner started | {len(pairs)} pairs | "
            f"EMA Trend + London Breakout + Order Block + Trendline + RSI Divergence"
        )
        self.notifier.scanner_started(
            self.pair_selector.get_pairs(),
            tf_trend=self.tf_trend,
            tf_entry=self.tf_entry,
            cooldown_min=self.cooldown_min,
            paper_enabled=self.paper_enabled,
            paper_balance=self.paper_balance,
            strategies=["FX EMA Trend", "London Breakout", "Order Block", "Trendline", "RSI Divergence", "RSI+MACD Reversal"],
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
