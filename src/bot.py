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
from src.strategies.sr_bounce import SRBounceStrategy
from src.strategies.order_block import OrderBlockStrategy
from src.strategies.trendline_break import TrendlineBreakStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.rsi_macd_reversal import RSIMACDReversalStrategy
from src.pair_selector import PairSelector
from src.notifier import Notifier
from src.bybit_executor import BybitExecutor

logger = logging.getLogger("futures_bot")


def ohlcv_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


class Bot:
    def __init__(self, cfg: dict, env: dict):
        # Mode switch: "scalp" swaps signal + filter params before strategies load
        self.mode = cfg.get("mode", "swing")
        self.send_warnings = (self.mode != "scalp")   # scalp = confirmed only
        if self.mode == "scalp":
            if "scalp_signal" in cfg:
                cfg = {**cfg, "signal": cfg["scalp_signal"]}
            if "scalp_filters" in cfg:
                cfg = {**cfg, "filters": cfg["scalp_filters"]}
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
        self.paper_risk_pct: float = paper_cfg.get("risk_per_trade_pct", 1.0) / 100.0
        self.paper_start_balance: float = self.paper_balance

        self.tf_sr: str = cfg.get("timeframe_sr", "4h")

        self.strategy    = Strategy(cfg)
        self.sr_strategy = SRBounceStrategy(cfg)
        self.ob_strategy = OrderBlockStrategy(cfg)
        self.tl_strategy = TrendlineBreakStrategy(cfg)
        self.rd_strategy = RSIDivergenceStrategy(cfg)
        self.rm_strategy = RSIMACDReversalStrategy(cfg)
        self.notifier    = Notifier(channel_name=cfg.get("channel_name", ""))
        self.exchange    = self._init_exchange(cfg, env)
        self.bybit       = BybitExecutor(risk_pct=self.paper_risk_pct)
        self.pair_selector = PairSelector(self.exchange, cfg)

        # Signal cooldown: (symbol, direction, stage) -> last alert time
        self._last_alert: dict[tuple, datetime] = {}

        # Paper positions: symbol -> Position
        self._paper_positions: dict[str, Position] = {}
        self._max_paper_positions = 10
        self._paper_paused = False

        # Lifetime trade stats
        self._trade_stats = {"sl": 0, "tp3": 0, "be_sl": 0, "total": 0}

        # Stats
        self._daily_alerts: list[dict] = []
        self._paper_trades: list[dict] = []   # closed paper trades for daily summary
        self._last_summary_date: Optional[date] = None
        self._last_positions_report: datetime = datetime.utcnow()
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

        # Cap at 10 open positions; resume only when count drops to ≤1
        if len(self._paper_positions) >= self._max_paper_positions and not self._paper_paused:
            self._paper_paused = True
            self.notifier.send(
                f"⏸️ <b>Paper Trading Paused</b>\n"
                f"10 positions open — waiting until ≤1 remains before new entries."
            )
        if self._paper_paused:
            logger.debug(f"[PAPER] Paused — {len(self._paper_positions)} positions open")
            return

        sl_dist = abs(signal.entry_price - signal.stop_loss)
        if sl_dist == 0:
            return

        risk_amount = self.paper_balance * self.paper_risk_pct
        size = round(risk_amount / sl_dist, 6)
        if size <= 0:
            return

        self.paper_balance -= risk_amount   # lock margin immediately

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
            margin_locked=risk_amount,
        )
        self._paper_positions[signal.symbol] = pos

        sl_pct = sl_dist / signal.entry_price * 100
        open_count = len(self._paper_positions)
        logger.info(
            f"[PAPER] OPENED {signal.direction.upper()} {signal.symbol} "
            f"@ {signal.entry_price:.4f} | Risk=${risk_amount:.2f} | Size={size:.4f} | "
            f"SL={signal.stop_loss:.4f} (-{sl_pct:.2f}%) | Available=${self.paper_balance:.2f}"
        )
        self.notifier.paper_opened(pos, self.paper_balance, open_count)

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
                self.paper_balance += pos.margin_locked + pnl
                tp_level = action.get("tp_level", 0)

                # Classify result
                if tp_level == 3:
                    self._trade_stats["tp3"] += 1
                    result = "tp3"
                elif reason == "SL hit" and pos.be_activated:
                    self._trade_stats["be_sl"] += 1
                    result = "be_sl"
                elif reason == "SL hit":
                    self._trade_stats["sl"] += 1
                    result = "sl"
                else:
                    result = "other"
                self._trade_stats["total"] += 1

                del self._paper_positions[symbol]
                open_count = len(self._paper_positions)

                # Resume if paused and slots are available again
                if self._paper_paused and open_count <= 1:
                    self._paper_paused = False
                    self.notifier.send(
                        f"▶️ <b>Paper Trading Resumed</b>\n"
                        f"Open positions back to {open_count} — accepting new entries."
                    )

                logger.info(
                    f"[PAPER] CLOSED {symbol} | {reason} "
                    f"@ {current_price:.4f} | PnL={pos.closed_pnl:+.2f} | "
                    f"Balance=${self.paper_balance:.2f} | Open: {open_count}"
                )
                self.notifier.paper_closed(pos, reason, current_price, pos.closed_pnl,
                                           self.paper_balance, tp_level, self._trade_stats)
                self._paper_trades.append({
                    "symbol": symbol, "direction": pos.direction,
                    "pnl": pos.closed_pnl, "result": result, "tp_level": tp_level,
                })
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

    def _bybit_order(self, sig):
        """Place order on Bybit testnet. Accepts Signal dataclass or dict."""
        if not self.bybit.enabled or self._paper_paused:
            return
        if hasattr(sig, "entry_price"):   # Signal dataclass
            d = {"symbol": sig.symbol, "direction": sig.direction,
                 "entry": sig.entry_price, "sl": sig.stop_loss, "tp3": sig.tp3, "atr": sig.atr}
        else:
            d = sig
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

    def _tick(self):
        symbols   = self.pair_selector.get_symbols()
        now       = datetime.utcnow().strftime("%H:%M:%S")
        open_pos  = len(self._paper_positions)
        paper_bal = f"${self.paper_balance:.2f}" if self.paper_enabled else ""
        paper_info = f" | Paper: {paper_bal} | Positions: {open_pos}" if self.paper_enabled else ""
        logger.info(f">>> Scanning {len(symbols)} pairs @ {now} UTC{paper_info}")

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

                current_price = float(entry_df.iloc[-2]["close"])

                # Paper position management (always runs if position is open)
                if self.paper_enabled and symbol in self._paper_positions:
                    self._paper_tick(symbol, current_price)

                in_paper = self.paper_enabled and symbol in self._paper_positions

                # ── Strategy 1: EMA Momentum ──────────────────────────────
                if not in_paper:
                    signal = self.strategy.generate_signal(symbol, htf_df, entry_df)

                    if signal.stage > 0 and not self._is_on_cooldown(symbol, signal.direction + "_ema", signal.stage):
                        stage_label = "CONFIRMED" if signal.stage == 2 else "WARNING"
                        quality = 3  # default quality for EMA strategy
                        logger.info(
                            f"[EMA {stage_label}] {signal.direction.upper()} {symbol} "
                            f"@ {signal.entry_price:.4f} | RSI={signal.rsi:.1f} | {signal.reason}"
                        )
                        if signal.stage == 2 and not self._paper_paused:
                            self.notifier.confirmed_signal(signal, "EMA Momentum", quality)
                            if self.paper_enabled:
                                self._paper_open(signal)
                            self._bybit_order(signal)
                        elif self.send_warnings:
                            self.notifier.warning_signal(signal, "EMA Momentum")

                        self._mark_sent(symbol, signal.direction + "_ema", signal.stage)
                        self._daily_alerts.append({"stage": signal.stage, "direction": signal.direction, "symbol": symbol})
                        signals_found += 1

                # ── Strategy 2: S/R Bounce ────────────────────────────────
                if sr_df is not None and not in_paper:
                    sr_sig = self.sr_strategy.generate_signal(symbol, sr_df, entry_df)

                    if sr_sig and not self._is_on_cooldown(symbol, sr_sig["direction"] + "_sr", sr_sig["stage"]):
                        stage_label = "CONFIRMED" if sr_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[SR {stage_label}] {sr_sig['direction'].upper()} {symbol} "
                            f"@ {sr_sig['entry']:.4f} | Lvl={sr_sig['level_price']:.4f} "
                            f"({sr_sig['level_touches']} touches) | {sr_sig['reason']}"
                        )
                        if sr_sig["stage"] == 2 and not self._paper_paused:
                            self.notifier.sr_confirmed_signal(sr_sig)
                            self._bybit_order(sr_sig)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                # Build a Signal-compatible object for paper trading
                                from src.strategy import Signal as Sig
                                dummy = Sig(
                                    stage=2, direction=sr_sig["direction"], symbol=symbol,
                                    entry_price=sr_sig["entry"], stop_loss=sr_sig["sl"],
                                    tp1=sr_sig["tp1"], tp2=sr_sig["tp2"], tp3=sr_sig["tp3"],
                                    atr=sr_sig["atr"], rsi=sr_sig["rsi"],
                                    volume_ratio=sr_sig.get("vol_ratio", 0),
                                    reason=sr_sig["reason"],
                                )
                                self._paper_open(dummy)
                        elif self.send_warnings:
                            self.notifier.sr_warning_signal(sr_sig)

                        self._mark_sent(symbol, sr_sig["direction"] + "_sr", sr_sig["stage"])
                        self._daily_alerts.append({"stage": sr_sig["stage"], "direction": sr_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 3: Order Block ───────────────────────────────
                if sr_df is not None and not in_paper:
                    ob_sig = self.ob_strategy.generate_signal(symbol, sr_df, entry_df)
                    if ob_sig and not self._is_on_cooldown(symbol, ob_sig["direction"] + "_ob", ob_sig["stage"]):
                        stage_label = "CONFIRMED" if ob_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[OB {stage_label}] {ob_sig['direction'].upper()} {symbol} "
                            f"@ {ob_sig['entry']:.4f} | {ob_sig['reason']}"
                        )
                        if ob_sig["stage"] == 2 and not self._paper_paused:
                            self.notifier.fx_confirmed_signal(ob_sig, "Order Block")
                            self._bybit_order(ob_sig)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=ob_sig["direction"], symbol=symbol,
                                            entry_price=ob_sig["entry"], stop_loss=ob_sig["sl"],
                                            tp1=ob_sig["tp1"], tp2=ob_sig["tp2"], tp3=ob_sig["tp3"],
                                            atr=ob_sig["atr"], rsi=ob_sig["rsi"], volume_ratio=0,
                                            reason=ob_sig["reason"])
                                self._paper_open(dummy)
                        elif self.send_warnings:
                            self.notifier.fx_warning_signal(ob_sig, "Order Block")
                        self._mark_sent(symbol, ob_sig["direction"] + "_ob", ob_sig["stage"])
                        self._daily_alerts.append({"stage": ob_sig["stage"], "direction": ob_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 4: Trendline ─────────────────────────────────
                if not in_paper:
                    tl_sig = self.tl_strategy.generate_signal(symbol, htf_df, entry_df)
                    if tl_sig and not self._is_on_cooldown(symbol, tl_sig["direction"] + "_tl", tl_sig["stage"]):
                        stage_label = "CONFIRMED" if tl_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[TL {stage_label}] {tl_sig['direction'].upper()} {symbol} "
                            f"@ {tl_sig['entry']:.4f} | {tl_sig['reason']}"
                        )
                        if tl_sig["stage"] == 2 and not self._paper_paused:
                            self.notifier.fx_confirmed_signal(tl_sig, "Trendline")
                            self._bybit_order(tl_sig)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=tl_sig["direction"], symbol=symbol,
                                            entry_price=tl_sig["entry"], stop_loss=tl_sig["sl"],
                                            tp1=tl_sig["tp1"], tp2=tl_sig["tp2"], tp3=tl_sig["tp3"],
                                            atr=tl_sig["atr"], rsi=tl_sig["rsi"], volume_ratio=0,
                                            reason=tl_sig["reason"])
                                self._paper_open(dummy)
                        elif self.send_warnings:
                            self.notifier.fx_warning_signal(tl_sig, "Trendline")
                        self._mark_sent(symbol, tl_sig["direction"] + "_tl", tl_sig["stage"])
                        self._daily_alerts.append({"stage": tl_sig["stage"], "direction": tl_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 5: RSI Divergence ────────────────────────────
                if not in_paper:
                    rd_sig = self.rd_strategy.generate_signal(symbol, htf_df, entry_df)
                    if rd_sig and not self._is_on_cooldown(symbol, rd_sig["direction"] + "_rd", rd_sig["stage"]):
                        stage_label = "CONFIRMED" if rd_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[DIV {stage_label}] {rd_sig['direction'].upper()} {symbol} "
                            f"@ {rd_sig['entry']:.4f} | {rd_sig['reason']}"
                        )
                        if rd_sig["stage"] == 2 and not self._paper_paused:
                            self.notifier.fx_confirmed_signal(rd_sig, "RSI Divergence")
                            self._bybit_order(rd_sig)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=rd_sig["direction"], symbol=symbol,
                                            entry_price=rd_sig["entry"], stop_loss=rd_sig["sl"],
                                            tp1=rd_sig["tp1"], tp2=rd_sig["tp2"], tp3=rd_sig["tp3"],
                                            atr=rd_sig["atr"], rsi=rd_sig["rsi"], volume_ratio=0,
                                            reason=rd_sig["reason"])
                                self._paper_open(dummy)
                        elif self.send_warnings:
                            self.notifier.fx_warning_signal(rd_sig, "RSI Divergence")
                        self._mark_sent(symbol, rd_sig["direction"] + "_rd", rd_sig["stage"])
                        self._daily_alerts.append({"stage": rd_sig["stage"], "direction": rd_sig["direction"], "symbol": symbol})
                        signals_found += 1

                # ── Strategy 6: RSI+MACD Reversal ────────────────────────
                if not in_paper:
                    rm_sig = self.rm_strategy.generate_signal(symbol, entry_df)
                    if rm_sig and not self._is_on_cooldown(symbol, rm_sig["direction"] + "_rm", rm_sig["stage"]):
                        stage_label = "CONFIRMED" if rm_sig["stage"] == 2 else "WARNING"
                        logger.info(
                            f"[RM {stage_label}] {rm_sig['direction'].upper()} {symbol} "
                            f"@ {rm_sig['entry']:.4f} | RSI={rm_sig['rsi']:.1f} | {rm_sig['reason']}"
                        )
                        if rm_sig["stage"] == 2 and not self._paper_paused:
                            self.notifier.fx_confirmed_signal(rm_sig, "RSI+MACD Reversal")
                            self._bybit_order(rm_sig)
                            if self.paper_enabled and symbol not in self._paper_positions:
                                from src.strategy import Signal as Sig
                                dummy = Sig(stage=2, direction=rm_sig["direction"], symbol=symbol,
                                            entry_price=rm_sig["entry"], stop_loss=rm_sig["sl"],
                                            tp1=rm_sig["tp1"], tp2=rm_sig["tp2"], tp3=rm_sig["tp3"],
                                            atr=rm_sig["atr"], rsi=rm_sig["rsi"], volume_ratio=0,
                                            reason=rm_sig["reason"])
                                self._paper_open(dummy)
                        elif self.send_warnings:
                            self.notifier.fx_warning_signal(rm_sig, "RSI+MACD Reversal")
                        self._mark_sent(symbol, rm_sig["direction"] + "_rm", rm_sig["stage"])
                        self._daily_alerts.append({"stage": rm_sig["stage"], "direction": rm_sig["direction"], "symbol": symbol})
                        signals_found += 1

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}\n{traceback.format_exc()}")
                self.notifier.error_alert(f"Scanning {symbol}", str(e)[:200])

        signal_note = f" | {signals_found} signal(s) fired" if signals_found > 0 else " | No signals"
        logger.info(f"<<< Scan complete{signal_note} | Next scan in {self.poll_interval}s")
        logger.info(f"    Pairs: {' | '.join(s.split('/')[0] for s in symbols)}")

        self._maybe_send_daily_summary()
        self._maybe_send_positions_report()

    def _maybe_send_positions_report(self):
        """Send open positions summary to Telegram every 60 minutes."""
        if not self.paper_enabled:
            return
        if (datetime.utcnow() - self._last_positions_report).total_seconds() < 3600:
            return
        self._last_positions_report = datetime.utcnow()
        self.notifier.paper_positions_update(self._paper_positions, self.paper_balance, self.paper_start_balance)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        paper_note = (
            f" | Paper: ON (balance={self.paper_balance:.0f} USDT, risk={self.paper_risk_pct*100:.1f}%)"
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

        bybit_note = f" | Bybit Testnet: {'ON' if self.bybit.enabled else 'OFF'}"
        self.notifier.scanner_started(
            symbols, self.tf_trend, self.tf_entry,
            self.cooldown_min, self.paper_enabled, self.paper_balance,
            strategies=["EMA Trend", "S/R Bounce", "Order Block", "Trendline", "RSI Divergence", "RSI+MACD Reversal"],
            label=f"Crypto Futures Scanner{bybit_note}",
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
