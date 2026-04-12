"""
Telegram Notifier
------------------
Clean signal format for paid channel distribution.
"""

import logging
import os
import requests
from datetime import datetime

logger = logging.getLogger("futures_bot.notifier")

_SUB_BOT_URL = os.getenv("SUB_BOT_URL", "").rstrip("/")
_SUB_BOT_KEY = os.getenv("SUB_BOT_API_KEY", "")

LINE  = "━" * 30
DLINE = "─" * 30


class Notifier:
    def __init__(self, channel_name: str = "", forex_symbols: set = None):
        self.token    = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id  = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled  = bool(self.token and self.chat_id)

        self.forex_token   = os.getenv("FOREX_BOT_TOKEN", "")
        self.forex_chat_id = os.getenv("FOREX_CHAT_ID", "")
        self.forex_enabled = bool(self.forex_token and self.forex_chat_id)

        self.channel       = channel_name
        self._signal_no    = 0
        self._forex_symbols: set = forex_symbols or set()

        if self.enabled:
            logger.info("Telegram main bot enabled")
        else:
            logger.info("Telegram main bot not configured — notifications disabled")

        if self.forex_enabled:
            logger.info("Telegram forex signals bot enabled")

    # ------------------------------------------------------------------
    # Core senders
    # ------------------------------------------------------------------

    def _forward_to_subbot(self, message: str):
        """Forward a formatted HTML message to the subscription bot for fan-out."""
        if not _SUB_BOT_URL:
            return
        try:
            headers = {}
            if _SUB_BOT_KEY:
                headers["X-API-Key"] = _SUB_BOT_KEY
            requests.post(
                f"{_SUB_BOT_URL}/broadcast",
                json={"message": message},
                headers=headers,
                timeout=5,
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"SubBot forward failed: {e}")

    def _post(self, token: str, chat_id: str, message: str):
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id":    chat_id,
                    "text":       message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=8,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Telegram send failed: {e}")

    def send(self, message: str):
        """Send to main channel only (errors, paper PnL, system alerts)."""
        if not self.enabled:
            return
        self._post(self.token, self.chat_id, message)

    def send_forex(self, message: str):
        """Send to forex channel only."""
        if not self.forex_enabled:
            return
        self._post(self.forex_token, self.forex_chat_id, message)

    def _is_forex_symbol(self, symbol: str) -> bool:
        return symbol in self._forex_symbols

    def send_signal(self, message: str, forex_message: str = "", is_forex: bool = False):
        if is_forex:
            if self.forex_enabled and forex_message:
                self._post(self.forex_token, self.forex_chat_id, forex_message)
            self._forward_to_subbot(forex_message or message)
        else:
            if self.enabled:
                self._post(self.token, self.chat_id, message)
            self._forward_to_subbot(message)

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(price: float) -> str:
        """Smart crypto price formatter."""
        if price >= 1000:
            return f"{price:,.2f}"
        if price >= 1:
            return f"{price:.4f}"
        return f"{price:.6f}"

    @staticmethod
    def _fmt_fx(price: float) -> str:
        return f"{price:.5f}"

    @staticmethod
    def _qty_for_risk(entry: float, sl: float, risk_usd: float = 5.0) -> float:
        """Units of base currency needed so that SL hit = exactly risk_usd loss."""
        sl_dist = abs(entry - sl)
        return risk_usd / sl_dist if sl_dist > 0 else 0.0

    @staticmethod
    def _fmt_qty(qty: float) -> str:
        if qty >= 100:  return f"{qty:.1f}"
        if qty >= 10:   return f"{qty:.2f}"
        if qty >= 1:    return f"{qty:.3f}"
        return f"{qty:.4f}"

    @staticmethod
    def _base_currency(symbol: str) -> str:
        """Extract base from 'BTC/USDT:USDT' → 'BTC', 'EUR/USD' → 'EUR'."""
        return symbol.split("/")[0]

    @staticmethod
    def _pct(entry: float, level: float, direction: str) -> float:
        if direction == "long":
            return (level - entry) / entry * 100
        return (entry - level) / entry * 100

    def _stars(self, quality: int) -> str:
        return "⭐" * quality + "☆" * (5 - quality)

    def _dir_tag(self, direction: str) -> str:
        return "🟢 LONG" if direction == "long" else "🔴 SHORT"

    def _ts(self) -> str:
        return datetime.utcnow().strftime("%H:%M UTC")

    def _footer(self) -> str:
        if self.channel:
            return f"⚡ <b>{self.channel}</b>  |  {self._ts()}"
        return f"<i>{self._ts()}</i>"

    def _signal_block(self, no: int, strategy: str, quality: int,
                      direction: str, symbol: str,
                      price: float, sl: float, tp1: float, tp2: float, tp3: float,
                      reason: str, extra: str = "", fmt=None) -> str:
        """
        Single reusable signal block used by all confirmed signal methods.
        extra: optional extra line between levels and footer (e.g. range pips for LB)
        fmt:   price formatter function (defaults to _fmt for crypto, _fmt_fx for forex)
        """
        if fmt is None:
            fmt = self._fmt
        f   = fmt
        d   = direction
        sl_pct  = self._pct(price, sl,  d)
        tp1_pct = self._pct(price, tp1, d)
        tp2_pct = self._pct(price, tp2, d)
        sl_abs  = abs(sl_pct)
        rr      = round(abs(tp2_pct) / sl_abs, 1) if sl_abs else 0
        stars   = self._stars(quality)
        dir_tag = self._dir_tag(d)
        qty     = self._qty_for_risk(price, sl)
        base    = self._base_currency(symbol)
        extra_line = f"\n{extra}" if extra else ""
        return (
            f"{dir_tag}  •  <b>{symbol}</b>  #{no:03d}\n"
            f"{LINE}\n"
            f"{strategy}  {stars}\n"
            f"{DLINE}\n"
            f"📌 Entry  <code>{f(price)}</code>\n"
            f"🛑 SL     <code>{f(sl)}</code>   -{sl_abs:.2f}%\n"
            f"{DLINE}\n"
            f"🎯 TP1   <code>{f(tp1)}</code>   +{tp1_pct:.2f}%   <i>(→ move SL to BE)</i>\n"
            f"🏆 TP2   <code>{f(tp2)}</code>   +{tp2_pct:.2f}%   <i>(close all)</i>\n"
            f"{DLINE}\n"
            f"R:R  1 : {rr}   📦 Qty  <code>{self._fmt_qty(qty)} {base}</code>  <i>(= $5 risk)</i>{extra_line}\n"
            f"<i>{reason}</i>\n"
            f"{self._footer()}"
        )

    # ------------------------------------------------------------------
    # Scanner started
    # ------------------------------------------------------------------

    def scanner_started(self, symbols: list, tf_trend: str, tf_entry: str,
                        cooldown_min: int, paper_enabled: bool = False, paper_balance: float = 0,
                        strategies=None, label: str = "Signal Scanner", mode: str = "swing"):
        mode_label = "SCALP" if mode == "scalp" else "SWING"
        strats = strategies or ["EMA Trend", "S/R Bounce", "BB Breakout", "EMA Ribbon", "RSI Div"]
        strat_str = "  •  ".join(strats)
        paper_line = (
            f"Paper: <b>ON</b>  (<code>${paper_balance:.0f}</code>)"
            if paper_enabled else "Paper: OFF"
        )
        self.send(
            f"🟢 <b>{label} Online</b>\n"
            f"{LINE}\n"
            f"<b>{len(symbols)} pairs</b>  ·  {mode_label}  ·  {tf_trend}/{tf_entry}\n"
            f"{DLINE}\n"
            f"{strat_str}\n"
            f"{DLINE}\n"
            f"{paper_line}\n"
            f"<i>Scanning every 60s — signals post here.</i>"
        )

    # ------------------------------------------------------------------
    # Stage 1 — WARNING
    # ------------------------------------------------------------------

    def warning_signal(self, signal, strategy_name: str = "EMA Momentum"):
        direction = signal.direction if hasattr(signal, 'direction') else signal["direction"]
        symbol    = signal.symbol if hasattr(signal, "symbol") else signal["symbol"]
        price     = signal.entry_price if hasattr(signal, "entry_price") else signal["entry"]
        sl        = signal.stop_loss if hasattr(signal, "stop_loss") else signal["sl"]
        tp3       = signal.tp3 if hasattr(signal, "tp3") else signal["tp3"]
        rsi       = signal.rsi if hasattr(signal, "rsi") else signal["rsi"]
        reason    = signal.reason if hasattr(signal, "reason") else signal["reason"]
        sl_pct    = abs(price - sl) / price * 100
        dir_tag   = self._dir_tag(direction)
        self.send(
            f"⚠️ <b>Setup Forming</b>  [{strategy_name}]\n"
            f"{DLINE}\n"
            f"{dir_tag}  •  <b>{symbol}</b>\n"
            f"Price  <code>{self._fmt(price)}</code>   RSI <code>{rsi:.0f}</code>\n"
            f"🛑 SL  <code>{self._fmt(sl)}</code>  (-{sl_pct:.2f}%)\n"
            f"🏆 TP3 <code>{self._fmt(tp3)}</code>\n"
            f"{DLINE}\n"
            f"<i>{reason}</i>\n"
            f"<i>{self._ts()}</i>"
        )

    # ------------------------------------------------------------------
    # Stage 2 — CONFIRMED (crypto)
    # ------------------------------------------------------------------

    def _confluence_block(self, conf_score: int, conf_labels: list) -> str:
        """Format the confluence section appended to signal messages."""
        from src.confluence import confluence_strength_label
        strength   = confluence_strength_label(conf_score)
        labels_str = "\n".join(conf_labels) if conf_labels else "—"
        return (
            f"\n{DLINE}\n"
            f"<b>Confluence  {conf_score}/5  {strength}</b>\n"
            f"{labels_str}"
        )

    def confirmed_signal(self, signal, strategy_name: str = "EMA Momentum",
                         quality: int = 3, confluence: tuple | None = None):
        self._signal_no += 1
        direction = signal.direction if hasattr(signal, "direction") else signal["direction"]
        symbol    = signal.symbol if hasattr(signal, "symbol") else signal["symbol"]
        price     = signal.entry_price if hasattr(signal, "entry_price") else signal["entry"]
        sl        = signal.stop_loss if hasattr(signal, "stop_loss") else signal["sl"]
        tp1       = signal.tp1 if hasattr(signal, "tp1") else signal["tp1"]
        tp2       = signal.tp2 if hasattr(signal, "tp2") else signal["tp2"]
        tp3       = signal.tp3 if hasattr(signal, "tp3") else signal["tp3"]
        rsi       = signal.rsi if hasattr(signal, "rsi") else signal["rsi"]
        vol       = signal.volume_ratio if hasattr(signal, "volume_ratio") else signal.get("vol_ratio", 0)
        reason    = signal.reason if hasattr(signal, "reason") else signal["reason"]

        extra = f"RSI <code>{rsi:.0f}</code>  ·  Vol <code>{vol:.1f}x</code>"
        msg = self._signal_block(
            self._signal_no, strategy_name, quality,
            direction, symbol, price, sl, tp1, tp2, tp3, reason,
            extra=extra,
        )
        forex_msg = self._signal_block(
            self._signal_no, strategy_name, quality,
            direction, symbol, price, sl, tp1, tp2, tp3, reason,
            extra=f"RSI <code>{rsi:.0f}</code>",
            fmt=self._fmt_fx,
        )
        if confluence:
            conf_block = self._confluence_block(*confluence)
            msg       += conf_block
            forex_msg += conf_block
        self.send_signal(msg, forex_message=forex_msg, is_forex=self._is_forex_symbol(symbol))

    # ------------------------------------------------------------------
    # S/R Bounce confirmed (extra level info)
    # ------------------------------------------------------------------

    def sr_confirmed_signal(self, sig: dict, confluence: tuple | None = None):
        self._signal_no += 1
        quality   = sig.get("quality", 3)
        lv_price  = sig["level_price"]
        lv_touch  = sig["level_touches"]
        direction = sig["direction"]
        price     = sig["entry"]
        sl        = sig["sl"]
        tp1, tp2, tp3 = sig["tp1"], sig["tp2"], sig["tp3"]
        rsi       = sig["rsi"]
        vol       = sig.get("vol_ratio", 0)
        reason    = sig["reason"]
        symbol    = sig["symbol"]
        lv_type   = "Support" if direction == "long" else "Resistance"

        extra = (
            f"{lv_type}: <code>{self._fmt(lv_price)}</code>  ({lv_touch} touches)\n"
            f"RSI <code>{rsi:.0f}</code>  ·  Vol <code>{vol:.1f}x</code>"
        )
        msg = self._signal_block(
            self._signal_no, "S/R Bounce", quality,
            direction, symbol, price, sl, tp1, tp2, tp3, reason, extra=extra,
        )
        forex_msg = self._signal_block(
            self._signal_no, "S/R Bounce", quality,
            direction, symbol, price, sl, tp1, tp2, tp3, reason,
            extra=f"{lv_type}: <code>{self._fmt_fx(lv_price)}</code>  ({lv_touch} touches)",
            fmt=self._fmt_fx,
        )
        if confluence:
            conf_block = self._confluence_block(*confluence)
            msg       += conf_block
            forex_msg += conf_block
        self.send_signal(msg, forex_message=forex_msg, is_forex=self._is_forex_symbol(symbol))

    def sr_warning_signal(self, sig: dict):
        self.warning_signal(sig, strategy_name="S/R Bounce")

    # ------------------------------------------------------------------
    # Forex confirmed signal
    # ------------------------------------------------------------------

    def fx_confirmed_signal(self, sig: dict, strategy_name: str = "FX EMA Trend",
                            force_forex_channel: bool = False):
        self._signal_no += 1
        quality   = sig.get("quality", 3)
        direction = sig["direction"]
        price     = sig["entry"]
        sl        = sig["sl"]
        tp1, tp2, tp3 = sig["tp1"], sig["tp2"], sig["tp3"]
        rsi       = sig["rsi"]
        symbol    = sig["symbol"]
        reason    = sig["reason"]

        extra = f"RSI <code>{rsi:.0f}</code>"
        msg = self._signal_block(
            self._signal_no, strategy_name, quality,
            direction, symbol, price, sl, tp1, tp2, tp3, reason,
            extra=extra, fmt=self._fmt_fx,
        )
        is_forex = force_forex_channel or self._is_forex_symbol(symbol)
        self.send_signal(msg, forex_message=msg, is_forex=is_forex)

    def fx_warning_signal(self, sig: dict, strategy_name: str = "FX EMA Trend"):
        self.warning_signal(sig, strategy_name=strategy_name)

    # ------------------------------------------------------------------
    # London Breakout confirmed
    # ------------------------------------------------------------------

    def lb_confirmed_signal(self, sig: dict, force_forex_channel: bool = False):
        self._signal_no += 1
        quality    = sig.get("quality", 3)
        direction  = sig["direction"]
        price      = sig["entry"]
        sl         = sig["sl"]
        tp1, tp2, tp3 = sig["tp1"], sig["tp2"], sig["tp3"]
        rsi        = sig["rsi"]
        symbol     = sig["symbol"]
        reason     = sig["reason"]
        range_pips = sig.get("range_pips", 0)
        asian_high = sig.get("asian_high", 0)
        asian_low  = sig.get("asian_low", 0)

        extra = (
            f"Asian range: <code>{range_pips:.0f} pips</code>  "
            f"(<code>{self._fmt_fx(asian_low)}</code> – <code>{self._fmt_fx(asian_high)}</code>)\n"
            f"RSI <code>{rsi:.0f}</code>"
        )
        msg = self._signal_block(
            self._signal_no, "London Breakout", quality,
            direction, symbol, price, sl, tp1, tp2, tp3, reason,
            extra=extra, fmt=self._fmt_fx,
        )
        is_forex = force_forex_channel or self._is_forex_symbol(symbol)
        self.send_signal(msg, forex_message=msg, is_forex=is_forex)

    # ------------------------------------------------------------------
    # Paper trading alerts
    # ------------------------------------------------------------------

    def paper_opened(self, pos, available_balance: float, open_count: int = 1, session_count: int = 0):
        dir_tag  = self._dir_tag(pos.direction)
        sl_pct   = abs(pos.entry_price - pos.stop_loss) / pos.entry_price * 100
        session_line = f"  ·  Session <code>{session_count}/50</code>" if session_count else ""
        self.send(
            f"📄 <b>Paper Opened</b>  [{getattr(pos, 'strategy_name', '')}]\n"
            f"{DLINE}\n"
            f"{dir_tag}  •  <b>{pos.symbol}</b>\n"
            f"Entry  <code>{self._fmt(pos.entry_price)}</code>   SL  <code>{self._fmt(pos.stop_loss)}</code>  (-{sl_pct:.2f}%)\n"
            f"TP1 <code>{self._fmt(pos.tp1)}</code>  TP2 <code>{self._fmt(pos.tp2)}</code>  TP3 <code>{self._fmt(pos.tp3)}</code>\n"
            f"{DLINE}\n"
            f"Risk <code>${pos.margin_locked:.2f}</code>  ·  Avail <code>${available_balance:.2f}</code>  ·  Open <code>{open_count}</code>{session_line}"
        )

    def paper_tp1_alert(self, pos, price: float, tp_level: int = 1):
        strat_tag = f"  [{pos.strategy_name}]" if getattr(pos, "strategy_name", "") else ""
        self.send(
            f"🎯 <b>[PAPER] TP{tp_level} — {pos.symbol}</b>{strat_tag}\n"
            f"Price <code>{self._fmt(price)}</code>  →  TP3 <code>{self._fmt(pos.tp3)}</code>\n"
            f"<i>Holding to TP3</i>"
        )

    def paper_tp_hit(self, pos, tp_level: int, price: float, pnl: float, balance: float):
        emojis = {2: "🎯🎯", 3: "🏆"}
        emoji  = emojis.get(tp_level, "🎯")
        strat_tag = f"  [{pos.strategy_name}]" if getattr(pos, "strategy_name", "") else ""
        be_note = "\n🔒 BE activated — SL → entry  ·  Riding to TP3" if tp_level == 2 else ""
        self.send(
            f"{emoji} <b>[PAPER] TP{tp_level} — {pos.symbol}</b>{strat_tag}\n"
            f"Price <code>{self._fmt(price)}</code>   Balance <code>${balance:.2f}</code>{be_note}"
        )

    def paper_closed(self, pos, reason: str, exit_price: float,
                     total_pnl: float, balance: float, tp_level: int = 0,
                     stats=None):
        if tp_level == 3:
            emoji = "🏆"
        elif reason == "SL hit" and pos.be_activated:
            emoji = "🔒"
        elif reason == "SL hit":
            emoji = "🛑"
        else:
            emoji = "✅"

        pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        if pos.direction == "short":
            pct = -pct

        strat_tag = f"  [{pos.strategy_name}]" if getattr(pos, "strategy_name", "") else ""
        stats_line = ""
        if stats and stats["total"] > 0:
            win_pct = stats["wins"] / stats["total"] * 100
            stats_line = (
                f"\n{DLINE}\n"
                f"All-time  {stats['total']} trades  ·  Win <code>{win_pct:.0f}%</code>  ·  "
                f"TP3 <code>{stats['tp3']}</code>  SL <code>{stats['sl']}</code>  BE <code>{stats['be_sl']}</code>"
            )

        self.send(
            f"{emoji} <b>[PAPER] {pos.symbol}</b>{strat_tag}  {reason}\n"
            f"{DLINE}\n"
            f"Entry <code>{self._fmt(pos.entry_price)}</code>  →  Exit <code>{self._fmt(exit_price)}</code>  ({pct:+.2f}%)\n"
            f"PnL <code>{total_pnl:+.2f} USDT</code>  ·  Balance <code>${balance:.2f}</code>"
            f"{stats_line}"
        )

    # ------------------------------------------------------------------
    # Batch / session summaries
    # ------------------------------------------------------------------

    def paper_batch_summary(self, total: int, wins: int, losses: int,
                             total_pnl: float, win_pct: float,
                             start_balance: float, current_balance: float,
                             stats: dict, strategy_stats=None):
        pnl_emoji  = "📈" if total_pnl >= 0 else "📉"
        bal_change = current_balance - start_balance
        all_win_pct = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0

        best_strat_line = ""
        if strategy_stats:
            ranked = sorted(
                strategy_stats.items(),
                key=lambda x: (x[1]["tp3"], x[1]["wins"] / max(x[1]["total"], 1)),
                reverse=True,
            )
            best_name, best = ranked[0]
            best_wr = best["wins"] / best["total"] * 100 if best["total"] > 0 else 0
            best_strat_line = (
                f"\n🥇 <b>{best_name}</b>  TP3 <code>{best['tp3']}</code>  "
                f"Win <code>{best_wr:.0f}%</code>  ({best['total']} trades)"
            )

        self.send(
            f"📋 <b>Batch Report — {total} Trades</b>\n"
            f"{LINE}\n"
            f"✅ <code>{wins}</code>W  ❌ <code>{losses}</code>L  "
            f"Win Rate <code>{win_pct:.0f}%</code>\n"
            f"🏆 TP3 <code>{stats['tp3']}</code>  🛑 SL <code>{stats['sl']}</code>  🔒 BE <code>{stats['be_sl']}</code>\n"
            f"{DLINE}\n"
            f"{pnl_emoji} Batch PnL  <code>{total_pnl:+.2f} USDT</code>\n"
            f"Balance  <code>${current_balance:.2f}</code>  ({bal_change:+.2f})\n"
            f"All-time  <code>{all_win_pct:.0f}%</code> win rate  ({stats['total']} trades)"
            f"{best_strat_line}"
        )

    def paper_session_summary(self, total: int, wins: int, losses: int,
                               total_pnl: float, win_pct: float,
                               start_balance: float, current_balance: float,
                               stats: dict, strategy_stats=None):
        pnl_emoji  = "📈" if total_pnl >= 0 else "📉"
        bal_change = current_balance - start_balance

        best_strat_line = ""
        if strategy_stats:
            ranked = sorted(
                strategy_stats.items(),
                key=lambda x: (x[1]["tp3"], x[1]["wins"] / max(x[1]["total"], 1)),
                reverse=True,
            )
            best_name, best = ranked[0]
            best_wr = best["wins"] / best["total"] * 100 if best["total"] > 0 else 0
            best_strat_line = (
                f"\n🥇 <b>{best_name}</b>  TP3 <code>{best['tp3']}</code>  "
                f"Win <code>{best_wr:.0f}%</code>  ({best['total']} trades)"
            )

        self.send(
            f"📊 <b>Session Complete — {total} Trades</b>\n"
            f"{LINE}\n"
            f"✅ <code>{wins}</code>W  ❌ <code>{losses}</code>L  "
            f"Win Rate <code>{win_pct:.0f}%</code>\n"
            f"🏆 TP3 <code>{stats['tp3']}</code>  🛑 SL <code>{stats['sl']}</code>  🔒 BE <code>{stats['be_sl']}</code>\n"
            f"{DLINE}\n"
            f"{pnl_emoji} Session PnL  <code>{total_pnl:+.2f} USDT</code>\n"
            f"Balance  <code>${current_balance:.2f}</code>  ({bal_change:+.2f})\n"
            f"Started at  <code>${start_balance:.2f}</code>"
            f"{best_strat_line}\n"
            f"{DLINE}\n"
            f"<i>⏸️ Pausing 5 hours — resumes automatically.</i>"
        )

    # ------------------------------------------------------------------
    # Open positions status (hourly)
    # ------------------------------------------------------------------

    def paper_positions_update(self, positions: dict, balance: float, start_balance: float):
        pct   = (balance - start_balance) / start_balance * 100
        emoji = "📈" if balance >= start_balance else "📉"
        if not positions:
            self.send(
                f"📊 <b>Paper Status</b>  No open positions\n"
                f"{emoji} Available <code>${balance:.2f}</code>  ({pct:+.1f}%)"
            )
            return
        lines = []
        for sym, pos in positions.items():
            d  = "🟢" if pos.direction == "long" else "🔴"
            tp = " TP1✅" if pos.tp1_hit else ""
            tp += " TP2✅" if pos.tp2_hit else ""
            be = " 🔒BE" if pos.be_activated else ""
            lines.append(f"{d} <b>{sym}</b> <code>{self._fmt(pos.entry_price)}</code>{tp}{be}")
        self.send(
            f"📊 <b>Paper Positions</b>  ({len(positions)} open)\n"
            f"{DLINE}\n"
            + "\n".join(lines) +
            f"\n{DLINE}\n"
            f"{emoji} Available <code>${balance:.2f}</code>  ({pct:+.1f}%)"
        )

    # ------------------------------------------------------------------
    # Error alert
    # ------------------------------------------------------------------

    def error_alert(self, context: str, error: str):
        self.send(
            f"⚠️ <b>Scanner Error</b>  {context}\n"
            f"<code>{error[:400]}</code>"
        )

    # ------------------------------------------------------------------
    # Forex channel — startup + paper alerts
    # ------------------------------------------------------------------

    def forex_scanner_started(self, symbols: list, tf_trend: str, tf_entry: str,
                              paper_balance: float, mode: str):
        mode_label = "SCALP (30m/15m)" if mode == "scalp" else "SWING (4h/1h)"
        self.send_forex(
            f"🟢 <b>Forex Signals Bot Online</b>\n"
            f"{LINE}\n"
            f"<b>{len(symbols)} pairs</b>  ·  {mode_label}\n"
            f"EMA Trend  ·  London Breakout  ·  Fib Pullback  ·  Daily Level  ·  BB Reversion\n"
            f"{DLINE}\n"
            f"Paper: <b>ON</b>  (<code>${paper_balance:.0f}</code>)\n"
            f"<i>Signals post here when all conditions align.</i>"
        )

    def forex_paper_opened(self, pos, balance: float, open_count: int):
        sl_pct  = abs(pos.entry_price - pos.stop_loss) / pos.entry_price * 100
        dir_tag = self._dir_tag(pos.direction)
        self.send_forex(
            f"📄 <b>Trade Opened</b>  [{pos.strategy_name}]\n"
            f"{DLINE}\n"
            f"{dir_tag}  •  <b>{pos.symbol}</b>\n"
            f"Entry  <code>{self._fmt_fx(pos.entry_price)}</code>   SL  <code>{self._fmt_fx(pos.stop_loss)}</code>  (-{sl_pct:.2f}%)\n"
            f"TP1 <code>{self._fmt_fx(pos.tp1)}</code>  TP2 <code>{self._fmt_fx(pos.tp2)}</code>  TP3 <code>{self._fmt_fx(pos.tp3)}</code>\n"
            f"{DLINE}\n"
            f"Risk <code>${pos.margin_locked:.2f}</code>  ·  Balance <code>${balance:.2f}</code>  ·  Open <code>{open_count}</code>"
        )

    def forex_paper_tp_alert(self, pos, price: float, tp_level: int):
        self.send_forex(
            f"🎯 <b>TP{tp_level} — {pos.symbol}</b>\n"
            f"Price <code>{self._fmt_fx(price)}</code>  →  TP3 <code>{self._fmt_fx(pos.tp3)}</code>  ·  <i>Holding</i>"
        )

    def forex_paper_closed(self, pos, reason: str, exit_price: float,
                           total_pnl: float, balance: float, tp_level: int, stats: dict):
        emoji   = "🏆" if tp_level == 3 else ("🛑" if reason == "SL hit" else "✅")
        pct     = (exit_price - pos.entry_price) / pos.entry_price * 100
        if pos.direction == "short":
            pct = -pct
        win_pct = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
        self.send_forex(
            f"{emoji} <b>{pos.symbol}</b>  {reason}\n"
            f"{DLINE}\n"
            f"Entry <code>{self._fmt_fx(pos.entry_price)}</code>  →  Exit <code>{self._fmt_fx(exit_price)}</code>  ({pct:+.2f}%)\n"
            f"PnL <code>{total_pnl:+.2f} USDT</code>  ·  Balance <code>${balance:.2f}</code>\n"
            f"Stats  {stats['total']} trades  ·  Win <code>{win_pct:.0f}%</code>  ·  TP3 <code>{stats['tp3']}</code>  SL <code>{stats['sl']}</code>"
        )
