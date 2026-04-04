"""
Telegram Notifier
------------------
Professional signal format ready for commercialization.
Every confirmed signal gets:
  - Signal number
  - Strategy tag
  - Quality score (stars)
  - Full trade levels with % distances
  - Analysis breakdown
"""

import logging
import os
import requests
from datetime import datetime

logger = logging.getLogger("futures_bot.notifier")

LINE  = "━" * 28
DLINE = "─" * 28


class Notifier:
    def __init__(self, channel_name: str = ""):
        self.token      = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id    = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled    = bool(self.token and self.chat_id)
        self.channel    = channel_name  # e.g. "@YourSignalsChannel"
        self._signal_no = 0            # auto-increments on every confirmed signal

        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram not configured — notifications disabled")

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, message: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id":    self.chat_id,
                    "text":       message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=8,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Telegram send failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stars(self, quality: int) -> str:
        return "⭐" * quality + "☆" * (5 - quality)

    def _dir_tag(self, direction: str) -> str:
        return "🟢 LONG" if direction == "long" else "🔴 SHORT"

    def _footer(self) -> str:
        ts = datetime.utcnow().strftime("%H:%M UTC")
        if self.channel:
            return f"{DLINE}\n⚡ {self.channel}  |  {ts}"
        return f"{DLINE}\n<i>{ts}</i>"

    # ------------------------------------------------------------------
    # Scanner started
    # ------------------------------------------------------------------

    def scanner_started(self, symbols: list, tf_trend: str, tf_entry: str,
                        cooldown_min: int, paper_enabled: bool = False, paper_balance: float = 0,
                        strategies=None, label: str = "Signal Scanner"):
        paper_line = (
            f"\n📄 Paper: <b>ON</b>  balance: <code>{paper_balance:.0f} USDT</code>"
            if paper_enabled else "\n📄 Paper: OFF"
        )
        strats = strategies or [
            "EMA Trend", "S/R Bounce", "Order Block", "Trendline", "RSI Divergence"
        ]
        strat_lines = "\n".join(f"  • {s}" for s in strats)
        self.send(
            f"🟢 <b>{label} Online</b>\n"
            f"{LINE}\n"
            f"Scanning <b>{len(symbols)} pairs</b>\n"
            f"Strategies ({len(strats)}):\n{strat_lines}\n"
            f"Trend TF: <code>{tf_trend}</code>  Entry TF: <code>{tf_entry}</code>\n"
            f"Cooldown: <code>{cooldown_min}min</code>"
            f"{paper_line}\n"
            f"{LINE}\n"
            f"<code>{' | '.join(s.split('/')[0] for s in symbols)}</code>\n"
            f"{DLINE}\n"
            f"<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</i>"
        )

    # ------------------------------------------------------------------
    # Stage 1 — WARNING (setup forming)
    # ------------------------------------------------------------------

    def warning_signal(self, signal, strategy_name: str = "EMA Momentum"):
        dir_tag = self._dir_tag(signal.direction if hasattr(signal, 'direction') else signal["direction"])
        symbol  = signal.symbol if hasattr(signal, "symbol") else signal["symbol"]
        price   = signal.entry_price if hasattr(signal, "entry_price") else signal["entry"]
        sl      = signal.stop_loss if hasattr(signal, "stop_loss") else signal["sl"]
        tp1     = signal.tp1 if hasattr(signal, "tp1") else signal["tp1"]
        tp3     = signal.tp3 if hasattr(signal, "tp3") else signal["tp3"]
        rsi     = signal.rsi if hasattr(signal, "rsi") else signal["rsi"]
        reason  = signal.reason if hasattr(signal, "reason") else signal["reason"]
        sl_pct  = abs(price - sl) / price * 100

        self.send(
            f"⚠️ <b>SETUP FORMING</b>  [{strategy_name}]\n"
            f"{LINE}\n"
            f"{dir_tag}  •  <b>{symbol}</b>\n"
            f"Price: <code>{price:.4f}</code>\n"
            f"{DLINE}\n"
            f"<b>Watch levels:</b>\n"
            f"🛑 SL:  <code>{sl:.4f}</code>  (-{sl_pct:.2f}%)\n"
            f"🎯 TP3: <code>{tp3:.4f}</code>  (+{sl_pct*3:.2f}%)\n"
            f"{DLINE}\n"
            f"RSI <code>{rsi:.1f}</code>\n"
            f"<i>{reason}</i>\n"
            f"{self._footer()}"
        )

    # ------------------------------------------------------------------
    # Stage 2 — CONFIRMED (professional signal format)
    # ------------------------------------------------------------------

    def confirmed_signal(self, signal, strategy_name: str = "EMA Momentum", quality: int = 3):
        self._signal_no += 1
        no      = self._signal_no
        direction = signal.direction if hasattr(signal, "direction") else signal["direction"]
        symbol  = signal.symbol if hasattr(signal, "symbol") else signal["symbol"]
        price   = signal.entry_price if hasattr(signal, "entry_price") else signal["entry"]
        sl      = signal.stop_loss if hasattr(signal, "stop_loss") else signal["sl"]
        tp1     = signal.tp1 if hasattr(signal, "tp1") else signal["tp1"]
        tp2     = signal.tp2 if hasattr(signal, "tp2") else signal["tp2"]
        tp3     = signal.tp3 if hasattr(signal, "tp3") else signal["tp3"]
        rsi     = signal.rsi if hasattr(signal, "rsi") else signal["rsi"]
        vol     = signal.volume_ratio if hasattr(signal, "volume_ratio") else signal.get("vol_ratio", 0)
        reason  = signal.reason if hasattr(signal, "reason") else signal["reason"]
        sl_pct  = abs(price - sl) / price * 100
        dir_tag = self._dir_tag(direction)

        self.send(
            f"🚨 <b>SIGNAL #{no:03d}</b>  [{strategy_name}]\n"
            f"{LINE}\n"
            f"{dir_tag}  •  <b>{symbol}</b>\n"
            f"Quality: {self._stars(quality)}\n"
            f"{DLINE}\n"
            f"Entry:  <code>{price:.4f}</code>\n"
            f"🛑 SL:  <code>{sl:.4f}</code>  (-{sl_pct:.2f}%)\n"
            f"🎯 TP1: <code>{tp1:.4f}</code>  (+{sl_pct*1:.2f}%) → move to BE\n"
            f"🎯 TP2: <code>{tp2:.4f}</code>  (+{sl_pct*2:.2f}%) → trail SL\n"
            f"🏆 TP3: <code>{tp3:.4f}</code>  (+{sl_pct*3:.2f}%) → full exit\n"
            f"R:R = 1 : 3\n"
            f"{DLINE}\n"
            f"📊 RSI: <code>{rsi:.1f}</code>  Vol: <code>{vol:.1f}x avg</code>\n"
            f"<i>{reason}</i>\n"
            f"{self._footer()}"
        )

    # ------------------------------------------------------------------
    # S/R Bounce specific confirmed signal (extra level info)
    # ------------------------------------------------------------------

    def sr_confirmed_signal(self, sig: dict):
        self._signal_no += 1
        no       = self._signal_no
        quality  = sig.get("quality", 3)
        lv_price = sig["level_price"]
        lv_touch = sig["level_touches"]
        direction = sig["direction"]
        price    = sig["entry"]
        sl       = sig["sl"]
        tp1, tp2, tp3 = sig["tp1"], sig["tp2"], sig["tp3"]
        rsi      = sig["rsi"]
        vol      = sig.get("vol_ratio", 0)
        reason   = sig["reason"]
        symbol   = sig["symbol"]
        sl_pct   = abs(price - sl) / price * 100
        dir_tag  = self._dir_tag(direction)
        lv_type  = "Support" if direction == "long" else "Resistance"

        self.send(
            f"🚨 <b>SIGNAL #{no:03d}</b>  [S/R Bounce]\n"
            f"{LINE}\n"
            f"{dir_tag}  •  <b>{symbol}</b>\n"
            f"Quality: {self._stars(quality)}\n"
            f"{DLINE}\n"
            f"Entry:  <code>{price:.4f}</code>\n"
            f"🛑 SL:  <code>{sl:.4f}</code>  (-{sl_pct:.2f}%)\n"
            f"🎯 TP1: <code>{tp1:.4f}</code>  (+{sl_pct*1:.2f}%) → move to BE\n"
            f"🎯 TP2: <code>{tp2:.4f}</code>  (+{sl_pct*2:.2f}%) → trail SL\n"
            f"🏆 TP3: <code>{tp3:.4f}</code>  (+{sl_pct*3:.2f}%) → full exit\n"
            f"R:R = 1 : 3\n"
            f"{DLINE}\n"
            f"📐 {lv_type}: <code>{lv_price:.4f}</code>  ({lv_touch} touches)\n"
            f"📊 RSI: <code>{rsi:.1f}</code>  Vol: <code>{vol:.1f}x avg</code>\n"
            f"<i>{reason}</i>\n"
            f"{self._footer()}"
        )

    def sr_warning_signal(self, sig: dict):
        self.warning_signal(sig, strategy_name="S/R Bounce")

    # ------------------------------------------------------------------
    # Forex confirmed signal (generic — works for both FX strategies)
    # ------------------------------------------------------------------

    def fx_confirmed_signal(self, sig: dict, strategy_name: str = "FX EMA Trend"):
        self._signal_no += 1
        no        = self._signal_no
        quality   = sig.get("quality", 3)
        direction = sig["direction"]
        price     = sig["entry"]
        sl        = sig["sl"]
        tp1, tp2, tp3 = sig["tp1"], sig["tp2"], sig["tp3"]
        rsi       = sig["rsi"]
        symbol    = sig["symbol"]
        reason    = sig["reason"]
        sl_pct    = abs(price - sl) / price * 100
        dir_tag   = self._dir_tag(direction)

        self.send(
            f"🚨 <b>SIGNAL #{no:03d}</b>  [{strategy_name}]\n"
            f"{LINE}\n"
            f"{dir_tag}  •  <b>{symbol}</b>\n"
            f"Quality: {self._stars(quality)}\n"
            f"{DLINE}\n"
            f"Entry:  <code>{price:.5f}</code>\n"
            f"🛑 SL:  <code>{sl:.5f}</code>  (-{sl_pct:.2f}%)\n"
            f"🎯 TP1: <code>{tp1:.5f}</code>  (+{sl_pct*1:.2f}%) → move to BE\n"
            f"🎯 TP2: <code>{tp2:.5f}</code>  (+{sl_pct*2:.2f}%) → trail SL\n"
            f"🏆 TP3: <code>{tp3:.5f}</code>  (+{sl_pct*3:.2f}%) → full exit\n"
            f"R:R = 1 : 3\n"
            f"{DLINE}\n"
            f"📊 RSI: <code>{rsi:.1f}</code>\n"
            f"<i>{reason}</i>\n"
            f"{self._footer()}"
        )

    def fx_warning_signal(self, sig: dict, strategy_name: str = "FX EMA Trend"):
        self.warning_signal(sig, strategy_name=strategy_name)

    # ------------------------------------------------------------------
    # London Breakout confirmed (includes range info)
    # ------------------------------------------------------------------

    def lb_confirmed_signal(self, sig: dict):
        self._signal_no += 1
        no         = self._signal_no
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
        sl_pct     = abs(price - sl) / price * 100
        dir_tag    = self._dir_tag(direction)

        self.send(
            f"🚨 <b>SIGNAL #{no:03d}</b>  [London Breakout]\n"
            f"{LINE}\n"
            f"{dir_tag}  •  <b>{symbol}</b>\n"
            f"Quality: {self._stars(quality)}\n"
            f"{DLINE}\n"
            f"Entry:  <code>{price:.5f}</code>\n"
            f"🛑 SL:  <code>{sl:.5f}</code>  (-{sl_pct:.2f}%)\n"
            f"🎯 TP1: <code>{tp1:.5f}</code>  (+{sl_pct*1:.2f}%) → move to BE\n"
            f"🎯 TP2: <code>{tp2:.5f}</code>  (+{sl_pct*2:.2f}%) → trail SL\n"
            f"🏆 TP3: <code>{tp3:.5f}</code>  (+{sl_pct*3:.2f}%) → full exit\n"
            f"R:R = 1 : 3\n"
            f"{DLINE}\n"
            f"📐 Asian Range: <code>{range_pips:.0f} pips</code>  "
            f"H: <code>{asian_high:.5f}</code>  L: <code>{asian_low:.5f}</code>\n"
            f"📊 RSI: <code>{rsi:.1f}</code>\n"
            f"<i>{reason}</i>\n"
            f"{self._footer()}"
        )

    # ------------------------------------------------------------------
    # Paper trading alerts
    # ------------------------------------------------------------------

    def paper_opened(self, pos, available_balance: float, open_count: int = 1, session_count: int = 0):
        dir_tag  = self._dir_tag(pos.direction)
        sl_pct   = abs(pos.entry_price - pos.stop_loss) / pos.entry_price * 100
        risk_amt = pos.margin_locked
        session_line = f"\nSession:     <code>{session_count}/50</code>" if session_count else ""
        self.send(
            f"📄 <b>Paper Trade Opened</b>\n"
            f"{DLINE}\n"
            f"{dir_tag}  •  <b>{pos.symbol}</b>\n"
            f"Entry:  <code>{pos.entry_price:.5f}</code>\n"
            f"🛑 SL:  <code>{pos.stop_loss:.5f}</code>  (-{sl_pct:.2f}%)\n"
            f"🎯 TP1: <code>{pos.tp1:.5f}</code>\n"
            f"🎯 TP2: <code>{pos.tp2:.5f}</code>\n"
            f"🏆 TP3: <code>{pos.tp3:.5f}</code>\n"
            f"{DLINE}\n"
            f"Risk locked: <code>${risk_amt:.2f}</code>\n"
            f"Available:   <code>${available_balance:.2f}</code>\n"
            f"Open trades: <code>{open_count}</code>"
            f"{session_line}"
        )

    def paper_tp1_alert(self, pos, price: float):
        """TP1 reached — notify only, no position change, SL stays at original."""
        strat_tag = f"  [{pos.strategy_name}]" if getattr(pos, "strategy_name", "") else ""
        self.send(
            f"🎯 <b>[PAPER] TP1 reached — {pos.symbol}</b>{strat_tag}\n"
            f"{DLINE}\n"
            f"Price: <code>{price:.5f}</code>\n"
            f"<i>Holding full position — waiting for TP2 to activate Break-Even</i>"
        )

    def paper_tp_hit(self, pos, tp_level: int, price: float, pnl: float, balance: float):
        emojis = {2: "🎯🎯", 3: "🏆"}
        emoji  = emojis.get(tp_level, "🎯")
        strat_tag = f"  [{pos.strategy_name}]" if getattr(pos, "strategy_name", "") else ""
        be_note = "\n📌 <b>Holding to TP3 — SL unchanged</b>" if tp_level == 2 else ""
        self.send(
            f"{emoji} <b>[PAPER] TP{tp_level} — {pos.symbol}</b>{strat_tag}\n"
            f"{DLINE}\n"
            f"Price:     <code>{price:.5f}</code>\n"
            f"PnL:       <code>{pnl:+.2f} USDT</code>\n"
            f"Remaining: <code>{pos.size_remaining:.4f}</code>"
            f"{be_note}\n"
            f"Balance: <code>{balance:.2f} USDT</code>"
        )

    def paper_closed(self, pos, reason: str, exit_price: float,
                     total_pnl: float, balance: float, tp_level: int = 0,
                     stats=None):
        if tp_level == 3:
            emoji = "🏆"
        elif reason == "SL hit" and pos.be_activated:
            emoji = "🔒"   # BE-SL: break-even stop hit
        elif reason == "SL hit":
            emoji = "🛑"
        else:
            emoji = "✅"
        pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        if pos.direction == "short":
            pct = -pct

        strat_tag = f"  [{pos.strategy_name}]" if getattr(pos, "strategy_name", "") else ""
        # Show strategy win note on TP2/TP3
        strat_win_line = ""
        if tp_level in (2, 3) and getattr(pos, "strategy_name", ""):
            strat_win_line = f"\n🏅 Strategy: <b>{pos.strategy_name}</b>"

        stats_line = ""
        if stats and stats["total"] > 0:
            win_pct = stats["wins"] / stats["total"] * 100
            stats_line = (
                f"\n{DLINE}\n"
                f"📊 <b>All-time</b> ({stats['total']} trades)  Win: <code>{win_pct:.0f}%</code>\n"
                f"🏆 TP3: <code>{stats['tp3']}</code>  "
                f"🛑 SL: <code>{stats['sl']}</code>  "
                f"🔒 BE-SL: <code>{stats['be_sl']}</code>"
            )

        self.send(
            f"{emoji} <b>[PAPER] Closed — {pos.symbol}</b>{strat_tag}\n"
            f"{DLINE}\n"
            f"Reason:    {reason}\n"
            f"Entry:     <code>{pos.entry_price:.5f}</code>\n"
            f"Exit:      <code>{exit_price:.5f}</code>  ({pct:+.2f}%)\n"
            f"Total PnL: <code>{total_pnl:+.2f} USDT</code>"
            f"{strat_win_line}\n"
            f"{DLINE}\n"
            f"Balance: <code>${balance:.2f}</code>"
            f"{stats_line}"
        )

    # ------------------------------------------------------------------
    # Batch summary (fires after 9/10 positions close, before resume)
    # ------------------------------------------------------------------

    def paper_batch_summary(self, total: int, wins: int, losses: int,
                             total_pnl: float, win_pct: float,
                             start_balance: float, current_balance: float,
                             stats: dict, strategy_stats=None):
        pnl_emoji  = "📈" if total_pnl >= 0 else "📉"
        bal_change = current_balance - start_balance
        all_win_pct = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0

        # Best strategy by TP3 count, then win rate
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
                f"\n{DLINE}\n"
                f"🥇 <b>Best Strategy: {best_name}</b>\n"
                f"   TP3: <code>{best['tp3']}</code>  TP2: <code>{best['tp2']}</code>  "
                f"SL: <code>{best['sl']}</code>  Win: <code>{best_wr:.0f}%</code>  "
                f"({best['total']} trades)"
            )

        self.send(
            f"📋 <b>Batch Report — {total} Trades</b>\n"
            f"{LINE}\n"
            f"✅ Wins:   <code>{wins}</code>\n"
            f"❌ Losses: <code>{losses}</code>\n"
            f"🎯 Win Rate: <code>{win_pct:.0f}%</code>\n"
            f"{DLINE}\n"
            f"🏆 TP3:   <code>{stats['tp3']}</code>  "
            f"🛑 SL: <code>{stats['sl']}</code>  "
            f"🔒 BE-SL: <code>{stats['be_sl']}</code>\n"
            f"{DLINE}\n"
            f"{pnl_emoji} Batch PnL:  <code>{total_pnl:+.2f} USDT</code>\n"
            f"💰 Balance:  <code>${current_balance:.2f}</code>  "
            f"({bal_change:+.2f} this batch)\n"
            f"{DLINE}\n"
            f"📊 All-time win rate: <code>{all_win_pct:.0f}%</code>  "
            f"({stats['total']} total trades)"
            f"{best_strat_line}\n"
            f"<i>New entries starting now...</i>"
        )

    # ------------------------------------------------------------------
    # Session summary (fires after all 50 trades open+close)
    # ------------------------------------------------------------------

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
                f"\n{DLINE}\n"
                f"🥇 <b>Best Strategy: {best_name}</b>\n"
                f"   TP3: <code>{best['tp3']}</code>  "
                f"SL: <code>{best['sl']}</code>  "
                f"Win: <code>{best_wr:.0f}%</code>  ({best['total']} trades)"
            )

        self.send(
            f"📊 <b>SESSION COMPLETE — 50 Trades</b>\n"
            f"{LINE}\n"
            f"✅ Wins:     <code>{wins}</code>\n"
            f"❌ Losses:   <code>{losses}</code>\n"
            f"🎯 Win Rate: <code>{win_pct:.0f}%</code>\n"
            f"{DLINE}\n"
            f"🏆 TP3:   <code>{stats['tp3']}</code>  "
            f"🛑 SL:  <code>{stats['sl']}</code>  "
            f"🔒 BE-SL: <code>{stats['be_sl']}</code>\n"
            f"{DLINE}\n"
            f"{pnl_emoji} Session PnL: <code>{total_pnl:+.2f} USDT</code>\n"
            f"💰 Balance:  <code>${current_balance:.2f}</code>  ({bal_change:+.2f})\n"
            f"📌 Started:  <code>${start_balance:.2f}</code>"
            f"{best_strat_line}\n"
            f"{DLINE}\n"
            f"<i>⏸️ Pausing 5 hours — next session starts automatically.</i>"
        )

    # ------------------------------------------------------------------
    # Open positions status (hourly)
    # ------------------------------------------------------------------

    def paper_positions_update(self, positions: dict, balance: float, start_balance: float):
        pct   = (balance - start_balance) / start_balance * 100
        emoji = "📈" if balance >= start_balance else "📉"
        if not positions:
            self.send(
                f"📊 <b>Paper Status</b>\n"
                f"{DLINE}\n"
                f"No open positions\n"
                f"{emoji} Available: <code>${balance:.2f}</code>  ({pct:+.1f}%)"
            )
            return
        lines = []
        for sym, pos in positions.items():
            d  = "🟢" if pos.direction == "long" else "🔴"
            tp = "✅TP1" if pos.tp1_hit else ""
            tp += " ✅TP2" if pos.tp2_hit else ""
            be = " 🔒BE" if pos.be_activated else ""
            lines.append(f"{d} <b>{sym}</b> @ <code>{pos.entry_price:.5f}</code>{tp}{be}")
        self.send(
            f"📊 <b>Paper Positions Update</b>\n"
            f"{DLINE}\n"
            f"Open: <b>{len(positions)}</b>\n"
            + "\n".join(lines) +
            f"\n{DLINE}\n"
            f"{emoji} Available: <code>${balance:.2f}</code>  ({pct:+.1f}%)\n"
            f"<i>{datetime.utcnow().strftime('%H:%M UTC')}</i>"
        )

    # ------------------------------------------------------------------
    # Error alert
    # ------------------------------------------------------------------

    def error_alert(self, context: str, error: str):
        self.send(
            f"⚠️ <b>Scanner Error</b>\n"
            f"{DLINE}\n"
            f"Context: {context}\n"
            f"<code>{error[:300]}</code>"
        )
