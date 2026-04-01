"""
Telegram Notifier
------------------
Two main alert types:
  warning_signal   — Stage 1: conditions forming, get ready
  confirmed_signal — Stage 2: all conditions met, enter manually
"""

import logging
import os
import requests
from datetime import datetime

logger = logging.getLogger("futures_bot.notifier")

LINE = "─" * 26


class Notifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
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
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=8,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Telegram send failed: {e}")

    # ------------------------------------------------------------------
    # Scanner started
    # ------------------------------------------------------------------

    def scanner_started(self, symbols: list, tf_trend: str, tf_entry: str,
                        cooldown_min: int, paper_enabled: bool = False, paper_balance: float = 0):
        paper_line = (
            f"\n📄 Paper trading: <b>ON</b> (balance: <code>{paper_balance:.0f} USDT</code>)"
            if paper_enabled else "\n📄 Paper trading: OFF"
        )
        self.send(
            f"🟢 <b>Signal Scanner Online</b>\n"
            f"{LINE}\n"
            f"Scanning <b>{len(symbols)} symbols</b>\n"
            f"Trend TF: <code>{tf_trend}</code>  |  Entry TF: <code>{tf_entry}</code>\n"
            f"Cooldown: <code>{cooldown_min} min</code> per signal"
            f"{paper_line}\n"
            f"{LINE}\n"
            f"<b>Symbols:</b>\n"
            f"<code>{chr(10).join(symbols)}</code>\n"
            f"{LINE}\n"
            f"<i>Started {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</i>"
        )

    # ------------------------------------------------------------------
    # Stage 1 — WARNING (conditions forming, no MACD cross yet)
    # ------------------------------------------------------------------

    def warning_signal(self, signal):
        dir_emoji = "📈" if signal.direction == "long" else "📉"
        dir_tag   = "LONG" if signal.direction == "long" else "SHORT"
        sl_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100

        self.send(
            f"⚠️ {dir_emoji} <b>SETUP FORMING — {signal.symbol}</b>\n"
            f"{LINE}\n"
            f"Direction: <b>{dir_tag}</b>\n"
            f"Price now: <code>{signal.entry_price:.4f}</code>\n"
            f"{LINE}\n"
            f"<b>Projected levels (if entry here):</b>\n"
            f"SL:  <code>{signal.stop_loss:.4f}</code>  (-{sl_pct:.2f}%)\n"
            f"TP1: <code>{signal.tp1:.4f}</code>  (+{sl_pct * 1:.2f}%)\n"
            f"TP2: <code>{signal.tp2:.4f}</code>  (+{sl_pct * 2:.2f}%)\n"
            f"TP3: <code>{signal.tp3:.4f}</code>  (+{sl_pct * 3:.2f}%)\n"
            f"{LINE}\n"
            f"RSI: <code>{signal.rsi:.1f}</code>  |  Vol: <code>{signal.volume_ratio:.2f}x avg</code>\n"
            f"ATR: <code>{signal.atr:.4f}</code>\n"
            f"{LINE}\n"
            f"⏳ <i>Waiting for MACD crossover to confirm...</i>\n"
            f"<i>{datetime.utcnow().strftime('%H:%M UTC')}</i>"
        )

    # ------------------------------------------------------------------
    # Stage 2 — CONFIRMED (all conditions met, enter now)
    # ------------------------------------------------------------------

    def confirmed_signal(self, signal):
        dir_emoji = "🟢" if signal.direction == "long" else "🔴"
        dir_tag   = "LONG" if signal.direction == "long" else "SHORT"
        sl_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100

        self.send(
            f"🚨 {dir_emoji} <b>SIGNAL CONFIRMED — {signal.symbol}</b>\n"
            f"{LINE}\n"
            f"Direction: <b>{dir_tag}</b>\n"
            f"Entry now: <code>{signal.entry_price:.4f}</code>\n"
            f"{LINE}\n"
            f"<b>Trade levels:</b>\n"
            f"🛑 SL:  <code>{signal.stop_loss:.4f}</code>  (-{sl_pct:.2f}%)\n"
            f"🎯 TP1: <code>{signal.tp1:.4f}</code>  (+{sl_pct * 1:.2f}%) → move to BE\n"
            f"🎯 TP2: <code>{signal.tp2:.4f}</code>  (+{sl_pct * 2:.2f}%) → trail SL\n"
            f"🏆 TP3: <code>{signal.tp3:.4f}</code>  (+{sl_pct * 3:.2f}%) → full exit\n"
            f"{LINE}\n"
            f"RSI: <code>{signal.rsi:.1f}</code>  |  Vol: <code>{signal.volume_ratio:.2f}x avg</code>\n"
            f"ATR: <code>{signal.atr:.4f}</code>\n"
            f"{LINE}\n"
            f"✅ <b>All conditions met. Enter manually.</b>\n"
            f"<i>{datetime.utcnow().strftime('%H:%M UTC')}</i>"
        )

    # ------------------------------------------------------------------
    # Paper trading alerts
    # ------------------------------------------------------------------

    def paper_opened(self, pos, balance: float):
        dir_tag = "🟢 LONG" if pos.direction == "long" else "🔴 SHORT"
        sl_pct  = abs(pos.entry_price - pos.stop_loss) / pos.entry_price * 100
        self.send(
            f"📄 <b>Paper Trade Opened — {pos.symbol}</b>\n"
            f"{LINE}\n"
            f"Direction: {dir_tag}\n"
            f"Entry:  <code>{pos.entry_price:.4f}</code>\n"
            f"Size:   <code>{pos.size:.4f}</code>\n"
            f"SL:     <code>{pos.stop_loss:.4f}</code>  (-{sl_pct:.2f}%)\n"
            f"TP1:    <code>{pos.tp1:.4f}</code>\n"
            f"TP2:    <code>{pos.tp2:.4f}</code>\n"
            f"TP3:    <code>{pos.tp3:.4f}</code>\n"
            f"{LINE}\n"
            f"Paper balance: <code>{balance:.2f} USDT</code>"
        )

    def paper_tp_hit(self, pos, tp_level: int, price: float, pnl: float, balance: float):
        emojis  = {1: "🎯", 2: "🎯🎯", 3: "🏆"}
        emoji   = emojis.get(tp_level, "🎯")
        be_note = "\n🔒 <b>Break-Even activated</b>" if tp_level == 1 else ""
        trail   = "\n📌 SL trailed to TP1" if tp_level == 2 else ""
        self.send(
            f"{emoji} <b>[PAPER] TP{tp_level} — {pos.symbol}</b>\n"
            f"{LINE}\n"
            f"Price:   <code>{price:.4f}</code>\n"
            f"PnL:     <code>{pnl:+.2f} USDT</code>\n"
            f"Remaining: <code>{pos.size_remaining:.4f}</code>"
            f"{be_note}{trail}\n"
            f"Balance: <code>{balance:.2f} USDT</code>"
        )

    def paper_closed(self, pos, reason: str, exit_price: float,
                     total_pnl: float, balance: float, tp_level: int = 0):
        if total_pnl >= 0:
            emoji = "🏆" if tp_level == 3 else "✅"
        else:
            emoji = "🛑"
        pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        if pos.direction == "short":
            pct = -pct
        self.send(
            f"{emoji} <b>[PAPER] Trade Closed — {pos.symbol}</b>\n"
            f"{LINE}\n"
            f"Reason:  {reason}\n"
            f"Entry:   <code>{pos.entry_price:.4f}</code>\n"
            f"Exit:    <code>{exit_price:.4f}</code>  ({pct:+.2f}%)\n"
            f"Total PnL: <code>{total_pnl:+.2f} USDT</code>\n"
            f"{LINE}\n"
            f"Balance: <code>{balance:.2f} USDT</code>"
        )

    # ------------------------------------------------------------------
    # Error alert
    # ------------------------------------------------------------------

    def error_alert(self, context: str, error: str):
        self.send(
            f"⚠️ <b>Scanner Error</b>\n"
            f"{LINE}\n"
            f"Context: {context}\n"
            f"<code>{error[:300]}</code>"
        )
