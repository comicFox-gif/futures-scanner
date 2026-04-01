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

    def scanner_started(self, symbols: list, tf_trend: str, tf_entry: str, cooldown_min: int):
        self.send(
            f"🟢 <b>Signal Scanner Online</b>\n"
            f"{LINE}\n"
            f"Scanning <b>{len(symbols)} symbols</b>\n"
            f"Trend TF: <code>{tf_trend}</code>  |  Entry TF: <code>{tf_entry}</code>\n"
            f"Cooldown: <code>{cooldown_min} min</code> per signal\n"
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
    # Error alert
    # ------------------------------------------------------------------

    def error_alert(self, context: str, error: str):
        self.send(
            f"⚠️ <b>Scanner Error</b>\n"
            f"{LINE}\n"
            f"Context: {context}\n"
            f"<code>{error[:300]}</code>"
        )
