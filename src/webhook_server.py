"""
TradingView Webhook Server
---------------------------
Receives Pine Script alerts via POST /webhook/tradingview.
Validates WEBHOOK_SECRET before accepting.
Puts validated signals onto a shared queue for the bot's main loop.

Setup:
  1. Set env var WEBHOOK_SECRET=<your_secret>
  2. In TradingView alert, set webhook URL to: https://<your-railway-url>/webhook/tradingview
  3. Set alert message to JSON (see Pine Script template below)

Pine Script alert message template (JSON body):
  {
    "secret":    "<WEBHOOK_SECRET>",
    "symbol":    "{{ticker}}",
    "direction": "long",
    "entry":     {{close}},
    "sl":        0,
    "tp1":       0,
    "tp2":       0,
    "tp3":       0,
    "reason":    "TradingView — 4H BOS confirmed",
    "score":     7
  }

Notes:
  - Flask must be installed: pip install flask
  - Server runs in a daemon thread on PORT env var (default 8080)
  - If WEBHOOK_SECRET is not set, the server does NOT start
  - The bot validates the signal further before sending for approval
"""
from __future__ import annotations
import logging
import os
import queue
import threading
from datetime import datetime

logger = logging.getLogger("futures_bot.webhook")


class WebhookServer:
    def __init__(self, signal_queue: queue.Queue):
        self._queue   = signal_queue
        self._secret  = os.getenv("WEBHOOK_SECRET", "").strip()
        self._enabled = bool(self._secret)

    @staticmethod
    def _port() -> int:
        return int(os.getenv("PORT", "8080"))

    def start(self):
        if not self._enabled:
            logger.info("[WEBHOOK] WEBHOOK_SECRET not set — TradingView webhook disabled")
            return
        t = threading.Thread(target=self._run, daemon=True, name="webhook-server")
        t.start()
        logger.info(f"[WEBHOOK] Server started on port {self._port()}")

    def _run(self):
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.warning(
                "[WEBHOOK] Flask not installed — webhook server disabled. "
                "Add 'flask' to requirements.txt and redeploy."
            )
            return

        app = Flask(__name__)
        secret = self._secret
        q      = self._queue

        # Silence Flask's startup banner
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.WARNING)

        @app.route("/webhook/tradingview", methods=["POST"])
        def tv_webhook():
            try:
                data = request.get_json(force=True, silent=True)
                if not data:
                    return jsonify({"ok": False, "error": "no JSON body"}), 400

                if data.get("secret") != secret:
                    logger.warning("[WEBHOOK] Rejected — bad secret")
                    return jsonify({"ok": False, "error": "unauthorized"}), 401

                direction = str(data.get("direction", "")).lower().strip()
                symbol    = str(data.get("symbol", "")).strip()

                if not symbol or direction not in ("long", "short"):
                    return jsonify({"ok": False, "error": "invalid signal"}), 400

                # Normalise symbol: "BTCUSDT" → "BTC/USDT:USDT" if needed
                if "/" not in symbol:
                    base   = symbol.replace("USDT", "").replace("usdt", "")
                    symbol = f"{base.upper()}/USDT:USDT"

                signal = {
                    "source":    "tradingview",
                    "symbol":    symbol,
                    "direction": direction,
                    "entry":     float(data.get("entry", 0)),
                    "sl":        float(data.get("sl",    0)),
                    "tp1":       float(data.get("tp1",   0)),
                    "tp2":       float(data.get("tp2",   0)),
                    "tp3":       float(data.get("tp3",   0)),
                    "reason":    str(data.get("reason",  "TradingView alert")),
                    "score":     int(data.get("score",   7)),
                    "stage":     2,
                    "rsi":       float(data.get("rsi",   50)),
                    "atr":       float(data.get("atr",   0)),
                    "quality":   int(data.get("score",   7)),
                    "vol_ratio": float(data.get("vol_ratio", 1.0)),
                    "received_at": datetime.utcnow().isoformat(),
                }

                # Derive tech/sent scores from total score
                signal["tech_score"] = signal["score"] // 2
                signal["sent_score"] = signal["score"] - signal["tech_score"]

                q.put(signal)
                logger.info(
                    f"[WEBHOOK] TV signal queued: {symbol} {direction} "
                    f"@ {signal['entry']} | score={signal['score']}"
                )
                return jsonify({"ok": True, "symbol": symbol, "direction": direction}), 200

            except Exception as e:
                logger.error(f"[WEBHOOK] Handler error: {e}")
                return jsonify({"ok": False, "error": str(e)}), 500

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "service": "futures-bot"}), 200

        try:
            app.run(
                host="0.0.0.0",
                port=self._port(),
                debug=False,
                use_reloader=False,
                threaded=True,
            )
        except Exception as e:
            logger.error(f"[WEBHOOK] Flask run error: {e}")
