"""
Combined Runner — Crypto Futures + Forex Signal Scanners
----------------------------------------------------------
Runs both bots in parallel threads from a single process.
This allows Railway free tier (single worker) to run both scanners.

Usage:
  python combined_main.py
  py -3.12 combined_main.py
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

from src.bot import Bot
from src.forex_bot import ForexBot


def setup_logging(log_dir: str = "logs"):
    Path(log_dir).mkdir(exist_ok=True)
    colorama_init(autoreset=True)

    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG:   Fore.LIGHTBLACK_EX,
            logging.INFO:    Fore.WHITE,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR:   Fore.RED,
        }
        HIGHLIGHTS = {
            "CONFIRMED": Fore.GREEN + Style.BRIGHT,
            "WARNING":   Fore.YELLOW + Style.BRIGHT,
            "SIGNAL":    Fore.CYAN,
            "ERROR":     Fore.RED,
        }

        def format(self, record):
            color = self.COLORS.get(record.levelno, "")
            msg   = super().format(record)
            for kw, kc in self.HIGHLIGHTS.items():
                if kw in msg:
                    msg = msg.replace(kw, kc + kw + Style.RESET_ALL + color)
            return color + msg + Style.RESET_ALL

    fmt      = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter(fmt, datefmt=date_fmt))
    console.setLevel(logging.INFO)

    for name, logfile in [("futures_bot", "logs/scanner.log"), ("forex_bot", "logs/forex_scanner.log")]:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        fh.setLevel(logging.DEBUG)
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(console)
        lg.addHandler(fh)


def run_crypto(cfg: dict, env: dict):
    logger = logging.getLogger("futures_bot")
    bot = Bot(cfg, env)
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Crypto bot crashed: {e}")


def run_forex(cfg: dict):
    logger = logging.getLogger("forex_bot")
    bot = ForexBot(cfg)
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Forex bot crashed: {e}")


def main():
    setup_logging()
    load_dotenv()

    logger = logging.getLogger("futures_bot")

    # Load configs
    crypto_cfg_path = Path("config.json")
    forex_cfg_path  = Path("forex_config.json")

    if not crypto_cfg_path.exists():
        logger.error("config.json not found")
        sys.exit(1)
    if not forex_cfg_path.exists():
        logger.error("forex_config.json not found")
        sys.exit(1)

    with open(crypto_cfg_path) as f:
        crypto_cfg = json.load(f)
    with open(forex_cfg_path) as f:
        forex_cfg = json.load(f)

    env = {
        "EXCHANGE":       os.getenv("EXCHANGE", crypto_cfg.get("exchange", "okx")),
        "API_KEY":        os.getenv("API_KEY", ""),
        "API_SECRET":     os.getenv("API_SECRET", ""),
        "API_PASSPHRASE": os.getenv("API_PASSPHRASE", ""),
    }

    logger.info("=" * 60)
    logger.info("Starting Crypto Futures + Forex scanners in parallel")
    logger.info("=" * 60)

    # Stagger start by 10s so both don't blast API at the same time
    crypto_thread = threading.Thread(target=run_crypto, args=(crypto_cfg, env), daemon=True, name="CryptoBot")
    forex_thread  = threading.Thread(target=run_forex,  args=(forex_cfg,),      daemon=True, name="ForexBot")

    crypto_thread.start()
    time.sleep(10)
    forex_thread.start()

    try:
        while True:
            if not crypto_thread.is_alive():
                logger.error("Crypto bot thread died — restarting in 30s")
                time.sleep(30)
                crypto_thread = threading.Thread(target=run_crypto, args=(crypto_cfg, env), daemon=True, name="CryptoBot")
                crypto_thread.start()
            if not forex_thread.is_alive():
                logger.error("Forex bot thread died — restarting in 30s")
                time.sleep(30)
                forex_thread = threading.Thread(target=run_forex, args=(forex_cfg,), daemon=True, name="ForexBot")
                forex_thread.start()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Shutting down both scanners...")


if __name__ == "__main__":
    main()
