"""
Futures Signal Scanner — Entry Point
--------------------------------------
Usage:
  python main.py
  py -3.12 main.py
  py -3.12 main.py --symbol BTC/USDT:USDT,ETH/USDT:USDT
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

from src.bot import Bot


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

def setup_logging(log_dir: str = "logs"):
    Path(log_dir).mkdir(exist_ok=True)
    colorama_init(autoreset=True)

    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG:    Fore.LIGHTBLACK_EX,
            logging.INFO:     Fore.WHITE,
            logging.WARNING:  Fore.YELLOW,
            logging.ERROR:    Fore.RED,
        }
        HIGHLIGHTS = {
            "CONFIRMED": Fore.GREEN + Style.BRIGHT,
            "WARNING":   Fore.YELLOW + Style.BRIGHT,
            "SIGNAL":    Fore.CYAN,
            "ERROR":     Fore.RED,
        }

        def format(self, record):
            color = self.COLORS.get(record.levelno, "")
            msg = super().format(record)
            for kw, kc in self.HIGHLIGHTS.items():
                if kw in msg:
                    msg = msg.replace(kw, kc + kw + Style.RESET_ALL + color)
            return color + msg + Style.RESET_ALL

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter(fmt, datefmt=date_fmt))
    console.setLevel(logging.INFO)

    fh = logging.FileHandler("logs/scanner.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    fh.setLevel(logging.DEBUG)

    root = logging.getLogger("futures_bot")
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(fh)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Futures Signal Scanner")
    p.add_argument("--config", default="config.json")
    p.add_argument("--symbol", help="Override symbols (comma separated)")
    return p.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging()
    load_dotenv()

    logger = logging.getLogger("futures_bot")

    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    if args.symbol:
        cfg["symbols"] = [s.strip() for s in args.symbol.split(",")]

    # SCALP_MODE — controls crypto bot timeframes
    # SCALP_MODE=true  → scalp  | trend=30m  entry=15m
    # SCALP_MODE=false → swing  | trend=4h   entry=1h
    scalp_mode = os.getenv("SCALP_MODE", "false").lower() == "true"
    if scalp_mode:
        cfg["mode"]             = "scalp"
        cfg["timeframe_trend"]  = "30m"
        cfg["timeframe_entry"]  = "15m"
        logger.info("Mode: SCALP (30m trend / 15m entry)")
    else:
        cfg["mode"]             = "swing"
        cfg["timeframe_trend"]  = "4h"
        cfg["timeframe_entry"]  = "1h"
        logger.info("Mode: SWING (4h trend / 1h entry)")

    # TREND_TF / ENTRY_TF — override individual timeframes without changing mode
    # e.g. TREND_TF=4h ENTRY_TF=30m → swing trend with faster entries
    if os.getenv("TREND_TF"):
        cfg["timeframe_trend"] = os.getenv("TREND_TF")
    if os.getenv("ENTRY_TF"):
        cfg["timeframe_entry"] = os.getenv("ENTRY_TF")
    logger.info(f"Timeframes: trend={cfg['timeframe_trend']} entry={cfg['timeframe_entry']}")

    # FOREX_SCALP_MODE — independent switch for forex paper trading mode label
    # Does not change scanning timeframes (same bot), only the forex startup message
    forex_scalp = os.getenv("FOREX_SCALP_MODE", "false").lower() == "true"
    cfg["forex_paper"] = {
        "balance": float(os.getenv("FOREX_PAPER_BALANCE", "1000")),
        "mode":    "scalp" if forex_scalp else "swing",
    }
    logger.info(f"Forex paper mode: {'SCALP' if forex_scalp else 'SWING'}")

    env = {
        "EXCHANGE":           os.getenv("EXCHANGE", cfg.get("exchange", "bybit")),
        "API_KEY":            os.getenv("API_KEY", ""),
        "API_SECRET":         os.getenv("API_SECRET", ""),
        "API_PASSPHRASE":     os.getenv("API_PASSPHRASE", ""),
        # Bybit
        "BYBIT_KEY":          os.getenv("BYBIT_KEY", ""),
        "BYBIT_SECRET":       os.getenv("BYBIT_SECRET", ""),
        "BYBIT_DEMO":         os.getenv("BYBIT_DEMO", "true"),   # demo trading (api-demo.bybit.com)
        "BYBIT_TESTNET":      os.getenv("BYBIT_TESTNET", "false"),
        "BYBIT_LEVERAGE":     os.getenv("BYBIT_LEVERAGE", "10"),
        # Gate.io
        "GATE_API_KEY":       os.getenv("GATE_API_KEY", ""),
        "GATE_API_SECRET":    os.getenv("GATE_API_SECRET", ""),
        "GATE_TESTNET":       os.getenv("GATE_TESTNET", "true"),
        "GATE_LEVERAGE":      os.getenv("GATE_LEVERAGE", "10"),
        "GATE_RISK_PCT":      os.getenv("GATE_RISK_PCT", "0.01"),
    }

    bot = Bot(cfg, env)
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bot.stop()


if __name__ == "__main__":
    main()
