"""
Forex Signal Scanner — Entry Point
-------------------------------------
Usage:
  python forex_main.py
  py -3.12 forex_main.py
  py -3.12 forex_main.py --config forex_config.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

from src.forex_bot import ForexBot


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

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

    fh = logging.FileHandler("logs/forex_scanner.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    fh.setLevel(logging.DEBUG)

    root = logging.getLogger("forex_bot")
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(fh)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Forex Signal Scanner")
    p.add_argument("--config", default="forex_config.json")
    args = p.parse_args()

    setup_logging()
    load_dotenv()

    logger = logging.getLogger("forex_bot")

    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    bot = ForexBot(cfg)
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down forex scanner...")
        bot.stop()


if __name__ == "__main__":
    main()
