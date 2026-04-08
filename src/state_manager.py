"""
Paper Trade State Persistence
------------------------------
Saves and loads paper trading state to/from a JSON file so that
balance, positions, stats, and session progress survive redeploys.

On Railway: mount a Volume at /data and set STATE_FILE=/data/paper_state.json
Locally   : defaults to ./paper_state.json
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("futures_bot.state")

STATE_FILE = os.getenv("STATE_FILE", "./paper_state.json")


def _pos_to_dict(pos) -> dict:
    return {
        "symbol":        pos.symbol,
        "direction":     pos.direction,
        "entry_price":   pos.entry_price,
        "stop_loss":     pos.stop_loss,
        "tp1":           pos.tp1,
        "tp2":           pos.tp2,
        "tp3":           pos.tp3,
        "size":          pos.size,
        "size_remaining": pos.size_remaining,
        "margin_locked": pos.margin_locked,
        "strategy_name": getattr(pos, "strategy_name", ""),
        "tp1_hit":       pos.tp1_hit,
        "tp2_hit":       pos.tp2_hit,
        "tp3_hit":       pos.tp3_hit,
        "be_activated":  pos.be_activated,
        "closed_pnl":    pos.closed_pnl,
    }


def _dict_to_pos(d: dict):
    from src.strategy import Position
    pos = Position(
        symbol        = d["symbol"],
        direction     = d["direction"],
        entry_price   = d["entry_price"],
        stop_loss     = d["stop_loss"],
        tp1           = d["tp1"],
        tp2           = d["tp2"],
        tp3           = d["tp3"],
        size          = d["size"],
        size_remaining= d["size_remaining"],
        margin_locked = d.get("margin_locked", 0.0),
        strategy_name = d.get("strategy_name", ""),
        tp1_hit       = d.get("tp1_hit", False),
        tp2_hit       = d.get("tp2_hit", False),
        tp3_hit       = d.get("tp3_hit", False),
        be_activated  = d.get("be_activated", False),
        closed_pnl    = d.get("closed_pnl", 0.0),
    )
    return pos


def save_state(bot) -> None:
    """Snapshot the bot's paper trading state to disk."""
    resume_at = None
    if bot._resume_at:
        resume_at = bot._resume_at.isoformat()

    state = {
        "paper_balance":       bot.paper_balance,
        "paper_start_balance": bot.paper_start_balance,
        "session_count":       bot._session_count,
        "session_paused":      bot._session_paused,
        "resume_at":           resume_at,
        "session_start_bal":   bot._session_start_bal,
        "trade_stats":         bot._trade_stats,
        "strategy_stats":      bot._strategy_stats,
        "signal_no":           bot.notifier._signal_no,
        "positions": {
            sym: _pos_to_dict(pos)
            for sym, pos in bot._paper_positions.items()
        },
    }

    try:
        path = Path(STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temp file then rename — avoids corruption on crash
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(path)
        logger.debug(f"[STATE] Saved to {STATE_FILE}")
    except Exception as e:
        logger.warning(f"[STATE] Save failed: {e}")


def load_state(bot) -> bool:
    """
    Restore paper trading state from disk into bot.
    Returns True if state was loaded, False if no file / fresh start.
    """
    path = Path(STATE_FILE)
    if not path.exists():
        logger.info("[STATE] No state file found — starting fresh")
        return False

    try:
        state = json.loads(path.read_text())

        bot.paper_balance       = state.get("paper_balance",       bot.paper_balance)
        bot.paper_start_balance = state.get("paper_start_balance", bot.paper_start_balance)
        bot._session_count      = state.get("session_count",       0)
        bot._session_paused     = state.get("session_paused",      False)
        bot._session_start_bal  = state.get("session_start_bal",   bot.paper_balance)
        bot._trade_stats        = state.get("trade_stats",         bot._trade_stats)
        bot._strategy_stats     = state.get("strategy_stats",      {})
        bot.notifier._signal_no = state.get("signal_no",           0)

        resume_at_str = state.get("resume_at")
        if resume_at_str:
            bot._resume_at = datetime.fromisoformat(resume_at_str)

        positions = state.get("positions", {})
        bot._paper_positions = {
            sym: _dict_to_pos(d) for sym, d in positions.items()
        }

        logger.info(
            f"[STATE] Loaded — balance=${bot.paper_balance:.2f}  "
            f"positions={len(bot._paper_positions)}  session={bot._session_count}/50  "
            f"signal_no={bot.notifier._signal_no}"
        )
        return True

    except Exception as e:
        logger.warning(f"[STATE] Load failed ({e}) — starting fresh")
        return False
