"""
Strategy — Whale Momentum
--------------------------
Enter WITH institutions the moment they start buying, not after the pump is done.

The problem with most strategies:
  They look for confirmation → confirmation comes AFTER the big move → entry is late.
  Price pumps 15%, pulls back, you enter the pullback, SL hits.

This strategy reads the whale footprint directly:
  - 2.5x+ volume spike on a strong bullish candle = institutional accumulation starting
  - Volume delta (approximated buy vs sell pressure) confirms it's real buying
  - Entry fires on the NEXT candle — you're in at candle 2 of the move, not candle 5
  - Distribution detection manages the exit: when whale volume dries up, get out

Why this catches the big green candles:
  The first candle is the whale entry. That candle is usually +3-8%.
  The second candle (our entry) still has +5-10% remaining before distribution.
  Most strategies only see this setup AFTER candles 3-4 when it's obvious — too late.

Quality (5 binary conditions):
  C1: HTF EMA50 trend aligned (no counter-trend trades)
  C2: Volume spike >= 2.5x average (institutional volume, not retail)
  C3: Candle body >= 55% of range (decisive accumulation, not wicky indecision)
  C4: Volume delta confirms buy pressure >= 1.8x sell pressure
  C5: Fresh signal — first spike, not 3rd candle of an existing pump
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    detect_whale_entry,
    detect_whale_sell,
    detect_bull_trap,
    bounce_candle_clean,
)

logger = logging.getLogger("futures_bot.whale_momentum")


class WhaleMomentumStrategy:
    NAME = "Whale Momentum"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]

        self.ema_fast          = s["ema_fast"]
        self.ema_mid           = s["ema_mid"]
        self.ema_slow          = s["ema_slow"]
        self.ema_trend         = s["ema_trend"]
        self.macd_fast         = s["macd_fast"]
        self.macd_slow         = s["macd_slow"]
        self.macd_signal       = s["macd_signal"]
        self.rsi_period        = s["rsi_period"]
        self.atr_period        = s["atr_period"]
        self.volume_sma_period = s["volume_sma_period"]

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.0)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        wm_cfg = cfg.get("whale_momentum", {})
        self.vol_mult    = wm_cfg.get("volume_multiplier", 2.5)
        self.body_min    = wm_cfg.get("body_min", 0.55)
        self.delta_mult  = wm_cfg.get("delta_mult", 1.8)
        self.lookback    = wm_cfg.get("lookback", 3)
        self.rsi_max     = wm_cfg.get("rsi_max", 70)   # don't long if already overbought
        self.rsi_min     = wm_cfg.get("rsi_min", 30)   # don't short if already oversold

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame,
                        precision_df: pd.DataFrame | None = None) -> dict | None:
        if len(entry_df) < 15:
            return None

        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        # RSI guard — don't enter if already overheated
        if rsi > self.rsi_max:
            logger.debug(f"[WHALE] {symbol} LONG blocked — RSI {rsi:.0f} already overbought")
            return None

        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0

        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None
        htf_bull = htf_price > ema50_htf
        htf_bear = htf_price < ema50_htf

        # SL anchored to precision candle wick — placed just beyond actual high/low, not ATR-based
        if precision_df is not None and len(precision_df) >= 2:
            _p      = precision_df.iloc[-2]
            _p_low  = float(_p["low"])
            _p_high = float(_p["high"])
        else:
            _p_low  = price - atr * self.atr_sl_mult
            _p_high = price + atr * self.atr_sl_mult
        _sl_buf       = atr * 0.2
        sl_dist_long  = max(price - (_p_low  - _sl_buf), atr * 0.3)
        sl_dist_short = max((_p_high + _sl_buf) - price, atr * 0.3)

        # ── LONG: institutions aggressively buying ─────────────────────────
        if htf_bull and rsi <= self.rsi_max:
            whale = detect_whale_entry(
                entry_df,
                vol_mult=self.vol_mult,
                body_min=self.body_min,
                delta_mult=self.delta_mult,
                lookback=self.lookback,
            )
            if whale:
                if detect_bull_trap(entry_df, f"ema_{self.ema_slow}"):
                    logger.debug(f"[WHALE] {symbol} LONG blocked — whale candle looks like bull trap")
                elif not bounce_candle_clean(row, "long"):
                    logger.debug(f"[WHALE] {symbol} LONG blocked — entry candle has large upper wick")
                else:
                    return {
                        "stage": 2, "direction": "long", "symbol": symbol,
                        "entry": price,
                        "sl":    price - sl_dist_long,
                        "tp1":   price + sl_dist_long * self.tp1_rr,
                        "tp2":   price + sl_dist_long * self.tp2_rr,
                        "tp3":   price + sl_dist_long * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": 5, "atr": atr,
                        "reason": f"🐋 {whale['reason']} | RSI={rsi:.0f}",
                    }

        # ── SHORT: institutions aggressively selling ───────────────────────
        if htf_bear and rsi >= self.rsi_min:
            whale_sell = detect_whale_sell(
                entry_df,
                vol_mult=self.vol_mult,
                body_min=self.body_min,
                delta_mult=self.delta_mult,
                lookback=self.lookback,
            )
            if whale_sell:
                if not bounce_candle_clean(row, "short"):
                    logger.debug(f"[WHALE] {symbol} SHORT blocked — entry candle has large lower wick")
                else:
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    price + sl_dist_short,
                        "tp1":   price - sl_dist_short * self.tp1_rr,
                        "tp2":   price - sl_dist_short * self.tp2_rr,
                        "tp3":   price - sl_dist_short * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": 5, "atr": atr,
                        "reason": f"🐋 {whale_sell['reason']} | RSI={rsi:.0f}",
                    }

        return None
