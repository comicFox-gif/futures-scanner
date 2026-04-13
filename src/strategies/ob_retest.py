"""
Strategy — Order Block Retest
-------------------------------
Institutions enter large positions at specific price levels, leaving an
'order block' — the last opposing candle before a strong impulse move.
When price returns to that zone, unfilled institutional orders get triggered.

Key insight: OB retests are PULLBACK entries into demand/supply. Volume is
naturally lower on the return — that's healthy (not a trend continuation).
RSI at an OB can be 38 (oversold demand zone) or 65 (overbought supply zone).
Old RSI 45-65 filter was backwards and killed most valid OB entries.

Tuning fixes:
  - Lookback: 80 → 40 (fresh OBs only — mitigated ones are unreliable)
  - RSI: not overextended against trade (< 72 long, > 28 short)
  - Volume: 0.8x (pullback vol is naturally low, that's ok)
  - SL: below OB low + 1.0 ATR buffer (gives trade room through noise)
  - Quality threshold: 4/5 (was 5/5 which was too strict for zone plays)

Quality (5 binary conditions):
  C1: Valid unmitigated OB found, price inside zone
  C2: Price in entry-friendly half (lower 60% for bull OB, upper 60% for bear)
  C3: Precision candle confirms direction
  C4: HTF EMA50 aligned
  C5: RSI not overextended against trade
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators, find_order_blocks, bounce_candle_clean

logger = logging.getLogger("futures_bot.ob_retest")


class OBRetestStrategy:
    NAME = "OB Retest"

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

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.5)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        ob_cfg = cfg.get("ob_retest", {})
        self.lookback        = ob_cfg.get("lookback_candles", 40)   # fresher OBs
        self.impulse_candles = ob_cfg.get("impulse_candles", 3)
        self.impulse_atr     = ob_cfg.get("impulse_atr_mult", 2.0)
        self.vol_mult        = ob_cfg.get("volume_multiplier", 0.8) # low vol pullback is fine

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
        if len(entry_df) < self.lookback + 10:
            return None

        row   = entry_df.iloc[-2]
        p_row = precision_df.iloc[-2] if precision_df is not None and len(precision_df) >= 2 else row
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0

        # HTF trend
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None
        htf_bull = htf_price > ema50_htf
        htf_bear = htf_price < ema50_htf

        obs = find_order_blocks(
            entry_df,
            lookback=self.lookback,
            impulse_candles=self.impulse_candles,
            impulse_atr_mult=self.impulse_atr,
        )
        if not obs:
            return None

        sl_buf = atr * self.atr_sl_mult  # buffer beyond OB edge

        # ── LONG: price inside a bullish OB in bull trend ──────────────────
        if htf_bull:
            bull_obs = [o for o in obs if o["type"] == "bullish"]
            for ob in reversed(bull_obs):
                if ob["low"] <= price <= ob["high"]:
                    ob_range = ob["high"] - ob["low"]
                    # Enter in lower 60% of OB (closer to low = tighter SL, better R:R)
                    in_lower = price <= ob["low"] + ob_range * 0.60
                    candle_ok = bounce_candle_clean(p_row, "long")
                    rsi_ok    = rsi < 72   # not overbought; any level below 72 is valid
                    vol_ok    = vol_ratio >= self.vol_mult

                    score = sum([True, in_lower, candle_ok, True, rsi_ok])  # C1 HTF always true here
                    if score < 4:
                        logger.debug(f"[OB] {symbol} LONG score {score} < 4 — skip")
                        return None

                    sl_price = ob["low"] - sl_buf
                    sl_dist  = price - sl_price
                    if sl_dist <= 0:
                        return None
                    return {
                        "stage": 2, "direction": "long", "symbol": symbol,
                        "entry": price,
                        "sl":    sl_price,
                        "tp1":   price + sl_dist * self.tp1_rr,
                        "tp2":   price + sl_dist * self.tp2_rr,
                        "tp3":   price + sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": score, "atr": atr,
                        "reason": (
                            f"Bullish OB Retest ↑ | Zone {ob['low']:.4f}–{ob['high']:.4f} "
                            f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        # ── SHORT: price inside a bearish OB in bear trend ─────────────────
        if htf_bear:
            bear_obs = [o for o in obs if o["type"] == "bearish"]
            for ob in reversed(bear_obs):
                if ob["low"] <= price <= ob["high"]:
                    ob_range  = ob["high"] - ob["low"]
                    # Enter in upper 60% of OB (closer to high = tighter SL, better R:R)
                    in_upper  = price >= ob["high"] - ob_range * 0.60
                    candle_ok = bounce_candle_clean(p_row, "short")
                    rsi_ok    = rsi > 28   # not oversold; any level above 28 is valid
                    vol_ok    = vol_ratio >= self.vol_mult

                    score = sum([True, in_upper, candle_ok, True, rsi_ok])
                    if score < 4:
                        logger.debug(f"[OB] {symbol} SHORT score {score} < 4 — skip")
                        return None

                    sl_price = ob["high"] + sl_buf
                    sl_dist  = sl_price - price
                    if sl_dist <= 0:
                        return None
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    sl_price,
                        "tp1":   price - sl_dist * self.tp1_rr,
                        "tp2":   price - sl_dist * self.tp2_rr,
                        "tp3":   price - sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": score, "atr": atr,
                        "reason": (
                            f"Bearish OB Retest ↓ | Zone {ob['low']:.4f}–{ob['high']:.4f} "
                            f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        return None
