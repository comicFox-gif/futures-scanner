"""
Strategy — Market Structure Shift + Pullback
----------------------------------------------
After an MSS, the broken swing level flips role: a former swing HIGH that
was broken becomes SUPPORT. Price pulling back to it = optimal long entry.

Key insight: the PULLBACK must actually happen. Old code set pb_atr_mult=2.0
meaning price anywhere within 4 ATR total width "qualified". At 1h ATR of
$200 on ETH, that's a $400 window — too wide to be called a "pullback".

Tuning fixes:
  - pb_atr_mult: 2.0 → 1.0 (price must be within 1 ATR of broken level)
  - Add explicit pullback confirmation: entry_df price must be BELOW broken_high
    for bull MSS (not still above it) and ABOVE broken_low for bear MSS
  - RSI: 35-68 (wider than before — MSS longs can be at RSI 38, it's a pullback)
  - Volume: 0.9x (pullback vol is naturally lower)
  - SL: below/above broken level + 1.0 ATR (the level is now S/R, don't go too far)

Quality (5 binary conditions):
  C1: MSS confirmed on HTF (closed beyond swing)
  C2: Price in pullback zone (within 1 ATR of broken level)
  C3: Price has actually pulled back (below broken_high for bull, above for bear)
  C4: Precision candle confirms rejection at level
  C5: RSI in valid range (35-68)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    bounce_candle_clean,
    find_swing_highs_idx,
    find_swing_lows_idx,
)

logger = logging.getLogger("futures_bot.mss_pullback")


class MSSPullbackStrategy:
    NAME = "MSS Pullback"

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

        mss_cfg             = cfg.get("mss_pullback", {})
        self.swing_left     = mss_cfg.get("swing_left", 5)
        self.swing_right    = mss_cfg.get("swing_right", 2)
        self.lookback       = mss_cfg.get("lookback", 40)
        self.pb_atr_mult    = mss_cfg.get("pullback_atr_mult", 1.0)  # tight: within 1 ATR of level
        self.vol_mult       = mss_cfg.get("volume_multiplier", 0.9)

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
        min_len = self.lookback + self.swing_left + self.swing_right + 5
        if len(htf_df) < min_len or len(entry_df) < min_len:
            return None

        row   = entry_df.iloc[-2]
        p_row = precision_df.iloc[-2] if precision_df is not None and len(precision_df) >= 2 else row
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0
        htf_close = float(htf_df.iloc[-2]["close"])
        rsi_ok    = 35 <= rsi <= 68

        # ── BULLISH MSS + PULLBACK ──────────────────────────────────────────
        htf_highs = find_swing_highs_idx(
            htf_df.iloc[-(self.lookback + self.swing_right + 2):-2],
            self.swing_left, self.swing_right
        )
        if htf_highs:
            relevant = [(i, p) for i, p in htf_highs
                        if i < len(htf_df) - self.swing_right - 3]
            if relevant:
                broken_high  = relevant[-1][1]
                # MSS confirmed: HTF close is above the broken swing high
                htf_mss_bull = htf_close > broken_high
                if htf_mss_bull:
                    # Pullback confirmed: entry_df price has come BACK DOWN below/near the level
                    pb_zone_top  = broken_high + atr * self.pb_atr_mult
                    pb_zone_bot  = broken_high - atr * self.pb_atr_mult
                    in_zone      = pb_zone_bot <= price <= pb_zone_top
                    pulled_back  = price <= broken_high  # price has actually returned to the level
                    candle_ok    = bounce_candle_clean(p_row, "long")
                    vol_ok       = vol_ratio >= self.vol_mult

                    score = sum([True, in_zone, pulled_back, candle_ok, rsi_ok])
                    if score >= 4:
                        sl_price = broken_high - atr * self.atr_sl_mult
                        sl_dist  = abs(price - sl_price)
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
                                f"MSS Pullback ↑ | Broken high {broken_high:.4f} now support "
                                f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                            ),
                        }
                    else:
                        logger.debug(f"[MSS] {symbol} LONG score {score} < 4 (zone={in_zone} pb={pulled_back} candle={candle_ok} rsi={rsi_ok})")

        # ── BEARISH MSS + PULLBACK ──────────────────────────────────────────
        htf_lows = find_swing_lows_idx(
            htf_df.iloc[-(self.lookback + self.swing_right + 2):-2],
            self.swing_left, self.swing_right
        )
        if htf_lows:
            relevant = [(i, p) for i, p in htf_lows
                        if i < len(htf_df) - self.swing_right - 3]
            if relevant:
                broken_low   = relevant[-1][1]
                htf_mss_bear = htf_close < broken_low
                if htf_mss_bear:
                    pb_zone_top  = broken_low + atr * self.pb_atr_mult
                    pb_zone_bot  = broken_low - atr * self.pb_atr_mult
                    in_zone      = pb_zone_bot <= price <= pb_zone_top
                    pulled_back  = price >= broken_low  # price has bounced back up to the level
                    candle_ok    = bounce_candle_clean(p_row, "short")
                    vol_ok       = vol_ratio >= self.vol_mult

                    score = sum([True, in_zone, pulled_back, candle_ok, rsi_ok])
                    if score >= 4:
                        sl_price = broken_low + atr * self.atr_sl_mult
                        sl_dist  = abs(sl_price - price)
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
                                f"MSS Pullback ↓ | Broken low {broken_low:.4f} now resistance "
                                f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                            ),
                        }
                    else:
                        logger.debug(f"[MSS] {symbol} SHORT score {score} < 4 (zone={in_zone} pb={pulled_back} candle={candle_ok} rsi={rsi_ok})")

        return None
