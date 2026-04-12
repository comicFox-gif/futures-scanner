"""
Strategy — Market Structure Shift + Pullback
----------------------------------------------
After a Market Structure Shift (MSS), the broken swing level flips role:
a former swing HIGH that was broken becomes SUPPORT on the first pullback.
A former swing LOW that was broken becomes RESISTANCE.

This is the cleanest institutional entry — you're buying the retest of
confirmed structure, not chasing the breakout candle.

Logic:
  1. MSS confirmed on HTF (close above prev swing high for bull / below for bear)
  2. Price has pulled back toward the broken swing level (within ATR tolerance)
  3. Rejection candle at the level confirms institutions defending it
  4. RSI 45-65, volume supports the bounce
  5. ATR-based SL placed below the retested level

Quality (5 binary conditions):
  C1: MSS confirmed on HTF (structure shifted)
  C2: Price within pullback zone (near the broken swing level)
  C3: Rejection candle confirms direction (bounce_candle_clean)
  C4: RSI in valid range (45-65)
  C5: Volume >= 1.3x average
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

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.0)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        mss_cfg             = cfg.get("mss_pullback", {})
        self.swing_left     = mss_cfg.get("swing_left", 5)
        self.swing_right    = mss_cfg.get("swing_right", 2)
        self.lookback       = mss_cfg.get("lookback", 40)
        self.pb_atr_mult    = mss_cfg.get("pullback_atr_mult", 2.0)   # how close to level = "at pullback"
        self.vol_mult       = mss_cfg.get("volume_multiplier", 1.3)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, mss_ok: bool, pb_ok: bool, candle_ok: bool,
                 rsi: float, vol_ratio: float) -> int:
        score = 0
        if mss_ok:                          score += 1  # C1
        if pb_ok:                           score += 1  # C2
        if candle_ok:                       score += 1  # C3
        if 45 <= rsi <= 65:                 score += 1  # C4
        if vol_ratio >= self.vol_mult:      score += 1  # C5
        return score

    def _find_swing_highs(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        return find_swing_highs_idx(window, self.swing_left, self.swing_right)

    def _find_swing_lows(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        return find_swing_lows_idx(window, self.swing_left, self.swing_right)

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

        # ── BULLISH MSS + PULLBACK ──────────────────────────────────────────
        # HTF broke above a swing high = MSS confirmed.
        # Entry: entry_df price pulled back near that broken swing high level.
        htf_highs = find_swing_highs_idx(
            htf_df.iloc[-(self.lookback + self.swing_right + 2):-2],
            self.swing_left, self.swing_right
        )
        if htf_highs:
            # MSS: HTF close broke above the most recent qualifying swing high
            relevant_highs = [(i, p) for i, p in htf_highs
                              if i < len(htf_df) - self.swing_right - 3]
            if relevant_highs:
                broken_high = relevant_highs[-1][1]
                htf_mss_bull = htf_close > broken_high

                if htf_mss_bull:
                    # Pullback: entry_df price has come back near the broken level
                    pb_zone_top = broken_high + atr * self.pb_atr_mult
                    pb_zone_bot = broken_high - atr * self.pb_atr_mult
                    at_pullback = pb_zone_bot <= price <= pb_zone_top

                    candle_ok = bounce_candle_clean(p_row, "long")
                    quality   = self._quality(True, at_pullback, candle_ok, rsi, vol_ratio)
                    if quality >= 5:
                        sl_price = broken_high - atr * self.atr_sl_mult
                        sl_dist  = abs(price - sl_price)
                        return {
                            "stage": 2, "direction": "long", "symbol": symbol,
                            "entry": price,
                            "sl":    sl_price,
                            "tp1":   price + sl_dist * self.tp1_rr,
                            "tp2":   price + sl_dist * self.tp2_rr,
                            "tp3":   price + sl_dist * self.tp3_rr,
                            "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                            "reason": (
                                f"MSS Pullback ↑ | Broken swing high {broken_high:.4f} now support "
                                f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                            ),
                        }
                    else:
                        logger.debug(f"[MSS] {symbol} LONG quality {quality} < 5 — skip")

        # ── BEARISH MSS + PULLBACK ──────────────────────────────────────────
        htf_lows = find_swing_lows_idx(
            htf_df.iloc[-(self.lookback + self.swing_right + 2):-2],
            self.swing_left, self.swing_right
        )
        if htf_lows:
            relevant_lows = [(i, p) for i, p in htf_lows
                             if i < len(htf_df) - self.swing_right - 3]
            if relevant_lows:
                broken_low  = relevant_lows[-1][1]
                htf_mss_bear = htf_close < broken_low

                if htf_mss_bear:
                    pb_zone_top = broken_low + atr * self.pb_atr_mult
                    pb_zone_bot = broken_low - atr * self.pb_atr_mult
                    at_pullback = pb_zone_bot <= price <= pb_zone_top

                    candle_ok = bounce_candle_clean(p_row, "short")
                    quality   = self._quality(True, at_pullback, candle_ok, rsi, vol_ratio)
                    if quality >= 5:
                        sl_price = broken_low + atr * self.atr_sl_mult
                        sl_dist  = abs(sl_price - price)
                        return {
                            "stage": 2, "direction": "short", "symbol": symbol,
                            "entry": price,
                            "sl":    sl_price,
                            "tp1":   price - sl_dist * self.tp1_rr,
                            "tp2":   price - sl_dist * self.tp2_rr,
                            "tp3":   price - sl_dist * self.tp3_rr,
                            "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                            "reason": (
                                f"MSS Pullback ↓ | Broken swing low {broken_low:.4f} now resistance "
                                f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                            ),
                        }
                    else:
                        logger.debug(f"[MSS] {symbol} SHORT quality {quality} < 5 — skip")

        return None
