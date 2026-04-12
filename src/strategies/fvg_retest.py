"""
Strategy — Fair Value Gap (FVG) Retest
----------------------------------------
When price moves impulsively it leaves an imbalance — a gap between candle 1's
high and candle 3's low (bullish) or candle 1's low and candle 3's high (bearish).
Institutions park unfilled limit orders inside these zones. Price returning to fill
the gap = high-probability reversal point.

Logic:
  1. Detect unfilled FVGs on entry_df (last 40 candles)
  2. Price is currently INSIDE an aligned FVG zone
  3. Rejection candle confirms we're bouncing — not passing through
  4. HTF EMA50 agrees with direction (no counter-trend fading)
  5. RSI 45-65, volume confirms institutional participation

Quality (5 binary conditions):
  C1: FVG zone present and price inside it
  C2: Rejection candle in zone (bounce_candle_clean)
  C3: HTF EMA50 trend aligned
  C4: RSI in valid range (45-65)
  C5: Volume >= 1.3x average
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators, bounce_candle_clean
from src.confluence import detect_fvg_zones

logger = logging.getLogger("futures_bot.fvg_retest")


class FVGRetestStrategy:
    NAME = "FVG Retest"

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

        fvg_cfg        = cfg.get("fvg_retest", {})
        self.vol_mult  = fvg_cfg.get("volume_multiplier", 1.3)
        self.lookback  = fvg_cfg.get("lookback", 40)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, fvg_ok: bool, candle_ok: bool, htf_ok: bool,
                 rsi: float, vol_ratio: float) -> int:
        score = 0
        if fvg_ok:                          score += 1  # C1
        if candle_ok:                       score += 1  # C2
        if htf_ok:                          score += 1  # C3
        if 45 <= rsi <= 65:                 score += 1  # C4
        if vol_ratio >= self.vol_mult:      score += 1  # C5
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        if len(entry_df) < self.lookback + 10:
            return None

        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0

        # HTF trend direction
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None
        htf_bull = htf_price > ema50_htf
        htf_bear = htf_price < ema50_htf

        zones = detect_fvg_zones(entry_df, lookback=self.lookback)
        if not zones:
            return None

        sl_dist = atr * self.atr_sl_mult

        # ── LONG: price inside a bullish FVG in bull trend ─────────────────
        if htf_bull:
            for z in reversed(zones):   # most recent FVG first
                if z["type"] == "bull" and z["bot"] <= price <= z["top"]:
                    candle_ok = bounce_candle_clean(row, "long")
                    quality   = self._quality(True, candle_ok, True, rsi, vol_ratio)
                    if quality < 5:
                        logger.debug(f"[FVG] {symbol} LONG quality {quality} < 5 — skip")
                        return None
                    gap_size  = z["top"] - z["bot"]
                    sl_price  = z["bot"] - atr * 0.3   # just below zone
                    sl_actual = price - max(sl_dist, price - sl_price)
                    return {
                        "stage": 2, "direction": "long", "symbol": symbol,
                        "entry": price,
                        "sl":    sl_actual,
                        "tp1":   price + abs(price - sl_actual) * self.tp1_rr,
                        "tp2":   price + abs(price - sl_actual) * self.tp2_rr,
                        "tp3":   price + abs(price - sl_actual) * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bullish FVG Retest ↑ | Zone {z['bot']:.4f}–{z['top']:.4f} "
                            f"| Gap={gap_size:.4f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        # ── SHORT: price inside a bearish FVG in bear trend ────────────────
        if htf_bear:
            for z in reversed(zones):
                if z["type"] == "bear" and z["bot"] <= price <= z["top"]:
                    candle_ok = bounce_candle_clean(row, "short")
                    quality   = self._quality(True, candle_ok, True, rsi, vol_ratio)
                    if quality < 5:
                        logger.debug(f"[FVG] {symbol} SHORT quality {quality} < 5 — skip")
                        return None
                    gap_size  = z["top"] - z["bot"]
                    sl_price  = z["top"] + atr * 0.3   # just above zone
                    sl_actual = price + max(sl_dist, sl_price - price)
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    sl_actual,
                        "tp1":   price - abs(sl_actual - price) * self.tp1_rr,
                        "tp2":   price - abs(sl_actual - price) * self.tp2_rr,
                        "tp3":   price - abs(sl_actual - price) * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bearish FVG Retest ↓ | Zone {z['bot']:.4f}–{z['top']:.4f} "
                            f"| Gap={gap_size:.4f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        return None
