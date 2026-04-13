"""
Strategy — Fair Value Gap (FVG) Retest
----------------------------------------
When price moves impulsively it leaves an imbalance zone. Institutions park
limit orders inside these gaps expecting price to return and bounce.

Key insight: FVG retests are PULLBACK entries. Volume is naturally lower on
the pullback — that's a good thing (sellers/buyers are exhausted). RSI can be
at any level — a bullish FVG retest at RSI 38 is a perfect oversold long.

Tuning fixes:
  - Lookback reduced to 20 candles (fresh zones only — stale FVGs less reliable)
  - RSI check: not extreme in wrong direction (< 72 for longs, > 28 for shorts)
  - Volume: 0.8x minimum (just needs activity, not a surge)
  - SL: below/above the ENTIRE zone + 1.0 ATR buffer (gives trade room to breathe)
  - Enter only in the favourable half of the zone (bottom 60% for longs = better R:R)

Quality (5 binary conditions):
  C1: Fresh FVG zone present (< 20 candles old) and price inside it
  C2: Price in favourable entry half of the zone
  C3: Precision candle confirms direction (bounce_candle_clean)
  C4: HTF EMA50 trend aligned
  C5: RSI not overextended against trade (< 72 for long, > 28 for short)
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

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.5)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        fvg_cfg        = cfg.get("fvg_retest", {})
        self.vol_mult  = fvg_cfg.get("volume_multiplier", 0.8)   # low vol on pullback is fine
        self.lookback  = fvg_cfg.get("lookback", 20)              # fresh zones only

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, fvg_ok: bool, zone_half_ok: bool, candle_ok: bool,
                 htf_ok: bool, rsi_ok: bool) -> int:
        score = 0
        if fvg_ok:       score += 1  # C1
        if zone_half_ok: score += 1  # C2
        if candle_ok:    score += 1  # C3
        if htf_ok:       score += 1  # C4
        if rsi_ok:       score += 1  # C5
        return score

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

        # SL placed below/above the entire zone + ATR buffer (room to breathe)
        sl_atr_buf = atr * self.atr_sl_mult

        # ── LONG: price inside a bullish FVG in bull trend ─────────────────
        if htf_bull:
            for z in reversed(zones):   # most recent first
                if z["type"] == "bull" and z["bot"] <= price <= z["top"]:
                    gap  = z["top"] - z["bot"]
                    # Favour entries in the LOWER 60% of the zone (closer to the gap bottom = better R:R)
                    in_lower_half = price <= z["bot"] + gap * 0.60
                    candle_ok     = bounce_candle_clean(p_row, "long")
                    rsi_ok        = rsi < 72           # not overbought; oversold (35-45) is fine for longs
                    vol_ok        = vol_ratio >= self.vol_mult
                    quality       = self._quality(True, in_lower_half, candle_ok, True, rsi_ok)
                    if quality < 4:
                        logger.debug(f"[FVG] {symbol} LONG quality {quality} < 4 — skip")
                        return None
                    sl_price = z["bot"] - sl_atr_buf
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
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bullish FVG Retest ↑ | Zone {z['bot']:.4f}–{z['top']:.4f} "
                            f"| Gap={gap:.4f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        # ── SHORT: price inside a bearish FVG in bear trend ────────────────
        if htf_bear:
            for z in reversed(zones):
                if z["type"] == "bear" and z["bot"] <= price <= z["top"]:
                    gap  = z["top"] - z["bot"]
                    # Favour entries in the UPPER 60% of the zone (closer to gap top = better R:R)
                    in_upper_half = price >= z["top"] - gap * 0.60
                    candle_ok     = bounce_candle_clean(p_row, "short")
                    rsi_ok        = rsi > 28           # not oversold; overbought (55-65) is fine for shorts
                    vol_ok        = vol_ratio >= self.vol_mult
                    quality       = self._quality(True, in_upper_half, candle_ok, True, rsi_ok)
                    if quality < 4:
                        logger.debug(f"[FVG] {symbol} SHORT quality {quality} < 4 — skip")
                        return None
                    sl_price = z["top"] + sl_atr_buf
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
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bearish FVG Retest ↓ | Zone {z['bot']:.4f}–{z['top']:.4f} "
                            f"| Gap={gap:.4f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        return None
