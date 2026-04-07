"""
Strategy — VWAP Pullback
--------------------------
Institutional anchor. VWAP is where the average participant is positioned
for the day. In a trending market, price pulls back to VWAP and bounces —
institutions re-accumulate/distribute at fair value.

Logic:
  Trend   : HTF EMA50 direction (bull = price > EMA50, bear = below)
  Pullback: price touches or crosses VWAP from the trend side
  Bounce  : current candle closes back on the trend side with momentum
  ADX gate: trend must have strength (ADX >= 22)

Quality (5 binary conditions):
  C1: HTF EMA trend aligned
  C2: Price pulled back to within 0.5x ATR of VWAP
  C3: Bounce candle closes back past VWAP (reclaim)
  C4: Volume >= 1.3x average (institutional re-entry)
  C5: ADX >= 22 AND RSI confirms (long 40-65, short 35-60)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators

logger = logging.getLogger("futures_bot.vwap_pullback")


class VWAPPullbackStrategy:
    NAME = "VWAP Pullback"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]

        self.ema_fast         = s["ema_fast"]
        self.ema_mid          = s["ema_mid"]
        self.ema_slow         = s["ema_slow"]
        self.ema_trend        = s["ema_trend"]
        self.macd_fast        = s["macd_fast"]
        self.macd_slow        = s["macd_slow"]
        self.macd_signal      = s["macd_signal"]
        self.rsi_period       = s["rsi_period"]
        self.atr_period       = s["atr_period"]
        self.volume_sma_period = s["volume_sma_period"]

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.0)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        vwap_cfg = cfg.get("vwap_pullback", {})
        self.touch_atr_mult = vwap_cfg.get("touch_atr_mult", 0.5)
        self.vol_mult       = vwap_cfg.get("volume_multiplier", 1.3)
        self.adx_min        = vwap_cfg.get("adx_min", 22)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, htf_aligned: bool, near_vwap: bool, reclaim: bool,
                 vol_ratio: float, adx: float, rsi: float, direction: str) -> int:
        score = 0
        if htf_aligned:                                            score += 1  # C1
        if near_vwap:                                              score += 1  # C2
        if reclaim:                                                score += 1  # C3
        if vol_ratio >= self.vol_mult:                             score += 1  # C4
        adx_ok  = adx >= self.adx_min
        rsi_ok  = (direction == "long"  and 40 <= rsi <= 65) or \
                  (direction == "short" and 35 <= rsi <= 60)
        if adx_ok and rsi_ok:                                      score += 1  # C5
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        if len(entry_df) < 30:
            return None

        row   = entry_df.iloc[-2]
        prev  = entry_df.iloc[-3]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        vwap = float(row.get("vwap", float("nan")))
        if pd.isna(vwap) or vwap == 0:
            return None

        adx      = float(row.get("adx", 0))
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0

        # HTF trend: EMA50 on HTF
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None

        bull_trend = htf_price > ema50_htf
        bear_trend = htf_price < ema50_htf

        sl_dist    = atr * self.atr_sl_mult
        touch_zone = atr * self.touch_atr_mult

        # LONG: previous candle touched/dipped below VWAP, current closes back above
        prev_close = float(prev["close"])
        near_vwap_long  = abs(prev_close - vwap) <= touch_zone or prev_close < vwap
        reclaim_long     = price > vwap and prev_close <= vwap

        if bull_trend and near_vwap_long and reclaim_long:
            quality = self._quality(True, near_vwap_long, reclaim_long, vol_ratio, adx, rsi, "long")
            if quality < 5:
                return None
            return {
                "stage": 2, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"VWAP Pullback ↑ | VWAP={vwap:.4f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # SHORT: previous touched/broke above VWAP, current closes back below
        near_vwap_short = abs(prev_close - vwap) <= touch_zone or prev_close > vwap
        reclaim_short    = price < vwap and prev_close >= vwap

        if bear_trend and near_vwap_short and reclaim_short:
            quality = self._quality(True, near_vwap_short, reclaim_short, vol_ratio, adx, rsi, "short")
            if quality < 5:
                return None
            return {
                "stage": 2, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    price + sl_dist,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"VWAP Pullback ↓ | VWAP={vwap:.4f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
