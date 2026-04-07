"""
Strategy — Bollinger Band Squeeze Breakout
-------------------------------------------
Volatility compression (squeeze) followed by an explosive directional breakout.
Institutions load positions during low-volatility consolidation; the squeeze
release shows where smart money is positioned.

Logic:
  Squeeze   : BB width drops to a 20-candle low (compression)
  Breakout  : price closes OUTSIDE the band with volume surge
  Trend gate: EMA50 direction confirms we're trading with the trend
  ADX gate  : ADX > 20 confirms real momentum, not a fakeout

Quality (5 binary conditions):
  C1: BB squeeze present (width at 20-bar low)
  C2: Close outside band (actual breakout)
  C3: Volume >= 1.5x avg (institutional participation)
  C4: ADX >= 20 (momentum, not noise)
  C5: RSI confirms direction (long: 45-70, short: 30-55)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators

logger = logging.getLogger("futures_bot.bollinger_breakout")


class BollingerBreakoutStrategy:
    NAME = "BB Breakout"

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

        # Strategy params (can override via config["bollinger_breakout"])
        bb_cfg = cfg.get("bollinger_breakout", {})
        self.squeeze_lookback  = bb_cfg.get("squeeze_lookback", 20)
        self.vol_mult          = bb_cfg.get("volume_multiplier", 1.5)
        self.adx_min           = bb_cfg.get("adx_min", 20)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, squeezed: bool, broke_out: bool, vol_ratio: float,
                 adx: float, rsi: float, direction: str) -> int:
        score = 0
        if squeezed:                                              score += 1  # C1
        if broke_out:                                             score += 1  # C2
        if vol_ratio >= self.vol_mult:                            score += 1  # C3
        if adx >= self.adx_min:                                   score += 1  # C4
        if direction == "long"  and 45 <= rsi <= 70:              score += 1  # C5
        elif direction == "short" and 30 <= rsi <= 55:            score += 1
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        """
        htf_df  : trend timeframe (e.g. 4H) — trend gate only
        entry_df: entry timeframe (e.g. 1H) — signal generation
        """
        if len(entry_df) < self.squeeze_lookback + 5:
            return None

        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        # Need BB columns
        for col in ("bb_upper", "bb_lower", "bb_mid", "bb_width"):
            if col not in entry_df.columns or pd.isna(row.get(col)):
                return None

        bb_upper = float(row["bb_upper"])
        bb_lower = float(row["bb_lower"])
        bb_width = float(row["bb_width"])
        adx      = float(row.get("adx", 0))
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0

        # Squeeze: current BB width is at or near 20-bar minimum
        recent_widths = entry_df["bb_width"].iloc[-(self.squeeze_lookback + 2):-2]
        if recent_widths.empty or recent_widths.isna().all():
            return None
        width_min = recent_widths.min()
        squeezed  = bb_width <= width_min * 1.05  # within 5% of recent minimum

        # Trend gate via EMA50 on HTF
        htf_row    = htf_df.iloc[-2]
        ema50_htf  = htf_row.get(f"ema_{self.ema_slow}", None)
        htf_price  = float(htf_row["close"])
        if ema50_htf is None or pd.isna(ema50_htf):
            return None

        bull_trend = htf_price > float(ema50_htf)
        bear_trend = htf_price < float(ema50_htf)

        sl_dist = atr * self.atr_sl_mult

        # LONG: price closes above upper band in bull trend
        if bull_trend and price > bb_upper:
            quality = self._quality(squeezed, True, vol_ratio, adx, rsi, "long")
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
                    f"BB Squeeze breakout ↑ | Width={bb_width:.2f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # SHORT: price closes below lower band in bear trend
        if bear_trend and price < bb_lower:
            quality = self._quality(squeezed, True, vol_ratio, adx, rsi, "short")
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
                    f"BB Squeeze breakout ↓ | Width={bb_width:.2f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
