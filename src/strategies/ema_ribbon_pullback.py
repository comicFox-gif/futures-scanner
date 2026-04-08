"""
Strategy — EMA Ribbon Pullback
--------------------------------
In a strong trend, price pulls back to the fast EMA and bounces — institutions
re-enter at the moving average. The EMA ribbon (9/21/50) must be perfectly
stacked and expanding, confirming we're in a clean trend, not a range.

Difference from the main EMA strategy:
  Main strategy fires on EMA crossovers (trend inception).
  This fires on pullbacks *during* an established trend — a continuation entry.

Logic:
  Ribbon  : ema9 > ema21 > ema50 (bull) or ema9 < ema21 < ema50 (bear)
  HTF     : HTF price above/below ema50_htf confirms macro direction
  Pullback: Price dips to touch ema21 zone (within 1.0x ATR) in last 3 candles
  Bounce  : Current candle closes back past ema9 with a decisive body
  Filters : ADX >= 25 (real trend), Volume >= 1.3x, RSI not exhausted

Quality (5 binary conditions):
  C1: HTF ema50 aligned with trade direction
  C2: EMA ribbon perfectly stacked (all three in order)
  C3: Pullback touched ema21 zone in last 3 candles
  C4: Bounce candle body >= 0.45 and close past ema9
  C5: Volume >= 1.3x average AND ADX >= 25 AND RSI in range
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators, detect_liquidity_sweep, bounce_candle_clean

logger = logging.getLogger("futures_bot.ema_ribbon_pullback")


class EMARibbonPullbackStrategy:
    NAME = "EMA Ribbon Pullback"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]

        self.ema_fast          = s["ema_fast"]          # 9
        self.ema_mid           = s["ema_mid"]           # 21
        self.ema_slow          = s["ema_slow"]          # 50
        self.ema_trend         = s["ema_trend"]         # 200
        self.macd_fast         = s["macd_fast"]
        self.macd_slow         = s["macd_slow"]
        self.macd_signal       = s["macd_signal"]
        self.rsi_period        = s["rsi_period"]
        self.atr_period        = s["atr_period"]
        self.volume_sma_period = s["volume_sma_period"]

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.2)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        rp_cfg = cfg.get("ema_ribbon_pullback", {})
        self.touch_atr_mult = rp_cfg.get("touch_atr_mult", 1.0)   # how close to ema21 = "touched"
        self.vol_mult       = rp_cfg.get("volume_multiplier", 1.3)
        self.adx_min        = rp_cfg.get("adx_min", 25)
        self.min_body       = rp_cfg.get("min_body", 0.45)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _ribbon_bullish(self, row: pd.Series) -> bool:
        e9  = float(row.get(f"ema_{self.ema_fast}", float("nan")))
        e21 = float(row.get(f"ema_{self.ema_mid}",  float("nan")))
        e50 = float(row.get(f"ema_{self.ema_slow}", float("nan")))
        if any(pd.isna(v) for v in [e9, e21, e50]):
            return False
        return e9 > e21 > e50

    def _ribbon_bearish(self, row: pd.Series) -> bool:
        e9  = float(row.get(f"ema_{self.ema_fast}", float("nan")))
        e21 = float(row.get(f"ema_{self.ema_mid}",  float("nan")))
        e50 = float(row.get(f"ema_{self.ema_slow}", float("nan")))
        if any(pd.isna(v) for v in [e9, e21, e50]):
            return False
        return e9 < e21 < e50

    def _touched_ema21_long(self, entry_df: pd.DataFrame, atr: float) -> bool:
        """Did price touch/dip into ema21 zone within last 3 confirmed candles?"""
        ema_col = f"ema_{self.ema_mid}"
        for i in [-2, -3, -4]:
            if abs(i) > len(entry_df):
                break
            row = entry_df.iloc[i]
            ema21 = float(row.get(ema_col, float("nan")))
            if pd.isna(ema21):
                continue
            # Low dipped to within touch_atr_mult * ATR of ema21
            if row["low"] <= ema21 + atr * self.touch_atr_mult:
                return True
        return False

    def _touched_ema21_short(self, entry_df: pd.DataFrame, atr: float) -> bool:
        """Did price touch/rise into ema21 zone within last 3 confirmed candles?"""
        ema_col = f"ema_{self.ema_mid}"
        for i in [-2, -3, -4]:
            if abs(i) > len(entry_df):
                break
            row = entry_df.iloc[i]
            ema21 = float(row.get(ema_col, float("nan")))
            if pd.isna(ema21):
                continue
            if row["high"] >= ema21 - atr * self.touch_atr_mult:
                return True
        return False

    def _quality(self, htf_aligned: bool, ribbon_ok: bool, touched: bool,
                 bounce_ok: bool, vol_ratio: float, adx: float,
                 rsi: float, direction: str) -> int:
        score = 0
        if htf_aligned:  score += 1  # C1
        if ribbon_ok:    score += 1  # C2
        if touched:      score += 1  # C3
        if bounce_ok:    score += 1  # C4
        rsi_ok = (direction == "long"  and 40 <= rsi <= 60) or \
                 (direction == "short" and 40 <= rsi <= 60)
        if vol_ratio >= self.vol_mult and adx >= self.adx_min and rsi_ok:
            score += 1              # C5
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        if len(entry_df) < 10:
            return None

        row  = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        adx = float(row.get("adx", 0))
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0

        ema9  = float(row.get(f"ema_{self.ema_fast}", float("nan")))
        ema21 = float(row.get(f"ema_{self.ema_mid}",  float("nan")))
        if pd.isna(ema9) or pd.isna(ema21):
            return None

        # HTF alignment: price vs ema50 on HTF
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None
        htf_bull = htf_price > ema50_htf
        htf_bear = htf_price < ema50_htf

        # Candle body ratio
        rng = row["high"] - row["low"]
        body = abs(row["close"] - row["open"]) / rng if rng > 0 else 0

        sl_dist = atr * self.atr_sl_mult

        # ── LONG ──────────────────────────────────────────────────────────
        ribbon_bull  = self._ribbon_bullish(row)
        touched_long = self._touched_ema21_long(entry_df, atr)
        # Bounce: current candle is bullish, body ok, and close is back above ema9
        bounce_long  = (price > ema9 and row["close"] > row["open"] and body >= self.min_body)

        # ── Stop hunt check (runs once, used for both directions) ─────────
        sweep = detect_liquidity_sweep(entry_df)

        if htf_bull and ribbon_bull and touched_long and bounce_long:
            if rsi > 60:                                    # overbought gate
                return None
            if sweep == "buy_side":                         # whales just swept highs → dump risk
                logger.debug(f"[EMA RIBBON] {symbol} LONG blocked — buy-side liquidity sweep detected")
                return None
            if not bounce_candle_clean(row, "long"):        # big upper wick = fake pump
                logger.debug(f"[EMA RIBBON] {symbol} LONG blocked — bounce candle upper wick too large")
                return None
            quality = self._quality(True, True, True, True, vol_ratio, adx, rsi, "long")
            if quality < 4:
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
                    f"EMA Ribbon Pullback ↑ | EMA9={ema9:.4f} EMA21={ema21:.4f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # ── SHORT ─────────────────────────────────────────────────────────
        ribbon_bear   = self._ribbon_bearish(row)
        touched_short = self._touched_ema21_short(entry_df, atr)
        bounce_short  = (price < ema9 and row["close"] < row["open"] and body >= self.min_body)

        if htf_bear and ribbon_bear and touched_short and bounce_short:
            if rsi < 40:                                    # oversold gate
                return None
            if sweep == "sell_side":                        # whales just swept lows → pump risk
                logger.debug(f"[EMA RIBBON] {symbol} SHORT blocked — sell-side liquidity sweep detected")
                return None
            if not bounce_candle_clean(row, "short"):       # big lower wick = fake dump
                logger.debug(f"[EMA RIBBON] {symbol} SHORT blocked — bounce candle lower wick too large")
                return None
            quality = self._quality(True, True, True, True, vol_ratio, adx, rsi, "short")
            if quality < 4:
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
                    f"EMA Ribbon Pullback ↓ | EMA9={ema9:.4f} EMA21={ema21:.4f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
