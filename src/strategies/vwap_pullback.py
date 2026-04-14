"""
Strategy — VWAP Pullback (v2)
---------------------------------
The classic VWAP pullback fails when entered too early — before whales finish
their stop hunt. This version waits for the sweep + reclaim pattern: price
dips BELOW VWAP to grab retail stops, then snaps back. That snap is the entry.

Logic:
  Trend      : HTF EMA50 direction — price must be above (long) / below (short)
  Sweep gate : Price briefly dipped under VWAP in last 3 candles (stop hunt)
               → use sweep low as SL anchor (below the manipulation wick)
  Reclaim    : Current candle closes back above VWAP with a strong body
  Wick reject: Lower wick >= 30% of candle range (rejection off VWAP)
  MACD       : Histogram >= 0 or turning positive (momentum aligned)
  Volume     : >= 1.3x average (institutional buying on the reclaim)
  ADX        : >= 25 (genuine trend, not chop)
  RSI        : 42–68 for longs, 32–58 for shorts (not overbought/oversold)
  VWAP slope : VWAP must be rising (long) / falling (short) — trending VWAP only

Quality scoring (5 points):
  C1: HTF trend + VWAP slope aligned
  C2: Sweep-reclaim detected (best) OR clean pullback to VWAP touch (acceptable)
  C3: Strong bounce candle — wick rejection + body in upper 60% of range
  C4: Volume spike + MACD histogram non-negative
  C5: ADX >= 25 AND RSI in range

Entry only fires at quality == 5.
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators, detect_bull_trap, bull_trap_short_confirmed

logger = logging.getLogger("futures_bot.vwap_pullback")


class VWAPPullbackStrategy:
    NAME = "VWAP Pullback"

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

        vwap_cfg = cfg.get("vwap_pullback", {})
        self.touch_atr_mult = vwap_cfg.get("touch_atr_mult", 0.5)
        self.vol_mult       = vwap_cfg.get("volume_multiplier", 1.3)
        self.adx_min        = vwap_cfg.get("adx_min", 25)   # raised from 22

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vwap_slope(self, df: pd.DataFrame, lookback: int = 3) -> float:
        """Return VWAP change over last `lookback` candles. + = rising."""
        try:
            vwap_now  = float(df.iloc[-2].get("vwap", float("nan")))
            vwap_prev = float(df.iloc[-2 - lookback].get("vwap", float("nan")))
            if pd.isna(vwap_now) or pd.isna(vwap_prev) or vwap_prev == 0:
                return 0.0
            return vwap_now - vwap_prev
        except Exception:
            return 0.0

    def _detect_sweep_reclaim(self, df: pd.DataFrame, vwap: float,
                               direction: str, lookback: int = 3) -> tuple[bool, float]:
        """
        Check if price swept below VWAP (long) or above VWAP (short) in the
        last `lookback` candles and then reclaimed.
        Returns (sweep_detected, sweep_extreme_price).
        sweep_extreme = lowest low (long) / highest high (short) of the sweep candles.
        """
        window = df.iloc[-2 - lookback:-2]   # candles before the signal candle
        if len(window) == 0:
            return False, vwap

        if direction == "long":
            # Any candle dipped below VWAP
            swept = any(float(r["low"]) < vwap for _, r in window.iterrows())
            extreme = float(window["low"].min()) if swept else vwap
        else:
            swept = any(float(r["high"]) > vwap for _, r in window.iterrows())
            extreme = float(window["high"].max()) if swept else vwap

        return swept, extreme

    def _candle_structure(self, row: pd.Series, direction: str) -> tuple[bool, bool]:
        """
        Returns (wick_rejection, strong_body).
        wick_rejection: lower wick >= 30% of range (long) / upper wick >= 30% (short)
        strong_body   : candle closes in upper 60% of its range (long) / lower 60% (short)
        """
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        rng = h - l
        if rng == 0:
            return False, False

        if direction == "long":
            lower_wick    = min(o, c) - l
            wick_rejection = lower_wick / rng >= 0.30
            # close in upper 60% of range
            strong_body   = (c - l) / rng >= 0.60
        else:
            upper_wick    = h - max(o, c)
            wick_rejection = upper_wick / rng >= 0.30
            strong_body   = (h - c) / rng >= 0.60

        return wick_rejection, strong_body

    def _quality(self, htf_aligned: bool, vwap_slope_ok: bool,
                 sweep_reclaim: bool, near_vwap: bool,
                 wick_ok: bool, body_ok: bool,
                 vol_ratio: float, macd_hist: float,
                 adx: float, rsi: float, direction: str) -> int:
        score = 0

        # C1: HTF trend AND VWAP slope both aligned
        if htf_aligned and vwap_slope_ok:
            score += 1

        # C2: Sweep-reclaim (best) OR plain VWAP touch (acceptable but lower quality)
        if sweep_reclaim:
            score += 1
        elif near_vwap:
            score += 1

        # C3: Candle structure — wick rejection AND strong close
        if wick_ok and body_ok:
            score += 1

        # C4: Volume spike AND MACD momentum aligned
        if vol_ratio >= self.vol_mult and macd_hist >= 0:
            score += 1

        # C5: ADX trend strength AND RSI in range
        adx_ok = adx >= self.adx_min
        if direction == "long":
            rsi_ok = 42 <= rsi <= 68
        else:
            rsi_ok = 32 <= rsi <= 58
        if adx_ok and rsi_ok:
            score += 1

        return score

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame,
                        precision_df: pd.DataFrame | None = None) -> dict | None:
        if len(entry_df) < 30:
            return None

        row  = entry_df.iloc[-2]
        prev = entry_df.iloc[-3]

        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        vwap = float(row.get("vwap", float("nan")))
        if pd.isna(vwap) or vwap == 0:
            return None

        adx       = float(row.get("adx", 0))
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0
        macd_hist = float(row.get("macd_hist", 0))

        # HTF trend
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None

        bull_trend = htf_price > ema50_htf
        bear_trend = htf_price < ema50_htf

        touch_zone = atr * self.touch_atr_mult

        # ── LONG ──────────────────────────────────────────────────────
        prev_close = float(prev["close"])
        near_vwap_long = abs(prev_close - vwap) <= touch_zone or prev_close < vwap
        reclaim_long   = price > vwap and prev_close <= vwap

        if bull_trend and reclaim_long:
            # Block bull traps — unless confirmed short fade
            if detect_bull_trap(entry_df, f"ema_{self.ema_slow}"):
                if bull_trap_short_confirmed(entry_df):
                    sl_buf = atr * 0.2
                    if precision_df is not None and len(precision_df) >= 2:
                        _p_high = float(precision_df.iloc[-2]["high"])
                    else:
                        _p_high = price + atr * self.atr_sl_mult
                    sl_dist = max(_p_high + sl_buf - price, atr * 0.3)
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    price + sl_dist,
                        "tp1":   price - sl_dist * self.tp1_rr,
                        "tp2":   price - sl_dist * self.tp2_rr,
                        "tp3":   price - sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": 5, "atr": atr,
                        "reason": f"Bull Trap ↓ Fade | VWAP pump overextended | RSI={rsi:.0f}",
                    }
                logger.debug(f"[VWAP] {symbol} LONG blocked — bull trap")
                return None

            sweep_detected, sweep_low = self._detect_sweep_reclaim(
                entry_df, vwap, "long"
            )
            wick_ok, body_ok = self._candle_structure(row, "long")
            vwap_slope_ok = self._vwap_slope(entry_df) > 0

            # SL: below sweep low if sweep detected, else below precision wick
            sl_buf = atr * 0.15
            if sweep_detected:
                sl_dist = max(price - (sweep_low - sl_buf), atr * 0.4)
                reason_tag = "Sweep+Reclaim ↑"
            else:
                if precision_df is not None and len(precision_df) >= 2:
                    _p_low = float(precision_df.iloc[-2]["low"])
                else:
                    _p_low = price - atr * self.atr_sl_mult
                sl_dist = max(price - (_p_low - sl_buf), atr * 0.4)
                reason_tag = "VWAP Pullback ↑"

            quality = self._quality(
                True, vwap_slope_ok,
                sweep_detected, near_vwap_long,
                wick_ok, body_ok,
                vol_ratio, macd_hist,
                adx, rsi, "long",
            )

            if quality < 5:
                logger.debug(
                    f"[VWAP] {symbol} LONG quality={quality}/5 — skip "
                    f"(sweep={sweep_detected} wick={wick_ok} body={body_ok} "
                    f"slope={vwap_slope_ok} adx={adx:.0f} rsi={rsi:.0f} "
                    f"macd_hist={macd_hist:.4f} vol={vol_ratio:.1f}x)"
                )
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
                    f"{reason_tag} | VWAP={vwap:.4f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # ── SHORT ─────────────────────────────────────────────────────
        near_vwap_short = abs(prev_close - vwap) <= touch_zone or prev_close > vwap
        reclaim_short   = price < vwap and prev_close >= vwap

        if bear_trend and reclaim_short:
            sweep_detected, sweep_high = self._detect_sweep_reclaim(
                entry_df, vwap, "short"
            )
            wick_ok, body_ok = self._candle_structure(row, "short")
            vwap_slope_ok = self._vwap_slope(entry_df) < 0

            sl_buf = atr * 0.15
            if sweep_detected:
                sl_dist = max((sweep_high + sl_buf) - price, atr * 0.4)
                reason_tag = "Sweep+Reclaim ↓"
            else:
                if precision_df is not None and len(precision_df) >= 2:
                    _p_high = float(precision_df.iloc[-2]["high"])
                else:
                    _p_high = price + atr * self.atr_sl_mult
                sl_dist = max(_p_high + sl_buf - price, atr * 0.4)
                reason_tag = "VWAP Pullback ↓"

            quality = self._quality(
                True, vwap_slope_ok,
                sweep_detected, near_vwap_short,
                wick_ok, body_ok,
                vol_ratio, macd_hist,
                adx, rsi, "short",
            )

            if quality < 5:
                logger.debug(
                    f"[VWAP] {symbol} SHORT quality={quality}/5 — skip "
                    f"(sweep={sweep_detected} wick={wick_ok} body={body_ok} "
                    f"slope={vwap_slope_ok} adx={adx:.0f} rsi={rsi:.0f} "
                    f"macd_hist={macd_hist:.4f} vol={vol_ratio:.1f}x)"
                )
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
                    f"{reason_tag} | VWAP={vwap:.4f} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
