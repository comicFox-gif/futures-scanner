"""
Strategy — Liquidity Sweep Reversal
--------------------------------------
Whales push price above a swing high (buy-side sweep) to grab short stops,
then immediately reverse and sell hard. Or they push below a swing low
(sell-side sweep) to grab long stops, then buy hard.

This is one of the highest-conviction reversal patterns in institutional trading.
The sweep IS the entry signal — once stops are grabbed and the candle closes back
inside structure, the smart money move has already begun.

Logic:
  1. Detect a recent liquidity sweep on entry_df
  2. buy_side sweep (wick above swing high, closed back below) → SHORT
  3. sell_side sweep (wick below swing low, closed back above) → LONG
  4. Rejection candle after the sweep confirms institutions are reversing
  5. RSI not at extremes, volume elevated on sweep candle

Quality (5 binary conditions):
  C1: Liquidity sweep detected
  C2: Current candle rejecting in reversal direction (bounce_candle_clean)
  C3: RSI in valid range (45-65) — not trading into extremes
  C4: Volume >= 1.5x avg on sweep candle (real institutional move)
  C5: HTF structure supports the reversal (no aggressive opposing trend)
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

logger = logging.getLogger("futures_bot.liquidity_sweep_reversal")

_STRUCT_LOOKBACK = 20
_SWEEP_LOOKBACK  = 5


def _detect_sweep_with_level(
    df: pd.DataFrame,
    wick_atr_min: float = 0.25,
) -> tuple[str | None, float | None]:
    """
    Same logic as detect_liquidity_sweep but also returns the swept level price.
    Returns (sweep_type, sweep_level) or (None, None).
    """
    min_len = _STRUCT_LOOKBACK + _SWEEP_LOOKBACK + 3
    if len(df) < min_len:
        return None, None

    atr = float(df.iloc[-2].get("atr", float("nan")))
    if pd.isna(atr) or atr == 0:
        return None, None

    struct_window = df.iloc[-(_STRUCT_LOOKBACK + _SWEEP_LOOKBACK):-_SWEEP_LOOKBACK]
    struct_high   = float(struct_window["high"].max())
    struct_low    = float(struct_window["low"].min())
    min_pierce    = atr * wick_atr_min

    recent = df.iloc[-_SWEEP_LOOKBACK - 1:-1]
    for _, row in recent.iterrows():
        if row["high"] > struct_high + min_pierce and row["close"] < struct_high:
            return "buy_side", struct_high
        if row["low"] < struct_low - min_pierce and row["close"] > struct_low:
            return "sell_side", struct_low

    return None, None


class LiquiditySweepReversalStrategy:
    NAME = "Sweep Reversal"

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

        sr_cfg           = cfg.get("sweep_reversal", {})
        self.vol_mult    = sr_cfg.get("volume_multiplier", 1.5)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, sweep_ok: bool, candle_ok: bool, rsi: float,
                 vol_ratio: float, htf_ok: bool) -> int:
        score = 0
        if sweep_ok:                        score += 1  # C1
        if candle_ok:                       score += 1  # C2
        if 45 <= rsi <= 65:                 score += 1  # C3
        if vol_ratio >= self.vol_mult:      score += 1  # C4
        if htf_ok:                          score += 1  # C5
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame,
                        precision_df: pd.DataFrame | None = None) -> dict | None:
        if len(entry_df) < _STRUCT_LOOKBACK + _SWEEP_LOOKBACK + 10:
            return None

        row   = entry_df.iloc[-2]
        p_row = precision_df.iloc[-2] if precision_df is not None and len(precision_df) >= 2 else row
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0

        # HTF context — not a hard block, just a quality factor
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])

        sweep_type, sweep_level = _detect_sweep_with_level(entry_df)
        if sweep_type is None or sweep_level is None:
            return None

        sl_dist = atr * self.atr_sl_mult

        # ── LONG: sell-side sweep (lows grabbed → now buying) ──────────────
        if sweep_type == "sell_side":
            candle_ok = bounce_candle_clean(p_row, "long")
            # HTF supports: not in a violent bear (price > EMA50 is ideal but not required)
            htf_ok = not pd.isna(ema50_htf) and htf_price > ema50_htf * 0.97
            quality = self._quality(True, candle_ok, rsi, vol_ratio, htf_ok)
            if quality < 5:
                logger.debug(f"[SWEEP] {symbol} LONG quality {quality} < 5 — skip")
                return None
            # SL below the swept low + small buffer
            sl_price = min(sweep_level - atr * 0.5, price - sl_dist)
            return {
                "stage": 2, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    sl_price,
                "tp1":   price + abs(price - sl_price) * self.tp1_rr,
                "tp2":   price + abs(price - sl_price) * self.tp2_rr,
                "tp3":   price + abs(price - sl_price) * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"Sell-Side Sweep Reversal ↑ | Lows swept @ {sweep_level:.4f} "
                    f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # ── SHORT: buy-side sweep (highs grabbed → now selling) ────────────
        if sweep_type == "buy_side":
            candle_ok = bounce_candle_clean(p_row, "short")
            # HTF supports: not in a violent bull
            htf_ok = not pd.isna(ema50_htf) and htf_price < ema50_htf * 1.03
            quality = self._quality(True, candle_ok, rsi, vol_ratio, htf_ok)
            if quality < 5:
                logger.debug(f"[SWEEP] {symbol} SHORT quality {quality} < 5 — skip")
                return None
            # SL above the swept high + small buffer
            sl_price = max(sweep_level + atr * 0.5, price + sl_dist)
            return {
                "stage": 2, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    sl_price,
                "tp1":   price - abs(sl_price - price) * self.tp1_rr,
                "tp2":   price - abs(sl_price - price) * self.tp2_rr,
                "tp3":   price - abs(sl_price - price) * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"Buy-Side Sweep Reversal ↓ | Highs swept @ {sweep_level:.4f} "
                    f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
