"""
Strategy — Liquidity Sweep Reversal
--------------------------------------
Whales push price above a swing high (buy-side sweep) to grab short stops,
then immediately reverse and sell hard. Or they push below a swing low
(sell-side sweep) to grab long stops, then buy.

Key insight: sweeps happen at EXTREMES — RSI is overextended at sweep time.
A buy-side sweep SHORT fires when RSI was high (price was running hot).
A sell-side sweep LONG fires when RSI was low (price was running cold).
Old code required RSI 45-65 which is the OPPOSITE of what sweep reversals look like.

Tuning fixes:
  - RSI check flipped: SHORT needs RSI > 55 (was overbought before sweep)
                        LONG  needs RSI < 45 (was oversold before sweep)
  - Volume check on the sweep candle itself (not the current candle)
  - SL placed beyond sweep extreme + 1.5 ATR buffer (wicks can be big)
  - HTF: require reasonable alignment but not hard block

Quality (5 binary conditions):
  C1: Liquidity sweep detected (recent wick beyond structure, closed back inside)
  C2: Precision candle confirms reversal direction
  C3: RSI confirms overextension at time of sweep
  C4: Sweep candle had elevated volume (institutional participation)
  C5: HTF doesn't violently oppose (price within 3% of EMA50)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    bounce_candle_clean,
)

logger = logging.getLogger("futures_bot.liquidity_sweep_reversal")

_STRUCT_LOOKBACK = 20
_SWEEP_LOOKBACK  = 5


def _detect_sweep_with_details(
    df: pd.DataFrame,
    wick_atr_min: float = 0.3,
) -> tuple[str | None, float | None, float | None]:
    """
    Detect liquidity sweep and return (type, level, sweep_candle_vol_ratio).
    Returns (None, None, None) if no sweep.
    """
    min_len = _STRUCT_LOOKBACK + _SWEEP_LOOKBACK + 3
    if len(df) < min_len:
        return None, None, None

    atr = float(df.iloc[-2].get("atr", float("nan")))
    if pd.isna(atr) or atr == 0:
        return None, None, None

    struct_window = df.iloc[-(_STRUCT_LOOKBACK + _SWEEP_LOOKBACK):-_SWEEP_LOOKBACK]
    struct_high   = float(struct_window["high"].max())
    struct_low    = float(struct_window["low"].min())
    min_pierce    = atr * wick_atr_min

    recent = df.iloc[-_SWEEP_LOOKBACK - 1:-1]
    for _, row in recent.iterrows():
        vol_sma = row.get("volume_sma", 0)
        vol_ratio = float(row["volume"]) / float(vol_sma) if vol_sma and float(vol_sma) > 0 else 0
        if row["high"] > struct_high + min_pierce and row["close"] < struct_high:
            return "buy_side", struct_high, vol_ratio
        if row["low"] < struct_low - min_pierce and row["close"] > struct_low:
            return "sell_side", struct_low, vol_ratio

    return None, None, None


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

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.5)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        sr_cfg           = cfg.get("sweep_reversal", {})
        self.sweep_vol_mult = sr_cfg.get("sweep_volume_multiplier", 1.3)  # vol on the SWEEP candle

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
        if len(entry_df) < _STRUCT_LOOKBACK + _SWEEP_LOOKBACK + 10:
            return None

        row   = entry_df.iloc[-2]
        p_row = precision_df.iloc[-2] if precision_df is not None and len(precision_df) >= 2 else row
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        # HTF context
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])

        sweep_type, sweep_level, sweep_vol = _detect_sweep_with_details(entry_df)
        if sweep_type is None or sweep_level is None:
            return None

        sweep_vol = sweep_vol or 0

        # ── LONG: sell-side sweep (lows grabbed → institutions now buying) ──
        if sweep_type == "sell_side":
            candle_ok  = bounce_candle_clean(p_row, "long")
            # Sweep LONG: RSI should be low (oversold at sweep = sellers exhausted)
            rsi_ok     = rsi < 45
            sweep_vol_ok = sweep_vol >= self.sweep_vol_mult
            # HTF: not in violent bear (price within 4% below EMA50 is ok)
            htf_ok     = pd.isna(ema50_htf) or htf_price > ema50_htf * 0.96

            score = sum([True, candle_ok, rsi_ok, sweep_vol_ok, htf_ok])
            if score < 4:
                logger.debug(f"[SWEEP] {symbol} LONG score {score} < 4 — skip")
                return None
            # SL below swept low + buffer
            sl_price = sweep_level - atr * self.atr_sl_mult
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
                "rsi": rsi, "vol_ratio": sweep_vol, "quality": score, "atr": atr,
                "reason": (
                    f"Sell-Side Sweep Reversal ↑ | Lows swept @ {sweep_level:.4f} "
                    f"| RSI={rsi:.0f} | SweepVol={sweep_vol:.1f}x"
                ),
            }

        # ── SHORT: buy-side sweep (highs grabbed → institutions now selling) ─
        if sweep_type == "buy_side":
            candle_ok  = bounce_candle_clean(p_row, "short")
            # Sweep SHORT: RSI should be high (overbought at sweep = buyers exhausted)
            rsi_ok     = rsi > 55
            sweep_vol_ok = sweep_vol >= self.sweep_vol_mult
            # HTF: not in violent bull (price within 4% above EMA50 is ok)
            htf_ok     = pd.isna(ema50_htf) or htf_price < ema50_htf * 1.04

            score = sum([True, candle_ok, rsi_ok, sweep_vol_ok, htf_ok])
            if score < 4:
                logger.debug(f"[SWEEP] {symbol} SHORT score {score} < 4 — skip")
                return None
            # SL above swept high + buffer
            sl_price = sweep_level + atr * self.atr_sl_mult
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
                "rsi": rsi, "vol_ratio": sweep_vol, "quality": score, "atr": atr,
                "reason": (
                    f"Buy-Side Sweep Reversal ↓ | Highs swept @ {sweep_level:.4f} "
                    f"| RSI={rsi:.0f} | SweepVol={sweep_vol:.1f}x"
                ),
            }

        return None
