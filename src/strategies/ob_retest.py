"""
Strategy — Order Block Retest
-------------------------------
Institutions enter large positions at specific price levels, leaving an
'order block' — the last opposing candle before a strong impulse move.
When price returns to that zone, unfilled institutional orders get triggered.

Bullish OB: the last bearish candle before a strong up-move → demand zone.
            Price returning = institutions buying again.
Bearish OB: the last bullish candle before a strong down-move → supply zone.
            Price returning = institutions selling again.

Logic:
  1. Find unfilled (un-mitigated) OBs on entry_df
  2. Price is currently inside or touching the OB zone
  3. Rejection candle confirms the bounce — not a clean pass-through
  4. HTF EMA50 trend aligned (OBs work best with the major trend)
  5. RSI 45-65, volume confirms participation

Quality (5 binary conditions):
  C1: Valid OB found and price inside it
  C2: Rejection candle in OB zone (bounce_candle_clean)
  C3: HTF EMA50 trend aligned
  C4: RSI in valid range (45-65)
  C5: Volume >= 1.3x average
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators, find_order_blocks, bounce_candle_clean

logger = logging.getLogger("futures_bot.ob_retest")


class OBRetestStrategy:
    NAME = "OB Retest"

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

        ob_cfg = cfg.get("ob_retest", {})
        self.lookback        = ob_cfg.get("lookback_candles", 80)
        self.impulse_candles = ob_cfg.get("impulse_candles", 3)
        self.impulse_atr     = ob_cfg.get("impulse_atr_mult", 2.0)
        self.vol_mult        = ob_cfg.get("volume_multiplier", 1.3)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _quality(self, ob_ok: bool, candle_ok: bool, htf_ok: bool,
                 rsi: float, vol_ratio: float) -> int:
        score = 0
        if ob_ok:                           score += 1  # C1
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

        # HTF trend
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None
        htf_bull = htf_price > ema50_htf
        htf_bear = htf_price < ema50_htf

        obs = find_order_blocks(
            entry_df,
            lookback=self.lookback,
            impulse_candles=self.impulse_candles,
            impulse_atr_mult=self.impulse_atr,
        )
        if not obs:
            return None

        # ── LONG: price inside a bullish OB in bull trend ──────────────────
        if htf_bull:
            bull_obs = [o for o in obs if o["type"] == "bullish"]
            for ob in reversed(bull_obs):
                if ob["low"] <= price <= ob["high"]:
                    candle_ok = bounce_candle_clean(row, "long")
                    quality   = self._quality(True, candle_ok, True, rsi, vol_ratio)
                    if quality < 5:
                        logger.debug(f"[OB] {symbol} LONG quality {quality} < 5 — skip")
                        return None
                    sl_price = ob["low"] - atr * 0.3
                    sl_dist  = price - sl_price
                    return {
                        "stage": 2, "direction": "long", "symbol": symbol,
                        "entry": price,
                        "sl":    sl_price,
                        "tp1":   price + sl_dist * self.tp1_rr,
                        "tp2":   price + sl_dist * self.tp2_rr,
                        "tp3":   price + sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bullish OB Retest ↑ | Zone {ob['low']:.4f}–{ob['high']:.4f} "
                            f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        # ── SHORT: price inside a bearish OB in bear trend ─────────────────
        if htf_bear:
            bear_obs = [o for o in obs if o["type"] == "bearish"]
            for ob in reversed(bear_obs):
                if ob["low"] <= price <= ob["high"]:
                    candle_ok = bounce_candle_clean(row, "short")
                    quality   = self._quality(True, candle_ok, True, rsi, vol_ratio)
                    if quality < 5:
                        logger.debug(f"[OB] {symbol} SHORT quality {quality} < 5 — skip")
                        return None
                    sl_price = ob["high"] + atr * 0.3
                    sl_dist  = sl_price - price
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price,
                        "sl":    sl_price,
                        "tp1":   price - sl_dist * self.tp1_rr,
                        "tp2":   price - sl_dist * self.tp2_rr,
                        "tp3":   price - sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                        "reason": (
                            f"Bearish OB Retest ↓ | Zone {ob['low']:.4f}–{ob['high']:.4f} "
                            f"| Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                        ),
                    }

        return None
