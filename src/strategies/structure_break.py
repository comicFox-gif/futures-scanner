"""
Strategy — Break of Structure (BOS)
--------------------------------------
Used by ICT, Smart Money Concepts, and institutional prop traders.
The idea: don't buy the dip — buy the CONFIRMATION that the dip is over.

Price makes a swing low → retraces → then BREAKS above the last swing high.
That break = market structure shifted bullish. Enter on the close of that candle.

Why this beats pullbacks:
  Pullback entry assumes the bounce will happen. Often it doesn't.
  BOS entry waits for PROOF the bounce happened before committing.
  You miss the absolute bottom, but you avoid the "pullback that keeps going" trap.

Logic:
  1. Detect the last significant swing high and swing low from recent structure
  2. Bullish BOS: price closes ABOVE the last swing high (structure broken upward)
     — previous candle must have been BELOW the swing high (fresh break, not continuation)
  3. Bearish BOS: price closes BELOW the last swing low (structure broken downward)
  4. HTF EMA50 must agree with the direction
  5. Stop hunt guard: if whales just swept the highs (buy-side sweep), skip longs —
     that's a fake BOS. A sell-side sweep BEFORE a bullish BOS is actually confirmation
     (lows were grabbed, then structure broke = institutional accumulation).

Quality (5 binary conditions):
  C1: HTF EMA50 confirms direction (macro trend aligned)
  C2: Clean structure break — CLOSE beyond swing point, not just a wick
  C3: Volume >= 1.5x average (institutions participating in the break)
  C4: ADX >= 22 (real momentum, not range noise)
  C5: RSI in momentum zone (long: 48–72, short: 28–52) AND body >= 0.50
      (strong decisive candle, not an exhaustion wick)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import (
    compute_all_indicators,
    find_swing_highs_idx,
    find_swing_lows_idx,
    detect_liquidity_sweep,
    bounce_candle_clean,
)

logger = logging.getLogger("futures_bot.structure_break")


class StructureBreakStrategy:
    NAME = "Break of Structure"

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

        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.2)
        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]

        bos_cfg = cfg.get("structure_break", {})
        self.swing_left    = bos_cfg.get("swing_left",  5)   # candles left of swing point
        self.swing_right   = bos_cfg.get("swing_right", 2)   # candles right of swing point
        self.lookback      = bos_cfg.get("lookback",   40)   # candles to search for swings
        self.vol_mult      = bos_cfg.get("volume_multiplier", 1.5)
        self.adx_min       = bos_cfg.get("adx_min", 22)
        self.min_body      = bos_cfg.get("min_body",   0.50)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _last_swing_high(self, df: pd.DataFrame) -> float | None:
        """Most recent confirmed swing high in the last `lookback` candles."""
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        swings = find_swing_highs_idx(window, self.swing_left, self.swing_right)
        if not swings:
            return None
        return swings[-1][1]   # price of the most recent swing high

    def _last_swing_low(self, df: pd.DataFrame) -> float | None:
        """Most recent confirmed swing low in the last `lookback` candles."""
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        swings = find_swing_lows_idx(window, self.swing_left, self.swing_right)
        if not swings:
            return None
        return swings[-1][1]   # price of the most recent swing low

    def _quality(self, htf_ok: bool, vol_ratio: float, adx: float,
                 rsi: float, body: float, direction: str) -> int:
        score = 0
        if htf_ok:                                          score += 1  # C1
        score += 1                                                      # C2: break confirmed (gate passed)
        if vol_ratio >= self.vol_mult:                      score += 1  # C3
        if adx >= self.adx_min:                             score += 1  # C4
        rsi_ok = (direction == "long"  and 48 <= rsi <= 62) or \
                 (direction == "short" and 38 <= rsi <= 52)
        if rsi_ok and body >= self.min_body:                score += 1  # C5
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        min_len = self.lookback + self.swing_left + self.swing_right + 5
        if len(entry_df) < min_len:
            return None

        row   = entry_df.iloc[-2]   # last confirmed close
        prev  = entry_df.iloc[-3]   # the candle before it
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        adx       = float(row.get("adx", 0))
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0
        rng       = row["high"] - row["low"]
        body      = abs(row["close"] - row["open"]) / rng if rng > 0 else 0

        # HTF trend
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None
        htf_bull = htf_price > ema50_htf
        htf_bear = htf_price < ema50_htf

        # Swing structure
        sh = self._last_swing_high(entry_df)
        sl_lvl = self._last_swing_low(entry_df)
        if sh is None or sl_lvl is None:
            return None

        sl_dist = atr * self.atr_sl_mult
        sweep   = detect_liquidity_sweep(entry_df)

        # ── BULLISH BOS ────────────────────────────────────────────────
        # Close breaks above the last swing high; previous close was below it
        if price > sh and float(prev["close"]) <= sh:
            # Fake BOS guard: if whales JUST swept above highs (buy-side sweep),
            # this break may be part of the same stop-hunt pump — skip it.
            # A sell-side sweep (lows grabbed) before the break = bullish confluence.
            if sweep == "buy_side":
                logger.debug(f"[BOS] {symbol} LONG blocked — buy-side sweep before break (fake BOS)")
                return None
            if not bounce_candle_clean(row, "long"):
                logger.debug(f"[BOS] {symbol} LONG blocked — wick-heavy break candle")
                return None

            quality = self._quality(htf_bull, vol_ratio, adx, rsi, body, "long")
            if quality < 5:
                return None

            swing_tag = f"Bull BOS — broke swing high {sh:.4f}"
            if sweep == "sell_side":
                swing_tag += " | Lows swept before break (institutional accumulation)"

            return {
                "stage": 2, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"{swing_tag} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # ── BEARISH BOS ────────────────────────────────────────────────
        # Close breaks below the last swing low; previous close was above it
        if price < sl_lvl and float(prev["close"]) >= sl_lvl:
            if sweep == "sell_side":
                logger.debug(f"[BOS] {symbol} SHORT blocked — sell-side sweep before break (fake BOS)")
                return None
            if not bounce_candle_clean(row, "short"):
                logger.debug(f"[BOS] {symbol} SHORT blocked — wick-heavy break candle")
                return None

            quality = self._quality(htf_bear, vol_ratio, adx, rsi, body, "short")
            if quality < 5:
                return None

            swing_tag = f"Bear BOS — broke swing low {sl_lvl:.4f}"
            if sweep == "buy_side":
                swing_tag += " | Highs swept before break (institutional distribution)"

            return {
                "stage": 2, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    price + sl_dist,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": quality, "atr": atr,
                "reason": (
                    f"{swing_tag} | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
