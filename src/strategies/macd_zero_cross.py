"""
Strategy — MACD Zero Line Cross
---------------------------------
One of the highest-precision momentum signals in professional trading.

The difference between MACD signal-line cross (common) vs zero-line cross (pro):
  Signal cross: fast EMA temporarily diverges from slow EMA — many false signals
  Zero cross  : fast EMA has FULLY crossed the slow EMA — the trend has actually shifted

When the MACD line (12-26 EMA difference) crosses ABOVE zero:
  → EMA12 > EMA26: short-term momentum is fully above long-term momentum
  → This is the definition of a bull trend confirmation
  → Price tends to follow through strongly — no guesswork needed

Filters stack to eliminate noise:
  HTF trend aligned: macro context agrees
  Price above EMA200: only trade with the major trend
  ADX >= 25: real momentum, not sideways chop
  RSI 45-62: momentum zone — not already overbought at entry
  Volume >= 1.3x: real participation, not thin air moves
  Clean candle body >= 0.45: decisive candle, not wicky indecision
  No buy-side liquidity sweep: not a fake pump

Quality (5 binary conditions):
  C1: HTF EMA50 aligned
  C2: Price on correct side of EMA200 (major trend filter)
  C3: Volume >= 1.3x AND ADX >= 25
  C4: RSI in momentum zone AND body >= 0.45
  C5: Zero cross just happened on last 3 confirmed candles (fresh signal)
"""

from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators, detect_liquidity_sweep, bounce_candle_clean, detect_bull_trap, bull_trap_short_confirmed

logger = logging.getLogger("futures_bot.macd_zero_cross")


class MACDZeroCrossStrategy:
    NAME = "MACD Zero Cross"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]

        self.ema_fast          = s["ema_fast"]
        self.ema_mid           = s["ema_mid"]
        self.ema_slow          = s["ema_slow"]
        self.ema_trend         = s["ema_trend"]        # 200
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

        mz_cfg = cfg.get("macd_zero_cross", {})
        self.cross_lookback = mz_cfg.get("cross_lookback", 3)   # candles to look back for the cross
        self.vol_mult       = mz_cfg.get("volume_multiplier", 1.3)
        self.adx_min        = mz_cfg.get("adx_min", 25)
        self.min_body       = mz_cfg.get("min_body", 0.45)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    def _zero_cross_bullish(self, df: pd.DataFrame) -> bool:
        """
        MACD line crossed above zero within the last `cross_lookback` confirmed candles.
        Requires: previous candle had MACD < 0, a candle within lookback has MACD > 0.
        """
        lookback = min(self.cross_lookback + 1, len(df) - 2)
        for i in range(2, lookback + 2):
            curr = df.iloc[-i]
            prev = df.iloc[-i - 1]
            if prev["macd"] < 0 and curr["macd"] > 0:
                return True
        return False

    def _zero_cross_bearish(self, df: pd.DataFrame) -> bool:
        """MACD line crossed below zero within the last `cross_lookback` confirmed candles."""
        lookback = min(self.cross_lookback + 1, len(df) - 2)
        for i in range(2, lookback + 2):
            curr = df.iloc[-i]
            prev = df.iloc[-i - 1]
            if prev["macd"] > 0 and curr["macd"] < 0:
                return True
        return False

    def _quality(self, htf_ok: bool, ema200_ok: bool, vol_ratio: float,
                 adx: float, rsi: float, body: float,
                 fresh_cross: bool, direction: str) -> int:
        score = 0
        if htf_ok:                                              score += 1  # C1
        if ema200_ok:                                           score += 1  # C2
        if vol_ratio >= self.vol_mult and adx >= self.adx_min: score += 1  # C3
        rsi_ok = (direction == "long"  and 45 <= rsi <= 65) or \
                 (direction == "short" and 45 <= rsi <= 65)  # min 45 avoids shorting oversold
        if rsi_ok and body >= self.min_body:                    score += 1  # C4
        if fresh_cross:                                         score += 1  # C5
        return score

    def generate_signal(self, symbol: str, htf_df: pd.DataFrame,
                        entry_df: pd.DataFrame) -> dict | None:
        if len(entry_df) < 40:
            return None

        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        macd_now = float(row.get("macd", float("nan")))
        if pd.isna(macd_now):
            return None

        adx       = float(row.get("adx", 0))
        vol_ratio = row["volume"] / row["volume_sma"] if row.get("volume_sma", 0) > 0 else 0
        rng       = row["high"] - row["low"]
        body      = abs(row["close"] - row["open"]) / rng if rng > 0 else 0

        # HTF alignment
        htf_row   = htf_df.iloc[-2]
        ema50_htf = float(htf_row.get(f"ema_{self.ema_slow}", float("nan")))
        htf_price = float(htf_row["close"])
        if pd.isna(ema50_htf):
            return None
        htf_bull = htf_price > ema50_htf
        htf_bear = htf_price < ema50_htf

        # EMA200 filter — only trade in the direction of the major trend
        ema200 = float(row.get(f"ema_{self.ema_trend}", float("nan")))
        if pd.isna(ema200):
            return None
        above_ema200 = price > ema200
        below_ema200 = price < ema200

        sl_dist = atr * self.atr_sl_mult
        sweep   = detect_liquidity_sweep(entry_df)

        # ── BULLISH ZERO CROSS ─────────────────────────────────────────
        if macd_now > 0 and above_ema200 and htf_bull:
            if not self._zero_cross_bullish(entry_df):
                return None   # MACD has been above zero for too long — not a fresh cross
            if sweep == "buy_side":
                logger.debug(f"[MACD0] {symbol} LONG blocked — buy-side sweep (fake pump)")
                return None
            if detect_bull_trap(entry_df, f"ema_{self.ema_slow}"):
                if bull_trap_short_confirmed(entry_df):
                    logger.debug(f"[MACD0] {symbol} bull trap → fading with SHORT")
                    return {
                        "stage": 2, "direction": "short", "symbol": symbol,
                        "entry": price, "sl": price + sl_dist,
                        "tp1": price - sl_dist * self.tp1_rr,
                        "tp2": price - sl_dist * self.tp2_rr,
                        "tp3": price - sl_dist * self.tp3_rr,
                        "rsi": rsi, "vol_ratio": vol_ratio, "quality": 5, "atr": atr,
                        "reason": f"Bull Trap ↓ Fade | MACD pump overextended | RSI={rsi:.0f}",
                    }
                logger.debug(f"[MACD0] {symbol} LONG blocked — bull trap (no wick confirmation)")
                return None
            if rsi > 62:
                logger.debug(f"[MACD0] {symbol} LONG blocked — RSI {rsi:.0f} too high")
                return None
            if not bounce_candle_clean(row, "long"):
                logger.debug(f"[MACD0] {symbol} LONG blocked — upper wick rejection")
                return None

            quality = self._quality(True, True, vol_ratio, adx, rsi, body, True, "long")
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
                    f"MACD zero cross ↑ | EMA200 aligned | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        # ── BEARISH ZERO CROSS ─────────────────────────────────────────
        if macd_now < 0 and below_ema200 and htf_bear:
            if not self._zero_cross_bearish(entry_df):
                return None
            if sweep == "sell_side":
                logger.debug(f"[MACD0] {symbol} SHORT blocked — sell-side sweep (fake dump)")
                return None
            if rsi < 38:
                logger.debug(f"[MACD0] {symbol} SHORT blocked — RSI {rsi:.0f} too low")
                return None
            if not bounce_candle_clean(row, "short"):
                logger.debug(f"[MACD0] {symbol} SHORT blocked — lower wick rejection")
                return None

            quality = self._quality(True, True, vol_ratio, adx, rsi, body, True, "short")
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
                    f"MACD zero cross ↓ | EMA200 aligned | "
                    f"ADX={adx:.0f} | Vol={vol_ratio:.1f}x | RSI={rsi:.0f}"
                ),
            }

        return None
