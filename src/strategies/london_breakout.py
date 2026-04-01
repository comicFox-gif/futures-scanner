"""
Forex Strategy 2 — London Session Breakout
--------------------------------------------
Classic institutional forex strategy. Detect the Asian session range, then
trade the breakout as London opens and institutional volume floods in.

How it works:
  Asian session (0:00–6:00 UTC): price consolidates in a tight range
  London open (7:00–12:00 UTC):  institutional flow breaks the range
  NY extension (13:00–16:00 UTC): NY open can confirm or re-test the break

Entry logic:
  LONG  → 15m candle closes clearly above Asian session high
  SHORT → 15m candle closes clearly below Asian session low

Filters:
  1. Session window    — only fires 7:00–12:00 UTC (or 13:00–16:00 for NY)
  2. Range size        — Asian range must be 15–80 pips (avoids noise & weekend gaps)
  3. Breakout buffer   — close must exceed range boundary by ATR * buffer (not a wick)
  4. Candle body ratio — must be a momentum candle, not a wick breakout
  5. RSI guard         — no extreme overbought/oversold (avoids exhaustion entries)
  6. One-per-day       — only fires once per pair per calendar day

Quality score (1–5 stars):
  + Candle body quality
  + Range size (20–50 pips is ideal)
  + RSI in neutral zone

Stage 1 WARNING is emitted when price is within 0.5 ATR of the range boundary
  (approaching but not yet broken).
"""

from __future__ import annotations
import logging
from datetime import datetime, date
import pandas as pd

from src.indicators import compute_all_indicators, candle_body_ratio

logger = logging.getLogger("forex_bot.london_breakout")


class LondonBreakoutStrategy:
    NAME = "London Breakout"

    def __init__(self, cfg: dict):
        s   = cfg["strategy"]
        sig = cfg["signal"]
        lb  = cfg.get("london_breakout", {})
        flt = cfg.get("filters", {})

        # Indicators (needed for enrich)
        self.ema_fast          = s["ema_fast"]
        self.ema_mid           = s["ema_mid"]
        self.ema_slow          = s["ema_slow"]
        self.ema_trend         = s["ema_trend"]
        self.macd_fast         = s["macd_fast"]
        self.macd_slow         = s["macd_slow"]
        self.macd_signal_p     = s["macd_signal"]
        self.rsi_period        = s["rsi_period"]
        self.atr_period        = s["atr_period"]
        self.volume_sma_period = s["volume_sma_period"]

        # Session windows (UTC hours)
        self.asian_start   = lb.get("asian_session_start_utc", 0)
        self.asian_end     = lb.get("asian_session_end_utc", 6)
        self.london_start  = lb.get("breakout_window_start_utc", 7)
        self.london_end    = lb.get("breakout_window_end_utc", 12)
        self.ny_start      = lb.get("ny_retest_start_utc", 13)
        self.ny_end        = lb.get("ny_retest_end_utc", 16)

        # Range requirements
        self.min_pips      = lb.get("min_range_pips", 15)
        self.max_pips      = lb.get("max_range_pips", 80)
        self.buf_atr       = lb.get("breakout_buffer_atr", 0.15)
        self.atr_sl_mult   = lb.get("atr_sl_multiplier", 1.2)

        # Signal levels
        self.tp1_rr        = sig["tp1_rr"]
        self.tp2_rr        = sig["tp2_rr"]
        self.tp3_rr        = sig["tp3_rr"]

        # Filters
        self.min_body      = flt.get("min_body_ratio", 0.40)

        # Daily dedup: pair -> date last fired
        self._fired: dict[str, date] = {}

    # ------------------------------------------------------------------
    # Indicator enrichment
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal_p,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_active_window(self) -> bool:
        h = datetime.utcnow().hour
        return (self.london_start <= h < self.london_end) or (self.ny_start <= h < self.ny_end)

    def _in_warning_window(self) -> bool:
        """One hour before London open — warn that breakout may be coming."""
        h = datetime.utcnow().hour
        return h == self.london_start - 1  # 6am UTC

    def _fired_today(self, pair: str) -> bool:
        return self._fired.get(pair) == datetime.utcnow().date()

    def _mark_fired(self, pair: str):
        self._fired[pair] = datetime.utcnow().date()

    def _to_pips(self, distance: float, pair: str) -> float:
        """Convert price distance to pips. JPY pairs: *100, others: *10000."""
        return distance * 100 if "JPY" in pair else distance * 10000

    def _get_asian_range(self, entry_df: pd.DataFrame) -> tuple[float | None, float | None]:
        """
        Find today's Asian session high/low from the 15m DataFrame.
        Requires at least 8 candles (2 hours of 15m data) in the session.
        """
        now       = datetime.utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        asian_end = now.replace(hour=self.asian_end, minute=0, second=0, microsecond=0)

        mask   = (entry_df.index >= day_start) & (entry_df.index < asian_end)
        session = entry_df[mask]

        if len(session) < 8:
            return None, None

        return float(session["high"].max()), float(session["low"].min())

    def _quality_score(self, body: float, range_pips: float, rsi: float) -> int:
        score = 1
        if body >= 0.65:
            score += 1
        elif body >= 0.50:
            score += 0.5
        # Ideal range: compact but meaningful
        if 20 <= range_pips <= 50:
            score += 1
        elif 15 <= range_pips <= 65:
            score += 0.5
        # RSI in neutral zone = not exhausted
        if 40 <= rsi <= 60:
            score += 1
        elif 35 <= rsi <= 65:
            score += 0.5
        return min(5, round(score))

    # ------------------------------------------------------------------
    # Main signal generator
    # ------------------------------------------------------------------

    def generate_signal(self, pair: str, entry_df: pd.DataFrame) -> dict | None:
        """
        Returns a signal dict or None.
        This strategy only uses the 15m entry_df (Asian range detection + breakout).
        """
        asian_high, asian_low = self._get_asian_range(entry_df)
        if asian_high is None:
            return None

        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        range_size = asian_high - asian_low
        range_pips = self._to_pips(range_size, pair)

        # Range quality gate
        if range_pips < self.min_pips or range_pips > self.max_pips:
            return None

        buffer = atr * self.buf_atr

        # ── Stage 1 WARNING: approaching range boundary ──────────────────
        if self._in_warning_window() and not self._fired_today(pair):
            warn_dist = atr * 0.5
            if abs(price - asian_high) < warn_dist:
                sl_dist = price - (asian_low - buffer)
                return {
                    "stage": 1, "direction": "long", "symbol": pair,
                    "entry": price,
                    "sl":    asian_low - buffer,
                    "tp1":   price + sl_dist * self.tp1_rr,
                    "tp2":   price + sl_dist * self.tp2_rr,
                    "tp3":   price + sl_dist * self.tp3_rr,
                    "rsi": rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                    "asian_high": asian_high, "asian_low": asian_low,
                    "range_pips": range_pips,
                    "reason": (
                        f"Approaching Asian high {asian_high:.5f} | "
                        f"Range {range_pips:.0f} pips | London opens soon"
                    ),
                }
            if abs(price - asian_low) < warn_dist:
                sl_dist = (asian_high + buffer) - price
                return {
                    "stage": 1, "direction": "short", "symbol": pair,
                    "entry": price,
                    "sl":    asian_high + buffer,
                    "tp1":   price - sl_dist * self.tp1_rr,
                    "tp2":   price - sl_dist * self.tp2_rr,
                    "tp3":   price - sl_dist * self.tp3_rr,
                    "rsi": rsi, "vol_ratio": 0, "quality": 2, "atr": atr,
                    "asian_high": asian_high, "asian_low": asian_low,
                    "range_pips": range_pips,
                    "reason": (
                        f"Approaching Asian low {asian_low:.5f} | "
                        f"Range {range_pips:.0f} pips | London opens soon"
                    ),
                }

        # ── Stage 2 CONFIRMED: breakout during active window ─────────────
        if not self._in_active_window() or self._fired_today(pair):
            return None

        body = candle_body_ratio(row)
        if body < self.min_body:
            return None

        # LONG: close above Asian high + buffer
        if price > asian_high + buffer:
            if rsi > 78:  # Overbought — exhaustion risk
                return None
            sl      = asian_low - buffer
            sl_dist = price - sl
            if sl_dist <= 0:
                return None
            quality = self._quality_score(body, range_pips, rsi)
            sig = {
                "stage": 2, "direction": "long", "symbol": pair,
                "entry": price,
                "sl":    sl,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                "asian_high": asian_high, "asian_low": asian_low,
                "range_pips": range_pips,
                "reason": (
                    f"London breakout LONG | Range {range_pips:.0f} pips | "
                    f"Break +{self._to_pips(price - asian_high, pair):.1f} pips | "
                    f"RSI {rsi:.1f}"
                ),
            }
            self._mark_fired(pair)
            return sig

        # SHORT: close below Asian low - buffer
        if price < asian_low - buffer:
            if rsi < 22:  # Oversold — exhaustion risk
                return None
            sl      = asian_high + buffer
            sl_dist = sl - price
            if sl_dist <= 0:
                return None
            quality = self._quality_score(body, range_pips, rsi)
            sig = {
                "stage": 2, "direction": "short", "symbol": pair,
                "entry": price,
                "sl":    sl,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": 0, "quality": quality, "atr": atr,
                "asian_high": asian_high, "asian_low": asian_low,
                "range_pips": range_pips,
                "reason": (
                    f"London breakout SHORT | Range {range_pips:.0f} pips | "
                    f"Break -{self._to_pips(asian_low - price, pair):.1f} pips | "
                    f"RSI {rsi:.1f}"
                ),
            }
            self._mark_fired(pair)
            return sig

        return None
