"""
EMA Confluence Momentum Strategy — Signal Scanner
---------------------------------------------------
Two alert stages:

  Stage 1 WARNING  — Trend aligned on 1H, RSI & volume conditions met on 15m,
                     but MACD has NOT crossed yet. Get ready.

  Stage 2 CONFIRMED — All conditions met including MACD crossover.
                      Enter the trade manually.

Trend filter  : 1H — EMA 9 > 21 > 50, price > EMA 200 (bull) or inverse (bear)
Entry signal  : 15m — MACD crossover + RSI in range + volume surge
Levels shown  : SL = ATR * 1.5 | TP1 = 1R | TP2 = 2R | TP3 = 3R
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger("futures_bot.strategy")

from src.indicators import (
    compute_all_indicators,
    is_macd_bullish_cross,
    is_macd_bearish_cross,
    macd_histogram_turning_positive,
    macd_histogram_turning_negative,
    ema_recently_crossed_bullish,
    ema_recently_crossed_bearish,
    ema_cross_imminent_bullish,
    ema_cross_imminent_bearish,
    price_near_ema,
    price_bouncing_bullish,
    price_bouncing_bearish,
    candle_body_ratio,
    upper_wick_ratio,
    lower_wick_ratio,
    consecutive_bullish_closes,
    consecutive_bearish_closes,
    macd_histogram_strong,
    volume_building,
)


@dataclass
class Position:
    symbol: str
    direction: str          # "long" | "short"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    size: float             # units
    size_remaining: float
    margin_locked: float = 0.0  # capital reserved for this trade (returned on close)
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    be_activated: bool = False
    closed_pnl: float = 0.0


@dataclass
class Signal:
    stage: int              # 1 = warning, 2 = confirmed, 0 = none
    direction: str          # "long" | "short" | "none"
    symbol: str
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    atr: float
    rsi: float
    volume_ratio: float     # current volume / volume SMA
    reason: str = ""


class Strategy:
    def __init__(self, cfg: dict):
        s = cfg["strategy"]
        sig = cfg["signal"]

        self.ema_fast = s["ema_fast"]
        self.ema_mid = s["ema_mid"]
        self.ema_slow = s["ema_slow"]
        self.ema_trend = s["ema_trend"]
        self.macd_fast = s["macd_fast"]
        self.macd_slow = s["macd_slow"]
        self.macd_signal = s["macd_signal"]
        self.rsi_period = s["rsi_period"]
        self.rsi_long_min = s["rsi_long_min"]
        self.rsi_long_max = s["rsi_long_max"]
        self.rsi_short_min = s["rsi_short_min"]
        self.rsi_short_max = s["rsi_short_max"]
        self.atr_period = s["atr_period"]
        self.atr_sl_mult = sig["atr_sl_multiplier"]
        self.volume_sma_period = s["volume_sma_period"]
        self.volume_filter_mult = s["volume_filter_multiplier"]

        self.tp1_rr = sig["tp1_rr"]
        self.tp2_rr = sig["tp2_rr"]
        self.tp3_rr = sig["tp3_rr"]

        # Fake breakout filters
        flt = cfg.get("filters", {})
        self.min_body_ratio          = flt.get("min_body_ratio", 0.45)
        self.max_wick_against        = flt.get("max_wick_against_trade", 0.35)
        self.consecutive_closes      = flt.get("consecutive_closes", 2)
        self.macd_hist_atr_mult      = flt.get("macd_hist_atr_multiplier", 0.08)
        self.volume_building_candles = flt.get("volume_building_candles", 2)

        # Trend inception settings
        ti = cfg.get("trend_inception", {})
        self.ema_cross_lookback      = ti.get("ema_cross_max_candles_ago", 8)
        self.ema_cross_proximity_pct = ti.get("ema_cross_proximity_pct", 0.002)
        self.pullback_atr_tolerance  = ti.get("pullback_atr_tolerance", 0.6)

    # ------------------------------------------------------------------
    # Indicator enrichment
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast,
            self.ema_mid,
            self.ema_slow,
            self.ema_trend,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            self.rsi_period,
            self.atr_period,
            self.volume_sma_period,
        )

    # ------------------------------------------------------------------
    # Trend inception helpers (1H)
    # ------------------------------------------------------------------

    def _htf_long_inception(self, htf_df: pd.DataFrame) -> tuple[int, int, bool]:
        """
        Returns (crossed_recently, candles_ago, cross_imminent).
        - crossed_recently: EMA9 crossed above EMA21 within ema_cross_lookback candles
        - candles_ago: how many 1H candles ago the cross happened (0 = just now)
        - cross_imminent: EMA9 is approaching EMA21 from below, hasn't crossed yet
        Also requires price to be above EMA200 (macro context is bullish).
        """
        row = htf_df.iloc[-2]
        above_200 = row["close"] > row[f"ema_{self.ema_trend}"]

        crossed, candles_ago = ema_recently_crossed_bullish(
            htf_df,
            f"ema_{self.ema_fast}",
            f"ema_{self.ema_mid}",
            self.ema_cross_lookback,
        )
        imminent = ema_cross_imminent_bullish(
            htf_df,
            f"ema_{self.ema_fast}",
            f"ema_{self.ema_mid}",
            self.ema_cross_proximity_pct,
        )
        return (crossed and above_200), candles_ago, (imminent and above_200)

    def _htf_short_inception(self, htf_df: pd.DataFrame) -> tuple[int, int, bool]:
        """
        Returns (crossed_recently, candles_ago, cross_imminent).
        Requires price to be below EMA200.
        """
        row = htf_df.iloc[-2]
        below_200 = row["close"] < row[f"ema_{self.ema_trend}"]

        crossed, candles_ago = ema_recently_crossed_bearish(
            htf_df,
            f"ema_{self.ema_fast}",
            f"ema_{self.ema_mid}",
            self.ema_cross_lookback,
        )
        imminent = ema_cross_imminent_bearish(
            htf_df,
            f"ema_{self.ema_fast}",
            f"ema_{self.ema_mid}",
            self.ema_cross_proximity_pct,
        )
        return (crossed and below_200), candles_ago, (imminent and below_200)

    # ------------------------------------------------------------------
    # Stage 1 — WARNING: cross imminent on 1H, set up your levels
    # ------------------------------------------------------------------

    def _long_warning(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[bool, float, float]:
        """
        EMA9 is about to cross EMA21 on 1H (cross imminent).
        Price above EMA200. Alert the trader to watch for entry.
        """
        _, _, imminent = self._htf_long_inception(htf_df)
        if not imminent:
            return False, 0.0, 0.0

        row = entry_df.iloc[-2]
        rsi = row["rsi"]
        vol_ratio = row["volume"] / row["volume_sma"] if row["volume_sma"] > 0 else 0
        rsi_ok = self.rsi_long_min <= rsi <= self.rsi_long_max

        return rsi_ok, rsi, vol_ratio

    def _short_warning(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[bool, float, float]:
        """
        EMA9 is about to cross EMA21 bearishly on 1H.
        """
        _, _, imminent = self._htf_short_inception(htf_df)
        if not imminent:
            return False, 0.0, 0.0

        row = entry_df.iloc[-2]
        rsi = row["rsi"]
        vol_ratio = row["volume"] / row["volume_sma"] if row["volume_sma"] > 0 else 0
        rsi_ok = self.rsi_short_min <= rsi <= self.rsi_short_max

        return rsi_ok, rsi, vol_ratio

    # ------------------------------------------------------------------
    # Stage 2 — CONFIRMED: cross happened + pullback + bounce
    # ------------------------------------------------------------------

    def _long_confirmed_inception(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[bool, float, float, int]:
        """
        1H EMA9 recently crossed above EMA21 (trend just started).
        15m price pulled back to EMA9 or EMA21 and is bouncing up.
        All fake breakout filters pass.
        Returns (ok, rsi, vol_ratio, candles_ago_cross).
        """
        crossed, candles_ago, _ = self._htf_long_inception(htf_df)
        if not crossed:
            return False, 0.0, 0.0, -1

        row = entry_df.iloc[-2]
        rsi       = row["rsi"]
        vol_ratio = row["volume"] / row["volume_sma"] if row["volume_sma"] > 0 else 0

        # Price must have pulled back to EMA9 or EMA21 on 15m
        near_fast = price_near_ema(entry_df, f"ema_{self.ema_fast}", self.pullback_atr_tolerance)
        near_mid  = price_near_ema(entry_df, f"ema_{self.ema_mid}",  self.pullback_atr_tolerance)
        pullback  = near_fast or near_mid
        if not pullback:
            return False, rsi, vol_ratio, candles_ago

        # Price is bouncing from that EMA (touched and closed above)
        ema_col   = f"ema_{self.ema_fast}" if near_fast else f"ema_{self.ema_mid}"
        bouncing  = price_bouncing_bullish(entry_df, ema_col)
        if not bouncing:
            return False, rsi, vol_ratio, candles_ago

        # MACD and RSI confirmation
        macd_ok   = is_macd_bullish_cross(entry_df) or macd_histogram_turning_positive(entry_df)
        rsi_ok    = self.rsi_long_min <= rsi <= self.rsi_long_max

        if not (macd_ok and rsi_ok):
            return False, rsi, vol_ratio, candles_ago

        # Fake breakout filters
        passed, _ = self._fake_breakout_check(entry_df, "long")
        if not passed:
            return False, rsi, vol_ratio, candles_ago

        return True, rsi, vol_ratio, candles_ago

    def _short_confirmed_inception(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[bool, float, float, int]:
        """
        1H EMA9 recently crossed below EMA21 (downtrend just started).
        15m price pulled back up to EMA9 or EMA21 and is rejecting down.
        """
        crossed, candles_ago, _ = self._htf_short_inception(htf_df)
        if not crossed:
            return False, 0.0, 0.0, -1

        row = entry_df.iloc[-2]
        rsi       = row["rsi"]
        vol_ratio = row["volume"] / row["volume_sma"] if row["volume_sma"] > 0 else 0

        near_fast = price_near_ema(entry_df, f"ema_{self.ema_fast}", self.pullback_atr_tolerance)
        near_mid  = price_near_ema(entry_df, f"ema_{self.ema_mid}",  self.pullback_atr_tolerance)
        pullback  = near_fast or near_mid
        if not pullback:
            return False, rsi, vol_ratio, candles_ago

        ema_col  = f"ema_{self.ema_fast}" if near_fast else f"ema_{self.ema_mid}"
        bouncing = price_bouncing_bearish(entry_df, ema_col)
        if not bouncing:
            return False, rsi, vol_ratio, candles_ago

        macd_ok  = is_macd_bearish_cross(entry_df) or macd_histogram_turning_negative(entry_df)
        rsi_ok   = self.rsi_short_min <= rsi <= self.rsi_short_max

        if not (macd_ok and rsi_ok):
            return False, rsi, vol_ratio, candles_ago

        passed, _ = self._fake_breakout_check(entry_df, "short")
        if not passed:
            return False, rsi, vol_ratio, candles_ago

        return True, rsi, vol_ratio, candles_ago

    # ------------------------------------------------------------------
    # Fake breakout filter check — returns (passed, reason_if_failed)
    # ------------------------------------------------------------------

    def _fake_breakout_check(self, entry_df: pd.DataFrame, direction: str) -> tuple[bool, str]:
        """
        Runs all 5 fake breakout filters. Returns (True, "") if all pass,
        or (False, "reason") on the first failure.
        """
        # 1. Candle body — must be decisive, not wicky
        body = candle_body_ratio(entry_df)
        if body < self.min_body_ratio:
            return False, f"Weak candle body ({body:.0%} < {self.min_body_ratio:.0%}) — possible fakeout"

        # 2. Wick against trade direction — rejection wick = fake move
        if direction == "long":
            wick = upper_wick_ratio(entry_df)
            if wick > self.max_wick_against:
                return False, f"Upper wick too large ({wick:.0%}) — price rejected at highs"
        else:
            wick = lower_wick_ratio(entry_df)
            if wick > self.max_wick_against:
                return False, f"Lower wick too large ({wick:.0%}) — price rejected at lows"

        # 3. Consecutive closes — need follow-through, not a single candle spike
        if direction == "long":
            if not consecutive_bullish_closes(entry_df, self.consecutive_closes):
                return False, f"No {self.consecutive_closes} consecutive bullish closes — no follow-through"
        else:
            if not consecutive_bearish_closes(entry_df, self.consecutive_closes):
                return False, f"No {self.consecutive_closes} consecutive bearish closes — no follow-through"

        # 4. MACD histogram strength — must be meaningful, not a noise cross
        if not macd_histogram_strong(entry_df, self.macd_hist_atr_mult):
            return False, "MACD histogram too weak — noise cross, not real momentum"

        # 5. Volume building — sustained interest, not a single stop-hunt spike
        if not volume_building(entry_df, self.volume_building_candles):
            return False, f"Volume not sustained over {self.volume_building_candles} candles — possible stop hunt"

        return True, ""


    # ------------------------------------------------------------------
    # Build signal levels
    # ------------------------------------------------------------------

    def _build_signal(self, symbol: str, direction: str, stage: int, entry_price: float,
                      atr: float, rsi: float, vol_ratio: float, reason: str) -> Signal:
        sl_dist = atr * self.atr_sl_mult
        if direction == "long":
            sl   = entry_price - sl_dist
            tp1  = entry_price + sl_dist * self.tp1_rr
            tp2  = entry_price + sl_dist * self.tp2_rr
            tp3  = entry_price + sl_dist * self.tp3_rr
        else:
            sl   = entry_price + sl_dist
            tp1  = entry_price - sl_dist * self.tp1_rr
            tp2  = entry_price - sl_dist * self.tp2_rr
            tp3  = entry_price - sl_dist * self.tp3_rr

        return Signal(
            stage=stage,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            atr=atr,
            rsi=rsi,
            volume_ratio=vol_ratio,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Main signal generator
    # ------------------------------------------------------------------

    def _htf_long_aligned(self, htf_df: pd.DataFrame) -> bool:
        """1H EMA9 > EMA21 and price above EMA50 — bull trend. Relaxed from EMA9>21>50."""
        row = htf_df.iloc[-2]
        return (
            row[f"ema_{self.ema_fast}"] > row[f"ema_{self.ema_mid}"]
            and row["close"] > row[f"ema_{self.ema_slow}"]
        )

    def _htf_short_aligned(self, htf_df: pd.DataFrame) -> bool:
        """1H EMA9 < EMA21 and price below EMA50 — bear trend."""
        row = htf_df.iloc[-2]
        return (
            row[f"ema_{self.ema_fast}"] < row[f"ema_{self.ema_mid}"]
            and row["close"] < row[f"ema_{self.ema_slow}"]
        )

    def _score_long(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[int, float, float, list[str], list[str]]:
        """
        Score 5 conditions for a long signal.
        Returns (score, rsi, vol_ratio, passed_conditions, failed_conditions).
        Requires HTF trend aligned as a hard gate (score=0 if not).
        """
        if not self._htf_long_aligned(htf_df):
            return 0, 0.0, 0.0, [], ["HTF trend not aligned"]

        row       = entry_df.iloc[-2]
        rsi       = float(row["rsi"])
        vol_ratio = row["volume"] / row["volume_sma"] if row["volume_sma"] > 0 else 0

        passed, failed = [], []

        # C1: EMA pullback
        near_fast = price_near_ema(entry_df, f"ema_{self.ema_fast}", self.pullback_atr_tolerance)
        near_mid  = price_near_ema(entry_df, f"ema_{self.ema_mid}",  self.pullback_atr_tolerance)
        if near_fast or near_mid:
            passed.append("EMA pullback")
        else:
            failed.append("EMA pullback")

        # C2: Bounce off EMA
        ema_col  = f"ema_{self.ema_fast}" if near_fast else f"ema_{self.ema_mid}"
        bouncing = price_bouncing_bullish(entry_df, ema_col)
        if bouncing:
            passed.append("Bounce confirmed")
        else:
            failed.append("Bounce not confirmed")

        # C3: MACD confirmation
        macd_ok = is_macd_bullish_cross(entry_df) or macd_histogram_turning_positive(entry_df)
        if macd_ok:
            passed.append("MACD bullish")
        else:
            failed.append("MACD not confirmed")

        # C4: RSI in range
        if self.rsi_long_min <= rsi <= self.rsi_long_max:
            passed.append(f"RSI {rsi:.0f} ✓")
        else:
            failed.append(f"RSI {rsi:.0f} out of range")

        # C5: Candle filter (body + wick)
        fb_passed, _ = self._fake_breakout_check(entry_df, "long")
        if fb_passed:
            passed.append("Candle quality ✓")
        else:
            failed.append("Weak candle")

        return len(passed), rsi, vol_ratio, passed, failed

    def _score_short(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[int, float, float, list[str], list[str]]:
        """Score 5 conditions for a short signal."""
        if not self._htf_short_aligned(htf_df):
            return 0, 0.0, 0.0, [], ["HTF trend not aligned"]

        row       = entry_df.iloc[-2]
        rsi       = float(row["rsi"])
        vol_ratio = row["volume"] / row["volume_sma"] if row["volume_sma"] > 0 else 0

        passed, failed = [], []

        near_fast = price_near_ema(entry_df, f"ema_{self.ema_fast}", self.pullback_atr_tolerance)
        near_mid  = price_near_ema(entry_df, f"ema_{self.ema_mid}",  self.pullback_atr_tolerance)
        if near_fast or near_mid:
            passed.append("EMA pullback")
        else:
            failed.append("EMA pullback")

        ema_col  = f"ema_{self.ema_fast}" if near_fast else f"ema_{self.ema_mid}"
        bouncing = price_bouncing_bearish(entry_df, ema_col)
        if bouncing:
            passed.append("Rejection confirmed")
        else:
            failed.append("Rejection not confirmed")

        macd_ok = is_macd_bearish_cross(entry_df) or macd_histogram_turning_negative(entry_df)
        if macd_ok:
            passed.append("MACD bearish")
        else:
            failed.append("MACD not confirmed")

        if self.rsi_short_min <= rsi <= self.rsi_short_max:
            passed.append(f"RSI {rsi:.0f} ✓")
        else:
            failed.append(f"RSI {rsi:.0f} out of range")

        fb_passed, _ = self._fake_breakout_check(entry_df, "short")
        if fb_passed:
            passed.append("Candle quality ✓")
        else:
            failed.append("Weak candle")

        return len(passed), rsi, vol_ratio, passed, failed

    def _long_continuation(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[bool, float, float, str]:
        """Returns (fires, rsi, vol_ratio, reason). Fires if 3 or more of 5 conditions pass."""
        score, rsi, vol_ratio, passed, failed = self._score_long(htf_df, entry_df)
        if score >= 3:
            score_tag = f"✅ {score}/5 conditions"
            missing   = f" | Missing: {', '.join(failed)}" if failed else ""
            reason    = f"1H bull trend | {score_tag}{missing}"
            return True, rsi, vol_ratio, reason
        return False, rsi, vol_ratio, ""

    def _short_continuation(self, htf_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple[bool, float, float, str]:
        """Returns (fires, rsi, vol_ratio, reason). Fires if 3 or more of 5 conditions pass."""
        score, rsi, vol_ratio, passed, failed = self._score_short(htf_df, entry_df)
        if score >= 3:
            score_tag = f"✅ {score}/5 conditions"
            missing   = f" | Missing: {', '.join(failed)}" if failed else ""
            reason    = f"1H bear trend | {score_tag}{missing}"
            return True, rsi, vol_ratio, reason
        return False, rsi, vol_ratio, ""

    def generate_signal(
        self,
        symbol: str,
        htf_df: pd.DataFrame,
        entry_df: pd.DataFrame,
    ) -> Signal:
        row   = entry_df.iloc[-2]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        # --- LONG ---
        # Path A: fresh EMA cross (trend inception)
        confirmed, c_rsi, c_vol, candles_ago = self._long_confirmed_inception(htf_df, entry_df)
        if confirmed:
            return self._build_signal(
                symbol, "long", 2, price, atr, c_rsi, c_vol,
                f"1H EMA cross {candles_ago}h ago | Pullback+bounce confirmed",
            )

        # Path B: established trend + pullback (continuation, fires on 4/5 conditions)
        cont, c_rsi, c_vol, cont_reason = self._long_continuation(htf_df, entry_df)
        if cont:
            return self._build_signal(symbol, "long", 2, price, atr, c_rsi, c_vol, cont_reason)

        warn, w_rsi, w_vol = self._long_warning(htf_df, entry_df)
        if warn:
            return self._build_signal(
                symbol, "long", 1, price, atr, w_rsi, w_vol,
                "1H EMA cross imminent — watch for pullback entry",
            )

        # Continuation warning: 2+ of 4 conditions → fire warning
        if self._htf_long_aligned(htf_df):
            w_row  = entry_df.iloc[-2]
            w_rsi  = float(w_row["rsi"])
            near_fast = price_near_ema(entry_df, f"ema_{self.ema_fast}", self.pullback_atr_tolerance * 3)
            near_mid  = price_near_ema(entry_df, f"ema_{self.ema_mid}",  self.pullback_atr_tolerance * 3)
            w_conds = {
                "EMA approaching": near_fast or near_mid,
                f"RSI {w_rsi:.0f} in range": self.rsi_long_min <= w_rsi <= self.rsi_long_max,
                "MACD turning up": macd_histogram_turning_positive(entry_df),
                "Candle bullish": candle_body_ratio(entry_df) >= self.min_body_ratio,
            }
            w_passed = [k for k, v in w_conds.items() if v]
            w_failed = [k for k, v in w_conds.items() if not v]
            if len(w_passed) >= 2:
                missing = f" | Missing: {', '.join(w_failed)}" if w_failed else ""
                return self._build_signal(
                    symbol, "long", 1, price, atr, w_rsi, 0,
                    f"1H bull trend | ⚡ {len(w_passed)}/4 warning conditions{missing}",
                )

        # --- SHORT ---
        confirmed, c_rsi, c_vol, candles_ago = self._short_confirmed_inception(htf_df, entry_df)
        if confirmed:
            return self._build_signal(
                symbol, "short", 2, price, atr, c_rsi, c_vol,
                f"1H EMA cross {candles_ago}h ago | Pullback+rejection confirmed",
            )

        cont, c_rsi, c_vol, cont_reason = self._short_continuation(htf_df, entry_df)
        if cont:
            return self._build_signal(symbol, "short", 2, price, atr, c_rsi, c_vol, cont_reason)

        warn, w_rsi, w_vol = self._short_warning(htf_df, entry_df)
        if warn:
            return self._build_signal(
                symbol, "short", 1, price, atr, w_rsi, w_vol,
                "1H EMA cross imminent — watch for pullback entry",
            )

        # Continuation warning: 2+ of 4 conditions → fire warning
        if self._htf_short_aligned(htf_df):
            w_row  = entry_df.iloc[-2]
            w_rsi  = float(w_row["rsi"])
            near_fast = price_near_ema(entry_df, f"ema_{self.ema_fast}", self.pullback_atr_tolerance * 3)
            near_mid  = price_near_ema(entry_df, f"ema_{self.ema_mid}",  self.pullback_atr_tolerance * 3)
            w_conds = {
                "EMA approaching": near_fast or near_mid,
                f"RSI {w_rsi:.0f} in range": self.rsi_short_min <= w_rsi <= self.rsi_short_max,
                "MACD turning down": macd_histogram_turning_negative(entry_df),
                "Candle bearish": candle_body_ratio(entry_df) >= self.min_body_ratio,
            }
            w_passed = [k for k, v in w_conds.items() if v]
            w_failed = [k for k, v in w_conds.items() if not v]
            if len(w_passed) >= 2:
                missing = f" | Missing: {', '.join(w_failed)}" if w_failed else ""
                return self._build_signal(
                    symbol, "short", 1, price, atr, w_rsi, 0,
                    f"1H bear trend | ⚡ {len(w_passed)}/4 warning conditions{missing}",
                )

        return Signal(
            stage=0, direction="none", symbol=symbol,
            entry_price=price, stop_loss=0, tp1=0, tp2=0, tp3=0,
            atr=atr, rsi=rsi, volume_ratio=0, reason="No signal",
        )

    # ------------------------------------------------------------------
    # Paper trade position management
    # ------------------------------------------------------------------

    def check_position(self, pos: Position, current_price: float) -> list[dict]:
        """
        Check open paper position against current price.
        Returns list of action dicts to execute.
        """
        tp1_close  = 0.30
        tp2_close  = 0.30
        actions = []

        if pos.direction == "long":
            if current_price <= pos.stop_loss:
                actions.append({"action": "close_all", "reason": "SL hit"})
                return actions
            if not pos.tp3_hit and current_price >= pos.tp3:
                actions.append({"action": "close_all", "reason": "TP3 hit", "tp_level": 3})
                pos.tp3_hit = True
                return actions
            if not pos.tp2_hit and current_price >= pos.tp2:
                actions.append({"action": "close_partial", "pct": tp2_close, "reason": "TP2 hit", "tp_level": 2})
                actions.append({"action": "move_sl", "new_sl": pos.tp1, "reason": "Trail SL to TP1"})
                pos.tp2_hit = True
            elif not pos.tp1_hit and current_price >= pos.tp1:
                actions.append({"action": "close_partial", "pct": tp1_close, "reason": "TP1 hit", "tp_level": 1})
                actions.append({"action": "move_sl", "new_sl": pos.entry_price, "reason": "SL to Break-Even"})
                pos.tp1_hit = True
                pos.be_activated = True

        elif pos.direction == "short":
            if current_price >= pos.stop_loss:
                actions.append({"action": "close_all", "reason": "SL hit"})
                return actions
            if not pos.tp3_hit and current_price <= pos.tp3:
                actions.append({"action": "close_all", "reason": "TP3 hit", "tp_level": 3})
                pos.tp3_hit = True
                return actions
            if not pos.tp2_hit and current_price <= pos.tp2:
                actions.append({"action": "close_partial", "pct": tp2_close, "reason": "TP2 hit", "tp_level": 2})
                actions.append({"action": "move_sl", "new_sl": pos.tp1, "reason": "Trail SL to TP1"})
                pos.tp2_hit = True
            elif not pos.tp1_hit and current_price <= pos.tp1:
                actions.append({"action": "close_partial", "pct": tp1_close, "reason": "TP1 hit", "tp_level": 1})
                actions.append({"action": "move_sl", "new_sl": pos.entry_price, "reason": "SL to Break-Even"})
                pos.tp1_hit = True
                pos.be_activated = True

        return actions
