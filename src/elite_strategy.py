"""
Elite 4H Confluence Strategy
------------------------------
Fires on 4H Break of Structure (BOS) when confluence score >= min_score (default 6/10).

Entry gate (ALL required):
  1. Market regime aligned with trade direction (bull regime for longs, bear for shorts)
     — neutral regime allowed only if score >= 8
  2. 4H close breaks above last confirmed swing high (long) or below swing low (short)
  3. Previous candle was NOT already beyond the swing level (fresh break, not continuation)
  4. RSI in valid momentum zone: 40–75 (long), 25–60 (short)

Technical scoring (0–5):
  T1. Weekly candle bullish (long) or bearish (short) → 1pt
  T2. Volume >= 1.5× 20-period SMA on the signal candle → 1pt
  T3. MACD histogram positive (long) or negative (short) → 1pt
  T4. ADX >= 25 (market is trending, not ranging) → 1pt
  T5. Candle body >= 50% of high–low range (decisive close) → 1pt

Sentiment scoring (0–5) — each point awarded neutrally if data unavailable:
  S1. Fear & Greed Index in fear zone (≤45) for longs, greed zone (≥55) for shorts → 1pt
  S2. Funding rate negative for longs (longs underpaid → bullish contrarian) → 1pt
  S3. Open interest data present (proxy for trend participation) → 1pt
  S4. Long/short ratio not extreme against the trade (contrarian check) → 1pt
  S5. No large liquidation cluster opposing the trade (or data unavailable) → 1pt

SL/TP:
  SL: Beyond the signal candle's opposing wick + 20% ATR buffer
  TP1: 1.0× risk  |  TP2: 2.0× risk  |  TP3: 3.0× risk
"""
from __future__ import annotations
import logging
import pandas as pd

from src.indicators import compute_all_indicators, find_swing_highs_idx, find_swing_lows_idx
from src.sentiment import (
    get_fear_greed,
    get_funding_rate,
    get_open_interest,
    get_long_short_ratio,
    get_liquidation_pressure,
)

logger = logging.getLogger("futures_bot.elite")


class EliteStrategy:
    NAME = "Elite 4H BOS"

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

        elite = cfg.get("elite", {})
        self.swing_left  = elite.get("swing_left",  5)
        self.swing_right = elite.get("swing_right", 2)
        self.lookback    = elite.get("lookback",   40)
        self.vol_mult    = elite.get("volume_multiplier", 1.5)
        self.adx_min     = elite.get("adx_min", 25)
        self.min_body    = elite.get("min_body", 0.50)
        self.min_score   = elite.get("min_score", 6)

        self.tp1_rr      = sig["tp1_rr"]
        self.tp2_rr      = sig["tp2_rr"]
        self.tp3_rr      = sig["tp3_rr"]
        self.atr_sl_mult = sig.get("atr_sl_multiplier", 1.2)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    # ── Swing structure ───────────────────────────────────────────────────────

    def _last_swing_high(self, df: pd.DataFrame) -> float | None:
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        swings = find_swing_highs_idx(window, self.swing_left, self.swing_right)
        return swings[-1][1] if swings else None

    def _last_swing_low(self, df: pd.DataFrame) -> float | None:
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        swings = find_swing_lows_idx(window, self.swing_left, self.swing_right)
        return swings[-1][1] if swings else None

    # ── Technical scoring ─────────────────────────────────────────────────────

    def _score_technical(
        self,
        row: pd.Series,
        weekly_df: pd.DataFrame | None,
        direction: str,
    ) -> tuple[int, list[str]]:
        score = 0
        notes: list[str] = []

        # T1: Weekly candle direction
        if weekly_df is not None and len(weekly_df) >= 3:
            w_row   = weekly_df.iloc[-2]
            w_bull  = float(w_row["close"]) > float(w_row["open"])
            if direction == "long"  and w_bull:  score += 1; notes.append("W↑")
            elif direction == "short" and not w_bull: score += 1; notes.append("W↓")

        # T2: Volume spike
        vol     = float(row.get("volume",     0))
        vol_sma = float(row.get("volume_sma", 0))
        if vol_sma > 0 and vol >= vol_sma * self.vol_mult:
            score += 1
            notes.append(f"Vol{vol/vol_sma:.1f}x")

        # T3: MACD histogram
        hist = float(row.get("macd_hist", float("nan")))
        if not pd.isna(hist):
            if   direction == "long"  and hist > 0: score += 1; notes.append("MACD↑")
            elif direction == "short" and hist < 0: score += 1; notes.append("MACD↓")

        # T4: ADX trending
        adx = float(row.get("adx", 0))
        if adx >= self.adx_min:
            score += 1
            notes.append(f"ADX{adx:.0f}")

        # T5: Decisive candle body
        rng  = float(row["high"]) - float(row["low"])
        body = abs(float(row["close"]) - float(row["open"])) / rng if rng > 0 else 0
        if body >= self.min_body:
            score += 1
            notes.append(f"Body{body:.0%}")

        return score, notes

    # ── Sentiment scoring ─────────────────────────────────────────────────────

    def _score_sentiment(
        self,
        symbol: str,
        direction: str,
        exchange=None,
    ) -> tuple[int, list[str]]:
        score = 0
        notes: list[str] = []
        base  = symbol.split("/")[0]

        # S1: Fear & Greed
        fng     = get_fear_greed()
        fng_val = fng["value"]
        if   direction == "long"  and fng_val <= 45: score += 1; notes.append(f"F&G{fng_val}≤45")
        elif direction == "short" and fng_val >= 55: score += 1; notes.append(f"F&G{fng_val}≥55")
        else: notes.append(f"F&G{fng_val}✗")

        # S2: Funding rate — negative = market pays to hold shorts = bullish lean
        if exchange is not None:
            rate = get_funding_rate(exchange, symbol)
            if rate is None:
                score += 1; notes.append("FR-N/A")   # neutral award
            elif direction == "long"  and rate <= 0:
                score += 1; notes.append(f"FR{rate:.4f}")
            elif direction == "short" and rate > 0:
                score += 1; notes.append(f"FR+{rate:.4f}")
            else:
                notes.append(f"FR{rate:.4f}✗")
        else:
            score += 1; notes.append("FR-skip")

        # S3: Open interest present (award if data exists — proxy for participation)
        if exchange is not None:
            oi = get_open_interest(exchange, symbol)
            if oi is not None and oi > 0:
                score += 1; notes.append(f"OI{oi:.0f}")
            else:
                score += 1; notes.append("OI-N/A")   # neutral award
        else:
            score += 1; notes.append("OI-skip")

        # S4: Long/short ratio (Coinglass) — extreme positioning = contrarian signal
        ls = get_long_short_ratio(base)
        if ls is None:
            score += 1; notes.append("L/S-N/A")      # neutral award
        elif direction == "long":
            # Extreme long (>2.5) = everyone long = risky; reward when NOT extreme
            if ls <= 2.5: score += 1; notes.append(f"L/S{ls:.2f}")
            else:         notes.append(f"L/S{ls:.2f}✗")
        else:  # short
            # Extreme short (<0.4) = everyone short = risky; reward when NOT extreme
            if ls >= 0.4: score += 1; notes.append(f"L/S{ls:.2f}")
            else:         notes.append(f"L/S{ls:.2f}✗")

        # S5: Liquidation pressure (Coinglass)
        liq = get_liquidation_pressure(base)
        if liq is None:
            score += 1; notes.append("Liq-N/A")      # neutral award
        elif direction == "long":
            # Large short liquidations (sell_liq) = price rising, confirms longs
            if liq["sell_liq"] >= liq["buy_liq"]: score += 1; notes.append("Liq↑")
            else:                                   notes.append("Liq✗")
        else:
            # Large long liquidations (buy_liq) = price falling, confirms shorts
            if liq["buy_liq"] >= liq["sell_liq"]: score += 1; notes.append("Liq↓")
            else:                                   notes.append("Liq✗")

        return score, notes

    # ── Main signal generator ─────────────────────────────────────────────────

    def generate_signal(
        self,
        symbol:     str,
        weekly_df:  pd.DataFrame | None,
        h4_df:      pd.DataFrame,
        regime:     dict,
        exchange=None,
    ) -> dict | None:
        """
        Scan for a 4H BOS signal with confluence scoring.
        Returns signal dict or None.
        """
        min_len = self.lookback + self.swing_left + self.swing_right + 5
        if len(h4_df) < min_len:
            return None

        row   = h4_df.iloc[-2]
        prev  = h4_df.iloc[-3]
        price = float(row["close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        swing_high = self._last_swing_high(h4_df)
        swing_low  = self._last_swing_low(h4_df)
        if swing_high is None or swing_low is None:
            return None

        mregime  = regime.get("regime", "neutral")
        _sl_buf  = atr * 0.2

        # ── BULLISH BOS ───────────────────────────────────────────────────────
        if price > swing_high and float(prev["close"]) <= swing_high:
            if rsi < 40 or rsi > 75:
                logger.debug(f"[ELITE] {symbol} LONG RSI={rsi:.0f} out of range")
                return None
            # Block longs in confirmed bear regime (allow in neutral)
            if mregime == "bear":
                logger.debug(f"[ELITE] {symbol} LONG blocked — bear regime")
                return None

            sl_dist = max(price - (float(row["low"]) - _sl_buf), atr * 0.5)

            t_score, t_notes = self._score_technical(row, weekly_df, "long")
            s_score, s_notes = self._score_sentiment(symbol, "long", exchange)
            total = t_score + s_score

            # Neutral regime requires higher bar
            required = 8 if mregime == "neutral" else self.min_score
            if total < required:
                logger.debug(
                    f"[ELITE] {symbol} LONG score={total}/10 < {required} "
                    f"(regime={mregime}) — skipped"
                )
                return None

            vol_ratio = float(row.get("volume", 0)) / max(float(row.get("volume_sma", 1)), 1)
            return {
                "stage": 2, "direction": "long", "symbol": symbol,
                "entry": price,
                "sl":    price - sl_dist,
                "tp1":   price + sl_dist * self.tp1_rr,
                "tp2":   price + sl_dist * self.tp2_rr,
                "tp3":   price + sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": total, "atr": atr,
                "score": total, "tech_score": t_score, "sent_score": s_score,
                "reason": (
                    f"4H BOS ↑ {swing_high:.5g} | Score {total}/10 | "
                    f"Tech [{', '.join(t_notes)}] | "
                    f"Sent [{', '.join(s_notes)}] | "
                    f"RSI={rsi:.0f}"
                ),
            }

        # ── BEARISH BOS ───────────────────────────────────────────────────────
        if price < swing_low and float(prev["close"]) >= swing_low:
            if rsi > 60 or rsi < 25:
                logger.debug(f"[ELITE] {symbol} SHORT RSI={rsi:.0f} out of range")
                return None
            # Block shorts in confirmed bull regime
            if mregime == "bull":
                logger.debug(f"[ELITE] {symbol} SHORT blocked — bull regime")
                return None

            sl_dist = max((float(row["high"]) + _sl_buf) - price, atr * 0.5)

            t_score, t_notes = self._score_technical(row, weekly_df, "short")
            s_score, s_notes = self._score_sentiment(symbol, "short", exchange)
            total = t_score + s_score

            required = 8 if mregime == "neutral" else self.min_score
            if total < required:
                logger.debug(
                    f"[ELITE] {symbol} SHORT score={total}/10 < {required} "
                    f"(regime={mregime}) — skipped"
                )
                return None

            vol_ratio = float(row.get("volume", 0)) / max(float(row.get("volume_sma", 1)), 1)
            return {
                "stage": 2, "direction": "short", "symbol": symbol,
                "entry": price,
                "sl":    price + sl_dist,
                "tp1":   price - sl_dist * self.tp1_rr,
                "tp2":   price - sl_dist * self.tp2_rr,
                "tp3":   price - sl_dist * self.tp3_rr,
                "rsi": rsi, "vol_ratio": vol_ratio, "quality": total, "atr": atr,
                "score": total, "tech_score": t_score, "sent_score": s_score,
                "reason": (
                    f"4H BOS ↓ {swing_low:.5g} | Score {total}/10 | "
                    f"Tech [{', '.join(t_notes)}] | "
                    f"Sent [{', '.join(s_notes)}] | "
                    f"RSI={rsi:.0f}"
                ),
            }

        return None
