"""
Elite 4H Confluence Strategy  —  20-point scoring
---------------------------------------------------
Fires on 4H Break of Structure (BOS) when score >= min_score (default 5/20).

Entry gate (ALL required):
  1. Kill zone active — London 07:00-09:00 UTC OR NY 12:00-14:00 UTC only
  2. 4H candle CLOSED (signals only on confirmed closed candles, never open)
  3. Market regime clear — Bull=longs only, Bear=shorts only, Neutral=no signals
  4. 4H BOS: close breaks above last swing high (long) / below swing low (short)
  5. Previous candle was NOT already beyond the swing level (fresh break only)
  6. RSI in valid momentum zone: 40–75 (long), 25–60 (short)
  7. Minimum 2:1 RR available — no known 4H structure blocking the 2:1 target
  8. At least 1pt from EACH category: Wyckoff, Liquidity, MMM, VSA

--- Existing scoring (0–10) ---
Technical (0–5):
  T1. Weekly candle bullish/bearish              → 1pt
  T2. Volume ≥ 1.5× 20-period SMA               → 1pt
  T3. MACD histogram confirms direction          → 1pt
  T4. ADX ≥ 25                                   → 1pt
  T5. Candle body ≥ 50% of range                → 1pt

Sentiment (0–5):
  S1. Fear & Greed aligned                       → 1pt
  S2. Funding rate aligned                       → 1pt
  S3. Open interest present                      → 1pt
  S4. Long/short ratio not extreme               → 1pt
  S5. Liquidation pressure aligned               → 1pt

--- New advanced scoring (capped at 10 extra) ---
  Wyckoff Spring / Upthrust                      → 3pt
  Wyckoff SOS / SOW                              → 2pt
  EQH / EQL sweep                                → 2pt
  Stop Hunt detected                             → 2pt
  MMM Manipulation phase                         → 3pt
  VSA No Supply / No Demand                      → 2pt
  VSA Stopping Volume                            → 2pt
  VSA Effort/No Result (penalty)                → -1pt
  Intermarket 3+/5 aligned                       → 1pt
  Intermarket 5/5 aligned                        → 2pt
  Kill Zone active                               → 1pt
  AMD NY confirmation                            → 2pt

Maximum score: 20  |  Min to execute: 5/20

--- RR by score ---
  5–7   → 2:1 TP, $5 risk
  8–11  → 3:1 TP, $7 risk
  12–14 → 5:1 TP, $10 risk
  15–17 → 5:1 TP, $12 risk
  18–20 → 5:1 TP, $15 risk

Trailing stop activates at 3:1 profit, trails by 1 ATR on 4H.
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
from src.advanced_confluence import (
    get_kill_zone,
    detect_wyckoff,
    detect_liquidity,
    detect_mmm,
    detect_vsa,
    detect_delta_divergence,
    get_intermarket_score,
    risk_usdt_for_score,
    tp_rr_for_score,
)

logger = logging.getLogger("futures_bot.elite")

_MAX_SCORE   = 20
_ADV_CAP     = 10   # advanced points are capped before adding to base 10


class EliteStrategy:
    NAME = "Elite 4H BOS"

    def __init__(self, cfg: dict):
        s     = cfg["strategy"]
        elite = cfg.get("elite", {})

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

        self.swing_left  = elite.get("swing_left",  5)
        self.swing_right = elite.get("swing_right", 2)
        self.lookback    = elite.get("lookback",   40)
        self.vol_mult    = elite.get("volume_multiplier", 1.5)
        self.adx_min     = elite.get("adx_min", 25)
        self.min_body    = elite.get("min_body", 0.50)
        self.min_score   = elite.get("min_score", 5)    # out of 20
        self.min_rr      = elite.get("min_rr",   2.0)
        self.trail_rr    = elite.get("trail_rr", 3.0)   # activates at 3:1
        self.kill_zone_only = elite.get("kill_zone_only", True)

    # ── Indicators ────────────────────────────────────────────────────────────

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_indicators(
            df,
            self.ema_fast, self.ema_mid, self.ema_slow, self.ema_trend,
            self.macd_fast, self.macd_slow, self.macd_signal,
            self.rsi_period, self.atr_period, self.volume_sma_period,
        )

    # ── Swing structure helpers ───────────────────────────────────────────────

    def _last_swing_high(self, df: pd.DataFrame) -> float | None:
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        swings = find_swing_highs_idx(window, self.swing_left, self.swing_right)
        return swings[-1][1] if swings else None

    def _last_swing_low(self, df: pd.DataFrame) -> float | None:
        window = df.iloc[-(self.lookback + self.swing_right + 2):-2]
        swings = find_swing_lows_idx(window, self.swing_left, self.swing_right)
        return swings[-1][1] if swings else None

    # ── RR helpers ────────────────────────────────────────────────────────────

    def _check_rr_clearance(self, h4_df, entry, sl_dist, direction) -> bool:
        """Return True if no 4H structure blocks the minimum 2:1 target."""
        min_tp = (
            entry + self.min_rr * sl_dist if direction == "long"
            else entry - self.min_rr * sl_dist
        )
        window = h4_df.iloc[-(self.lookback + self.swing_right + 2):-2]

        if direction == "long":
            highs    = find_swing_highs_idx(window, self.swing_left, self.swing_right)
            blocking = [sh[1] for sh in highs if entry < sh[1] < min_tp]
        else:
            lows     = find_swing_lows_idx(window, self.swing_left, self.swing_right)
            blocking = [sl[1] for sl in lows if min_tp < sl[1] < entry]

        if blocking:
            lvl = blocking[-1]
            logger.debug(
                f"[ELITE] RR blocked at {lvl:.5g} "
                f"({abs(lvl - entry) / sl_dist:.1f}:1 before target)"
            )
            return False
        return True

    def _find_structural_tp(self, weekly_df, entry, sl_dist, direction,
                             required_rr: float = 2.0) -> tuple[float | None, str]:
        """
        Find nearest weekly swing level between required_rr and 6:1.
        Returns (price, reason_label) or (None, "").
        """
        if weekly_df is None or len(weekly_df) < 4:
            return None, ""

        if direction == "long":
            swings     = find_swing_highs_idx(weekly_df.iloc[:-2], 2, 1)
            candidates = sorted([sh[1] for sh in swings if sh[1] > entry])
        else:
            swings     = find_swing_lows_idx(weekly_df.iloc[:-2], 2, 1)
            candidates = sorted([sl[1] for sl in swings if sl[1] < entry], reverse=True)

        for lvl in candidates:
            rr = abs(lvl - entry) / sl_dist
            if required_rr <= rr <= 6.0:
                return lvl, f"Weekly level {lvl:.5g} ({rr:.1f}:1 RR)"

        return None, ""

    # ── Technical scoring (0–5, unchanged) ───────────────────────────────────

    def _score_technical(self, row, weekly_df, direction) -> tuple[int, list[str]]:
        score = 0
        notes: list[str] = []

        # T1: Weekly candle direction
        if weekly_df is not None and len(weekly_df) >= 3:
            w   = weekly_df.iloc[-2]
            bull = float(w["close"]) > float(w["open"])
            if   direction == "long"  and bull:      score += 1; notes.append("W↑")
            elif direction == "short" and not bull:  score += 1; notes.append("W↓")

        # T2: Volume spike
        vol     = float(row.get("volume",     0))
        vol_sma = float(row.get("volume_sma", 0))
        if vol_sma > 0 and vol >= vol_sma * self.vol_mult:
            score += 1; notes.append(f"Vol{vol / vol_sma:.1f}x")

        # T3: MACD histogram
        hist = float(row.get("macd_hist", float("nan")))
        if not pd.isna(hist):
            if   direction == "long"  and hist > 0: score += 1; notes.append("MACD↑")
            elif direction == "short" and hist < 0: score += 1; notes.append("MACD↓")

        # T4: ADX
        adx = float(row.get("adx", 0))
        if adx >= self.adx_min:
            score += 1; notes.append(f"ADX{adx:.0f}")

        # T5: Decisive candle body
        rng  = float(row["high"]) - float(row["low"])
        body = abs(float(row["close"]) - float(row["open"])) / rng if rng > 0 else 0
        if body >= self.min_body:
            score += 1; notes.append(f"Body{body:.0%}")

        return score, notes

    # ── Sentiment scoring (0–5, unchanged) ───────────────────────────────────

    def _score_sentiment(self, symbol, direction, exchange=None) -> tuple[int, list[str]]:
        score = 0
        notes: list[str] = []
        base  = symbol.split("/")[0]

        # S1: Fear & Greed
        fng = get_fear_greed()
        fv  = fng["value"]
        if   direction == "long"  and fv <= 45: score += 1; notes.append(f"F&G{fv}≤45")
        elif direction == "short" and fv >= 55: score += 1; notes.append(f"F&G{fv}≥55")
        else: notes.append(f"F&G{fv}✗")

        # S2: Funding rate
        if exchange is not None:
            rate = get_funding_rate(exchange, symbol)
            if rate is None:
                score += 1; notes.append("FR-N/A")
            elif direction == "long"  and rate <= 0: score += 1; notes.append(f"FR{rate:.4f}")
            elif direction == "short" and rate >  0: score += 1; notes.append(f"FR+{rate:.4f}")
            else: notes.append(f"FR{rate:.4f}✗")
        else:
            score += 1; notes.append("FR-skip")

        # S3: Open interest (proxy — award if data present)
        if exchange is not None:
            oi = get_open_interest(exchange, symbol)
            score += 1
            notes.append(f"OI{oi:.0f}" if oi else "OI-N/A")
        else:
            score += 1; notes.append("OI-skip")

        # S4: Long/short ratio
        ls = get_long_short_ratio(base)
        if ls is None:
            score += 1; notes.append("L/S-N/A")
        elif direction == "long"  and ls <= 2.5: score += 1; notes.append(f"L/S{ls:.2f}")
        elif direction == "short" and ls >= 0.4: score += 1; notes.append(f"L/S{ls:.2f}")
        else: notes.append(f"L/S{ls:.2f}✗")

        # S5: Liquidation pressure
        liq = get_liquidation_pressure(base)
        if liq is None:
            score += 1; notes.append("Liq-N/A")
        elif direction == "long"  and liq["sell_liq"] >= liq["buy_liq"]:
            score += 1; notes.append("Liq↑")
        elif direction == "short" and liq["buy_liq"]  >= liq["sell_liq"]:
            score += 1; notes.append("Liq↓")
        else: notes.append("Liq✗")

        return score, notes

    # ── Advanced scoring (new, capped at _ADV_CAP) ───────────────────────────

    def _score_advanced(
        self,
        h4_df: pd.DataFrame,
        weekly_df,
        symbol: str,
        direction: str,
        exchange=None,
        kz: dict | None = None,
    ) -> tuple[int, list[str], dict]:
        """
        Run all 8 advanced detectors and return (raw_score, note_lines, detail_dict).
        raw_score is NOT capped here — caller caps at _ADV_CAP.
        """
        score = 0
        notes: list[str] = []
        detail: dict = {}

        # ── Kill Zone ─────────────────────────────────────────────────────
        kz = kz or get_kill_zone()
        detail["kz"] = kz
        if kz["score"]:
            score += kz["score"]
            notes.append(kz["label"])
        # AMD NY session bonus
        if kz["amd_score"]:
            score += kz["amd_score"]
            notes.append(f"AMD: {kz['amd_phase'].title()} +{kz['amd_score']}")

        # ── Wyckoff ───────────────────────────────────────────────────────
        wyck = detect_wyckoff(h4_df)
        detail["wyckoff"] = wyck
        # Only award Wyckoff points when phase matches direction
        if wyck["score"] > 0:
            aligned = (
                (direction == "long"  and wyck["phase"] == "accumulation") or
                (direction == "short" and wyck["phase"] == "distribution")
            )
            if aligned:
                score += wyck["score"]
                notes.append(wyck["label"])
            else:
                notes.append(f"Wyckoff: {wyck['phase']} (direction mismatch)")
        elif wyck["event"]:
            notes.append(wyck["label"])

        # ── Liquidity (EQH/EQL + Stop Hunt) ──────────────────────────────
        liq = detect_liquidity(h4_df)
        detail["liquidity"] = liq
        # Award EQH sweep for shorts, EQL sweep for longs
        liq_score = 0
        if direction == "long"  and liq["eql_swept"]:   liq_score += 2
        if direction == "short" and liq["eqh_swept"]:   liq_score += 2
        if liq["stop_hunt"]:                             liq_score += 2
        if liq_score:
            score += liq_score
            for lbl in liq["label"]:
                notes.append(lbl)
        # "Clear target visible" (+1) — EQH above for longs, EQL below for shorts
        # Counts toward category gate even when nothing has been swept yet
        if liq_score == 0:
            price_now = float(h4_df.iloc[-2]["close"])
            if direction == "long":
                above = [lvl for lvl in liq["eqh_levels"] if lvl > price_now]
                if above:
                    liq_score += 1
                    score += 1
                    notes.append(f"Liquidity: Clear EQH target ${min(above):,.4g} +1")
            else:
                below = [lvl for lvl in liq["eql_levels"] if lvl < price_now]
                if below:
                    liq_score += 1
                    score += 1
                    notes.append(f"Liquidity: Clear EQL target ${max(below):,.4g} +1")
        # Store per-category score for gate checks
        detail["liquidity"]["liq_scored"] = liq_score
        # Informational: next liquidity target
        if direction == "long" and liq["eqh_levels"]:
            nearest = min(liq["eqh_levels"])
            if not any("EQH target" in n for n in notes):
                notes.append(f"Next EQH target: ${nearest:,.4g}")
        elif direction == "short" and liq["eql_levels"]:
            nearest = max(liq["eql_levels"])
            if not any("EQL target" in n for n in notes):
                notes.append(f"Next EQL target: ${nearest:,.4g}")

        # ── Market Maker Model ────────────────────────────────────────────
        mmm = detect_mmm(h4_df)
        detail["mmm"] = mmm
        if mmm["score"] > 0:
            mmm_aligned = (
                mmm.get("phase") == "consolidation" or    # direction-neutral, always award
                (direction == "long"  and mmm.get("direction") == "long") or
                (direction == "short" and mmm.get("direction") == "short")
            )
            if mmm_aligned:
                score += mmm["score"]
                notes.append(mmm["label"])
            else:
                notes.append(f"MMM: {mmm['label']} (mismatch)")

        # ── VSA ───────────────────────────────────────────────────────────
        vsa = detect_vsa(h4_df)
        detail["vsa"] = vsa
        if vsa["score"] != 0:
            vsa_aligned = (
                (direction == "long"  and vsa.get("bullish") is True) or
                (direction == "short" and vsa.get("bullish") is False)
            )
            if vsa_aligned or vsa["score"] < 0:   # penalties always apply
                score += vsa["score"]
                notes.append(vsa["label"])
            else:
                notes.append(f"VSA: {vsa['signal']} (direction mismatch)")

        # ── Delta Divergence ──────────────────────────────────────────────
        delta = detect_delta_divergence(h4_df)
        detail["delta"] = delta
        notes.append(delta["label"])
        # Divergence against trade direction is a soft warning (no penalty applied)

        # ── Intermarket ───────────────────────────────────────────────────
        try:
            im = get_intermarket_score(exchange=exchange, direction=direction)
        except Exception:
            im = {"score": 0, "label": "Intermarket: N/A"}
        detail["intermarket"] = im
        if im["score"]:
            score += im["score"]
        notes.append(im["label"])

        return score, notes, detail

    # ── Main signal generator ─────────────────────────────────────────────────

    def generate_signal(
        self,
        symbol:    str,
        weekly_df,
        h4_df:     pd.DataFrame,
        regime:    dict,
        exchange=None,
    ) -> dict | None:
        """
        Scan for a 4H BOS signal with 20-point confluence scoring.
        Returns a signal dict or None.
        """
        min_len = self.lookback + self.swing_left + self.swing_right + 5
        if len(h4_df) < min_len:
            return None

        # ── Kill Zone gate ────────────────────────────────────────────────
        kz = get_kill_zone()
        if self.kill_zone_only and not kz["active"]:
            logger.debug(f"[ELITE] {symbol} skipped — {kz['label']}")
            return None

        row   = h4_df.iloc[-2]
        prev  = h4_df.iloc[-3]
        price = float(row["close"])
        atr   = float(row.get("atr", float("nan")))
        rsi   = float(row.get("rsi", float("nan")))

        if pd.isna(atr) or atr == 0 or pd.isna(rsi):
            return None

        swing_high = self._last_swing_high(h4_df)
        swing_low  = self._last_swing_low(h4_df)
        if swing_high is None or swing_low is None:
            return None

        mregime = regime.get("regime", "neutral")
        _sl_buf = atr * 0.2

        for direction in ("long", "short"):
            # ── BOS check ─────────────────────────────────────────────────
            if direction == "long":
                bos = price > swing_high and float(prev["close"]) <= swing_high
                if not bos:
                    continue
                if rsi < 40 or rsi > 75:
                    continue
                # Neutral = no signals; Bear = no longs
                if mregime in ("bear", "neutral"):
                    continue
                sl_dist = max(price - (float(row["low"]) - _sl_buf), atr * 0.5)
                swing_ref = swing_high

            else:  # short
                bos = price < swing_low and float(prev["close"]) >= swing_low
                if not bos:
                    continue
                if rsi > 60 or rsi < 25:
                    continue
                # Neutral = no signals; Bull = no shorts
                if mregime in ("bull", "neutral"):
                    continue
                sl_dist = max((float(row["high"]) + _sl_buf) - price, atr * 0.5)
                swing_ref = swing_low

            # ── 2:1 RR clearance gate ─────────────────────────────────────
            if not self._check_rr_clearance(h4_df, price, sl_dist, direction):
                logger.debug(f"[ELITE] {symbol} {direction.upper()} — no 2:1 RR clearance")
                continue

            # ── Score existing factors ─────────────────────────────────────
            t_score, t_notes = self._score_technical(row, weekly_df, direction)
            s_score, s_notes = self._score_sentiment(symbol, direction, exchange)
            base_score = t_score + s_score   # 0–10

            # ── Score advanced factors ─────────────────────────────────────
            adv_raw, adv_notes, adv_detail = self._score_advanced(
                h4_df, weekly_df, symbol, direction, exchange, kz
            )
            adv_score = max(0, min(_ADV_CAP, adv_raw))   # cap at 10
            total     = min(_MAX_SCORE, base_score + adv_score)

            # ── Category gates — at least 1pt from each of 4 categories ──────
            wyck_d = adv_detail.get("wyckoff",   {})
            liq_d  = adv_detail.get("liquidity", {})
            mmm_d  = adv_detail.get("mmm",       {})
            vsa_d  = adv_detail.get("vsa",       {})
            kz_d   = adv_detail.get("kz",        {})

            # Wyckoff: spring/upthrust/sos/sow aligned with direction
            wyck_gate = (
                wyck_d.get("score", 0) > 0 and (
                    (direction == "long"  and wyck_d.get("phase") == "accumulation") or
                    (direction == "short" and wyck_d.get("phase") == "distribution")
                )
            )

            # Liquidity: EQL/EQH swept, stop hunt, OR clear target visible
            if direction == "long":
                liq_gate = bool(
                    liq_d.get("eql_swept") or
                    liq_d.get("stop_hunt") or
                    liq_d.get("liq_scored", 0) > 0
                )
            else:
                liq_gate = bool(
                    liq_d.get("eqh_swept") or
                    liq_d.get("stop_hunt") or
                    liq_d.get("liq_scored", 0) > 0
                )

            # MMM: manipulation aligned, OR consolidation, OR AMD NY (+2)
            mmm_manip_aligned = (
                mmm_d.get("phase") == "manipulation" and (
                    (direction == "long"  and mmm_d.get("direction") == "long") or
                    (direction == "short" and mmm_d.get("direction") == "short")
                )
            )
            mmm_gate = (
                mmm_manip_aligned or
                mmm_d.get("phase") == "consolidation" or
                kz_d.get("amd_score", 0) >= 2
            )

            # VSA: any positive aligned signal
            vsa_gate = (
                vsa_d.get("score", 0) > 0 and (
                    (direction == "long"  and vsa_d.get("bullish") is True) or
                    (direction == "short" and vsa_d.get("bullish") is False)
                )
            )

            if not (wyck_gate and liq_gate and mmm_gate and vsa_gate):
                logger.debug(
                    f"[ELITE] {symbol} {direction.upper()} — category gates failed: "
                    f"Wyckoff={wyck_gate} Liq={liq_gate} MMM={mmm_gate} VSA={vsa_gate}"
                )
                continue

            # ── Minimum score gate ─────────────────────────────────────────
            if total < self.min_score:
                logger.debug(
                    f"[ELITE] {symbol} {direction.upper()} score={total}/20 "
                    f"< {self.min_score} — skipped"
                )
                continue

            # ── Determine TP based on score ────────────────────────────────
            score_tp_rr = tp_rr_for_score(total)
            struct_tp, struct_label = self._find_structural_tp(
                weekly_df, price, sl_dist, direction, required_rr=score_tp_rr
            )
            if struct_tp:
                tp_price  = struct_tp
                tp_reason = struct_label
            else:
                tp_price  = (
                    price + score_tp_rr * sl_dist if direction == "long"
                    else price - score_tp_rr * sl_dist
                )
                tp_reason = f"Fixed {score_tp_rr:.0f}:1 RR"

            actual_rr      = abs(tp_price - price) / sl_dist
            trail_activate = (
                price + self.trail_rr * sl_dist if direction == "long"
                else price - self.trail_rr * sl_dist
            )
            risk_usdt = risk_usdt_for_score(total)

            sl_price  = price - sl_dist if direction == "long" else price + sl_dist
            vol_ratio = float(row.get("volume", 0)) / max(float(row.get("volume_sma", 1)), 1)

            # ── Build reason string ────────────────────────────────────────
            conf_label = (
                "ELITE 🏆" if total >= 15 else
                "Strong"   if total >= 12 else
                "Medium"   if total >= 8  else "Base"
            )
            kz_lbl    = adv_detail.get("kz", {}).get("label", "")
            wyck_lbl  = adv_detail.get("wyckoff", {}).get("label", "")
            mmm_lbl   = adv_detail.get("mmm", {}).get("label", "")
            vsa_lbl   = adv_detail.get("vsa", {}).get("label", "")
            im_lbl    = adv_detail.get("intermarket", {}).get("label", "")[:60]
            delta_lbl = adv_detail.get("delta", {}).get("label", "")

            reason = (
                f"4H BOS {'↑' if direction == 'long' else '↓'} {swing_ref:.5g} | "
                f"Score {total}/20 [{conf_label}] | "
                f"TP {tp_reason} | Trail @{self.trail_rr:.0f}:1 | "
                f"Tech [{', '.join(t_notes)}] | "
                f"Sent [{', '.join(s_notes)}] | "
                f"{kz_lbl} | {wyck_lbl} | {mmm_lbl} | {vsa_lbl} | "
                f"{delta_lbl} | {im_lbl} | "
                f"RSI={rsi:.0f}"
            )

            return {
                "stage": 2, "direction": direction, "symbol": symbol,
                "entry":   price,
                "sl":      sl_price,
                "tp1":     tp_price,
                "tp2":     tp_price,
                "tp3":     tp_price,
                "sl_dist": sl_dist,
                "tp_rr":   round(actual_rr, 2),
                "trail_activate": trail_activate,
                "risk_usdt":   risk_usdt,
                "rsi":         rsi,
                "vol_ratio":   vol_ratio,
                "atr":         atr,
                "quality":     total,
                "score":       total,
                "tech_score":  t_score,
                "sent_score":  s_score,
                "adv_score":   adv_score,
                "adv_detail":  adv_detail,
                "reason":      reason,
                "kz_label":    kz.get("label", ""),
                "wyck_label":  wyck_lbl,
                "mmm_label":   mmm_lbl,
                "vsa_label":   vsa_lbl,
                "im_label":    im_lbl,
            }

        return None
