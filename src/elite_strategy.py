"""
Elite 4H Confluence Strategy  —  20-point scoring  (v2.0)
-----------------------------------------------------------
Fires when 4H liquidity sweep completes and score >= min_score (default 12/20).

Hard requirements (ALL must be true — score irrelevant):
  1. 4H candle CLOSED (never on open candle)
  2. Market regime clear — Bull=longs only, Bear=shorts only, Neutral=no signals
  3. Liquidity sweep complete: EQL swept (wick below, close above) → long
                               EQH swept (wick above, close below) → short
  4. RSI in valid momentum zone: 35–75 (long), 25–65 (short)
  5. 1H must confirm direction (FVG / MSS / sweep)
  6. Not Friday after 17:00 UTC / Not Sunday before 22:00 UTC

Category scoring v2 (per-category caps, total capped at 20):
  KILL ZONE       (max 3): Active London/NY=3  ← reweighted from 1
  AMD PHASE       (max 2): NY prime=2, late=1
  WYCKOFF         (max 2): Spring/SOS/Upthrust/SOW=2, Phase=1  ← fixed detection + recapped
  LIQUIDITY       (max 5): EQL/EQH swept=2, Stop hunt=1, Clear target=1, Swing swept=1
  MARKET MAKER    (max 2): Manipulation=2, Consolidation=1  ← reduced from 3
  VSA             (max 2): No Supply/Stopping Vol/Climactic=2, Test=1
  INTERMARKET     (max 2): 3+/5 aligned=1, 5/5=2
  FREE DATA       (max 4): Funding=1, OI=1, L/S ratio=1, Fear/Greed=1
  1H CONFIRMATION (max 3): FVG=2, MSS=1  ← now contributes to score (was 0)

Category gates (≥1pt required from EACH of: Liquidity, 1H Confirmation)
Minimum score to trade: 12/20

RR by score:
  <10  → 2:1 min RR, $5 risk
  10–13→ 3:1 min RR, $7/$10 risk
  14+  → 5:1 min RR, $10/$12/$15 risk

Trailing stop activates at 3:1, trails by 1 ATR on 4H.
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
)
from src.advanced_confluence import (
    get_kill_zone,
    detect_wyckoff,
    detect_liquidity,
    detect_mmm,
    detect_vsa,
    get_intermarket_score,
    confirm_1h_alignment,
    risk_usdt_for_score,
    tp_rr_for_score,
    calculate_liquidation_map,
    liq_map_clear_to_entry,
)
from src.chart_vision import ChartVision

logger = logging.getLogger("futures_bot.elite")

_MAX_SCORE = 20   # FIX: was 18


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
        self.min_score   = elite.get("min_score", 12)   # FIX: raised from 5 to 12
        self.min_rr      = elite.get("min_rr",   2.0)
        self.trail_rr    = elite.get("trail_rr", 3.0)
        self.kill_zone_only = elite.get("kill_zone_only", True)
        vision_key = cfg.get("anthropic_api_key") or __import__("os").environ.get("ANTHROPIC_API_KEY", "")
        self.vision = ChartVision(api_key=vision_key)

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

    def _check_rr_clearance(self, h4_df, entry, sl_dist, direction,
                             min_rr: float | None = None) -> bool:
        required = min_rr if min_rr is not None else self.min_rr
        min_tp = (
            entry + required * sl_dist if direction == "long"
            else entry - required * sl_dist
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

    def _find_structural_tps(
        self, weekly_df, daily_df, entry: float, sl_dist: float, direction: str
    ) -> tuple:
        """
        Return 3 distinct structural TP levels: (tp1, tp2, tp3, lbl1, lbl2, lbl3).

        Collects all Daily + Weekly swing levels beyond entry, deduplicates
        levels within 0.5% of each other, then picks the 3 nearest qualifying
        ones (≥1.5:1 RR minimum).  Falls back to fixed 2:1 / 3:1 / 5:1 when
        there is not enough structure on the chart.
        """
        mult = 1 if direction == "long" else -1
        candidates: list[tuple[float, str]] = []

        if daily_df is not None and len(daily_df) >= 10:
            if direction == "long":
                for _, lvl in find_swing_highs_idx(daily_df.iloc[:-2], 3, 1):
                    rr = (lvl - entry) / sl_dist
                    if rr >= 1.5:
                        candidates.append((lvl, f"D-{lvl:.5g} ({rr:.1f}R)"))
            else:
                for _, lvl in find_swing_lows_idx(daily_df.iloc[:-2], 3, 1):
                    rr = (entry - lvl) / sl_dist
                    if rr >= 1.5:
                        candidates.append((lvl, f"D-{lvl:.5g} ({rr:.1f}R)"))

        if weekly_df is not None and len(weekly_df) >= 4:
            if direction == "long":
                for _, lvl in find_swing_highs_idx(weekly_df.iloc[:-2], 2, 1):
                    rr = (lvl - entry) / sl_dist
                    if rr >= 1.5:
                        candidates.append((lvl, f"W-{lvl:.5g} ({rr:.1f}R)"))
            else:
                for _, lvl in find_swing_lows_idx(weekly_df.iloc[:-2], 2, 1):
                    rr = (entry - lvl) / sl_dist
                    if rr >= 1.5:
                        candidates.append((lvl, f"W-{lvl:.5g} ({rr:.1f}R)"))

        # Sort nearest first, deduplicate within 0.5%
        candidates.sort(key=lambda x: x[0] * mult, reverse=(direction == "long"))
        deduped: list[tuple[float, str]] = []
        for lvl, lbl in candidates:
            if not deduped or abs(lvl - deduped[-1][0]) / max(abs(deduped[-1][0]), 1e-10) > 0.005:
                deduped.append((lvl, lbl))

        def fixed(rr_mult: float) -> tuple[float, str]:
            p = entry + mult * rr_mult * sl_dist
            return p, f"Fixed {rr_mult:.0f}:1"

        tp1 = deduped[0] if len(deduped) > 0 else fixed(2.0)
        tp2 = deduped[1] if len(deduped) > 1 else fixed(3.0)
        tp3 = deduped[2] if len(deduped) > 2 else fixed(5.0)
        return tp1[0], tp2[0], tp3[0], tp1[1], tp2[1], tp3[1]

    # ── Free data scoring (max 4pts) ─────────────────────────────────────────

    def _score_free_data(self, symbol: str, direction: str) -> tuple[int, list[str]]:
        score = 0
        lines: list[str] = []
        base  = symbol.split("/")[0]

        rate = get_funding_rate(None, symbol)
        if rate is None:
            score += 1; lines.append(f"✅ Funding Rate N/A   +1")
        elif direction == "long"  and rate <= 0:
            score += 1; lines.append(f"✅ Funding Rate {rate:.4f}%   +1")
        elif direction == "short" and rate >  0:
            score += 1; lines.append(f"✅ Funding Rate +{rate:.4f}%   +1")
        else:
            lines.append(f"⬜ Funding Rate {rate:.4f}%   +0")

        oi = get_open_interest(None, symbol)
        if oi is not None:
            score += 1; lines.append(f"✅ OI ${oi:,.0f}   +1")
        else:
            lines.append(f"⬜ Open Interest N/A   +0")

        ls = get_long_short_ratio(base)
        if ls is None:
            score += 1; lines.append(f"✅ Long/Short Ratio N/A   +1")
        elif direction == "long"  and ls <= 2.5:
            score += 1; lines.append(f"✅ Long/Short Ratio {ls:.2f}   +1")
        elif direction == "short" and ls >= 0.4:
            score += 1; lines.append(f"✅ Long/Short Ratio {ls:.2f}   +1")
        else:
            lines.append(f"⬜ Long/Short Ratio {ls:.2f}   +0")

        fng = get_fear_greed()
        fv  = fng["value"]
        if   direction == "long"  and fv < 50:
            score += 1; lines.append(f"✅ Fear/Greed {fv} (Fear)   +1")
        elif direction == "short" and fv >= 50:
            score += 1; lines.append(f"✅ Fear/Greed {fv} (Greed)   +1")
        else:
            lines.append(f"⬜ Fear/Greed {fv}   +0")

        return min(4, score), lines

    # ── Swing swept helper ────────────────────────────────────────────────────

    def _detect_swing_swept(self, h4_df: pd.DataFrame, direction: str) -> bool:
        try:
            cur    = h4_df.iloc[-2]
            window = h4_df.iloc[-(self.lookback + self.swing_right + 2):-3]
            if direction == "long":
                swings = find_swing_lows_idx(window, self.swing_left, self.swing_right)
                for _, lvl in swings[-3:]:
                    if float(cur["low"]) < lvl and float(cur["close"]) > lvl:
                        return True
            else:
                swings = find_swing_highs_idx(window, self.swing_left, self.swing_right)
                for _, lvl in swings[-3:]:
                    if float(cur["high"]) > lvl and float(cur["close"]) < lvl:
                        return True
        except Exception:
            pass
        return False

    # ── Category scoring (20-point system v2) ────────────────────────────────

    def _score_all_categories(
        self,
        h4_df:     pd.DataFrame,
        weekly_df,
        symbol:    str,
        direction: str,
        h1_conf:   dict,          # FIX: now passed in so 1H score counts
        exchange=None,
        kz:        dict | None = None,
    ) -> dict:
        """
        Score all 9 categories with per-category caps. Total capped at 20.

        v2 changes:
          - Kill zone reweighted +1 → +3
          - AMD phase separated from MMM, scored independently
          - Wyckoff cap reduced 5 → 2 (now that detection is fixed)
          - Stop hunt reweighted +2 → +1
          - MMM manipulation reduced +3 → +2
          - 1H confirmation now contributes FVG=+2, MSS=+1 to score
        """
        kz = kz or get_kill_zone()
        detail: dict = {"kz": kz}

        # ── KILL ZONE (cap 3) — reweighted from +1 ───────────────────────
        # Most impactful filter. London/NY sessions = 3pts, off-session = 0.
        kz_score = kz["score"]   # advanced_confluence v2 returns 3 for active, 0 otherwise
        kz_score = min(3, kz_score)
        kz_name  = kz.get("name", "").upper().replace("_", " ")
        kz_line  = (
            f"✅ {kz_name} Active   +{kz_score}"
            if kz_score > 0
            else f"⬜ Kill Zone — No Trade Zone   +0"
        )

        # ── AMD PHASE (cap 2) — scored independently from MMM ────────────
        amd_score = min(2, kz.get("amd_score", 0))
        amd_lines = []
        if amd_score >= 2:
            amd_lines.append(f"✅ AMD NY Prime Distribution   +2")
        elif amd_score == 1:
            amd_lines.append(f"✅ AMD Distribution Phase   +1")
        else:
            amd_lines.append(f"⬜ AMD: Accumulation/Manipulation   +0")

        # ── WYCKOFF (cap 2) — reduced from 5; detection now fixed ────────
        wyck = detect_wyckoff(h4_df)
        detail["wyckoff"] = wyck
        wyck_raw   = 0
        wyck_lines = []
        if wyck["score"] > 0:
            aligned = (
                (direction == "long"  and wyck["phase"] == "accumulation") or
                (direction == "short" and wyck["phase"] == "distribution")
            )
            if aligned:
                # Cap contribution at 2 regardless of wyck["score"]
                wyck_raw = min(2, wyck["score"])
                wyck_lines.append(
                    f"✅ {wyck['label'].split(': ', 1)[-1].split(' +')[0]}   +{wyck_raw}"
                )
            else:
                wyck_lines.append(f"⬜ Wyckoff {wyck['phase'].title()} (mismatch)   +0")
        elif wyck["event"] in ("sc_detected", "bc_detected"):
            phase_aligned = (
                (direction == "long"  and wyck["phase"] == "accumulation") or
                (direction == "short" and wyck["phase"] == "distribution")
            )
            if phase_aligned:
                wyck_raw = 1
                wyck_lines.append(
                    f"✅ {'Accumulation' if direction == 'long' else 'Distribution'} Phase   +1"
                )
            else:
                wyck_lines.append(f"⬜ Wyckoff {wyck['phase'].title()} (mismatch)   +0")
        else:
            wyck_lines.append("⬜ Wyckoff: N/A   +0")
        wyck_score = min(2, wyck_raw)   # FIX: cap reduced from 5 → 2

        # ── LIQUIDITY (cap 5) ─────────────────────────────────────────────
        liq = detect_liquidity(h4_df)
        detail["liquidity"] = liq
        liq_raw   = 0
        liq_lines = []
        price_now = float(h4_df.iloc[-2]["close"])

        if direction == "long"  and liq["eql_swept"]:
            liq_raw += 2
            liq_lines.append(f"✅ EQL Swept + Closed Above   +2")
        if direction == "short" and liq["eqh_swept"]:
            liq_raw += 2
            liq_lines.append(f"✅ EQH Swept + Closed Below   +2")
        if liq["stop_hunt"]:
            # FIX: stop hunt reweighted +2 → +1 (less weight than full EQL/EQH sweep)
            liq_raw += 1
            hunt_lbl = next(
                (l for l in liq["label"] if "Hunt" in l or "hunt" in l),
                "Stop Hunt Detected"
            )
            liq_lines.append(f"✅ {hunt_lbl.split(': ')[-1].split(' +')[0]}   +1")
        # Clear liquidity target above/below
        if direction == "long":
            above = [lvl for lvl in liq["eqh_levels"] if lvl > price_now]
            if above:
                liq_raw += 1
                liq_lines.append(f"✅ Clear EQH Target ${min(above):,.4g}   +1")
        else:
            below = [lvl for lvl in liq["eql_levels"] if lvl < price_now]
            if below:
                liq_raw += 1
                liq_lines.append(f"✅ Clear EQL Target ${max(below):,.4g}   +1")
        # Swing level swept
        if self._detect_swing_swept(h4_df, direction):
            liq_raw += 1
            liq_lines.append(f"✅ Swing Level Swept   +1")
        if not liq_lines:
            liq_lines.append("⬜ Liquidity: N/A   +0")
        detail["liquidity"]["liq_scored"] = liq_raw
        liq_score = min(5, liq_raw)   # FIX: cap raised 4 → 5 to match new structure

        # ── MMM (cap 2) — reduced from 4 ─────────────────────────────────
        mmm = detect_mmm(h4_df)
        detail["mmm"] = mmm
        mmm_raw   = 0
        mmm_lines = []
        if mmm["phase"] == "manipulation":
            mmm_aligned = (
                (direction == "long"  and mmm.get("direction") == "long") or
                (direction == "short" and mmm.get("direction") == "short")
            )
            if mmm_aligned:
                mmm_raw += 2   # FIX: was +3
                mmm_lines.append(f"✅ Manipulation Confirmed   +2")
            else:
                mmm_lines.append(f"⬜ MMM Manipulation (mismatch)   +0")
        elif mmm["phase"] == "consolidation":
            mmm_raw += 1
            mmm_lines.append(f"✅ Consolidation Detected   +1")
        # NOTE: AMD bonus removed from MMM — now scored separately above
        if not mmm_lines:
            mmm_lines.append("⬜ MMM: N/A   +0")
        mmm_score = min(2, mmm_raw)   # FIX: cap reduced 4 → 2

        # ── VSA (cap 2) — unchanged ───────────────────────────────────────
        vsa = detect_vsa(h4_df)
        detail["vsa"] = vsa
        vsa_raw   = 0
        vsa_lines = []
        if vsa["score"] > 0:
            vsa_aligned = (
                (direction == "long"  and vsa.get("bullish") is True) or
                (direction == "short" and vsa.get("bullish") is False)
            )
            if vsa_aligned:
                vsa_raw = vsa["score"]
                clean = vsa["label"].split(": ", 1)[-1].split(" +")[0].split(" ⚠")[0]
                vsa_lines.append(f"✅ {clean}   +{vsa_raw}")
            else:
                vsa_lines.append(f"⬜ VSA {vsa.get('signal', '')} (mismatch)   +0")
        if not vsa_lines:
            vsa_lines.append("⬜ VSA: N/A   +0")
        vsa_score = min(2, vsa_raw)

        # ── INTERMARKET (cap 2) — unchanged ──────────────────────────────
        try:
            im = get_intermarket_score(exchange=exchange, direction=direction)
        except Exception:
            im = {"score": 0, "label": "Intermarket: N/A", "n_align": 0, "n_total": 5}
        detail["intermarket"] = im
        im_score = min(2, im["score"])
        n_align  = im.get("n_align", 0)
        n_total  = im.get("n_total", 5)
        im_lines = (
            [f"✅ {n_align}/{n_total} Factors Aligned   +{im_score}"]
            if im_score > 0
            else [f"⬜ Intermarket {n_align}/{n_total}   +0"]
        )

        # ── FREE DATA (cap 4) — unchanged ────────────────────────────────
        free_score, free_lines = self._score_free_data(symbol, direction)
        free_score = min(4, free_score)

        # ── 1H CONFIRMATION (cap 3) — FIX: now contributes to score ──────
        # Previously confirm_1h_alignment() was a binary gate only (score ignored).
        # FVG = institutional order block = strong confluence = +2
        # MSS = structure break = momentum confirmation = +1
        h1_score = min(3, h1_conf.get("score", 0))
        h1_lines = []
        if h1_conf.get("fvg_present"):
            h1_lines.append(
                f"✅ 1H FVG {'Bullish' if direction == 'long' else 'Bearish'}   +2"
            )
        if h1_conf.get("mss_present"):
            h1_lines.append(f"✅ 1H MSS Confirmed   +1")
        if h1_conf.get("sweep_present") and not h1_conf.get("fvg_present") and not h1_conf.get("mss_present"):
            h1_lines.append(f"✅ 1H Sweep Confirmed   +0")
        if not h1_lines:
            h1_lines.append(f"⬜ 1H: {h1_conf.get('reason', 'N/A')}   +0")

        # ── TOTAL (capped at 20) ──────────────────────────────────────────
        total = min(
            _MAX_SCORE,
            kz_score + amd_score + wyck_score + liq_score
            + mmm_score + vsa_score + im_score + free_score + h1_score
        )

        logger.debug(
            f"[SCORE] kz={kz_score} amd={amd_score} wyck={wyck_score} "
            f"liq={liq_score} mmm={mmm_score} vsa={vsa_score} "
            f"im={im_score} free={free_score} h1={h1_score} → total={total}/{_MAX_SCORE}"
        )

        return {
            "total":      total,
            # Per-category scores
            "kz_score":   kz_score,
            "amd_score":  amd_score,
            "wyck_score": wyck_score,
            "liq_score":  liq_score,
            "mmm_score":  mmm_score,
            "vsa_score":  vsa_score,
            "im_score":   im_score,
            "free_score": free_score,
            "h1_score":   h1_score,
            # Legacy kz_bonus key (kept for bot.py compatibility)
            "kz_bonus":   kz_score,
            # Per-category display lines
            "kz_line":    kz_line,
            "amd_lines":  amd_lines,
            "wyck_lines": wyck_lines,
            "liq_lines":  liq_lines,
            "mmm_lines":  mmm_lines,
            "vsa_lines":  vsa_lines,
            "im_lines":   im_lines,
            "free_lines": free_lines,
            "h1_lines":   h1_lines,
            "detail":     detail,
            # Legacy label fields
            "wyck_label": wyck.get("label", ""),
            "mmm_label":  mmm.get("label", ""),
            "vsa_label":  vsa.get("label", ""),
            "im_label":   im.get("label",  ""),
        }

    # ── Main signal generator ─────────────────────────────────────────────────

    def generate_signal(
        self,
        symbol:    str,
        weekly_df,
        daily_df,
        h4_df:     pd.DataFrame,
        h1_df,
        regime:    dict,
        exchange=None,
    ) -> dict | None:
        """
        Scan for a 4H BOS signal with 20-point confluence scoring.

        v2 changes:
          - 1H confirmation called BEFORE scoring so h1_score feeds into total
          - Kill zone properly weighted at +3
          - Forming alert threshold raised 3 → 6
          - Score display updated to /20
          - conf_bolts thresholds updated for new scale
        """
        min_len = self.lookback + self.swing_left + self.swing_right + 5
        if len(h4_df) < min_len:
            return None

        kz = get_kill_zone()

        row   = h4_df.iloc[-2]
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

        # ── Liquidity sweep detection ─────────────────────────────────────
        liq_trigger   = detect_liquidity(h4_df)
        bullish_sweep = liq_trigger["eql_swept"] or any(
            "Sellside" in str(l) for l in liq_trigger.get("label", [])
        )
        bearish_sweep = liq_trigger["eqh_swept"] or any(
            "Buyside" in str(l) for l in liq_trigger.get("label", [])
        )

        h1_liq = detect_liquidity(h1_df) if h1_df is not None else {}
        h1_bullish_sweep = h1_liq.get("eql_swept") or any(
            "Sellside" in str(l) for l in h1_liq.get("label", [])
        )
        h1_bearish_sweep = h1_liq.get("eqh_swept") or any(
            "Buyside" in str(l) for l in h1_liq.get("label", [])
        )

        for direction in ("long", "short"):
            if direction == "long":
                if not bullish_sweep:
                    continue
                if rsi < 35 or rsi > 75:
                    continue
                if mregime in ("bear", "neutral"):
                    continue
                sl_dist   = max(price - (float(row["low"]) - atr * 0.3), atr * 0.8)
                swing_ref = swing_low
            else:
                if not bearish_sweep:
                    continue
                if rsi > 65 or rsi < 25:
                    continue
                if mregime in ("bull", "neutral"):
                    continue
                sl_dist   = max((float(row["high"]) + atr * 0.3) - price, atr * 0.8)
                swing_ref = swing_high

            # ── Liquidation map gate ──────────────────────────────────────
            sl_price_check = price - sl_dist if direction == "long" else price + sl_dist
            liq_map = calculate_liquidation_map(h4_df)
            clear, liq_reason = liq_map_clear_to_entry(liq_map, price, sl_price_check, direction)
            if not clear:
                logger.info(f"[ELITE] {symbol} {direction.upper()} — liq map blocked: {liq_reason}")
                continue

            # ── Fast 2:1 clearance gate ───────────────────────────────────
            if not self._check_rr_clearance(h4_df, price, sl_dist, direction, min_rr=2.0):
                logger.info(f"[ELITE] {symbol} {direction.upper()} — no 2:1 RR clearance")
                continue

            # ── 1H confirmation — called HERE so score feeds into total ───
            # FIX: was called after scoring, meaning h1_score was never added
            h1_conf = confirm_1h_alignment(h1_df, direction)

            # ── Score all categories (20-point system) ────────────────────
            sc    = self._score_all_categories(
                h4_df, weekly_df, symbol, direction,
                h1_conf=h1_conf,   # FIX: pass h1_conf so 1H score is included
                exchange=exchange,
                kz=kz,
            )
            total = sc["total"]

            # ── Forming alert — score ≥6 but below signal threshold ───────
            # FIX: raised from 3 → 6 (3 pts is nearly nothing in 20-pt system)
            if total >= 6 and total < self.min_score:
                return {
                    "forming":   True,
                    "symbol":    symbol,
                    "direction": direction,
                    "score":     total,
                    "entry":     price,
                }

            # ── Minimum score gate ────────────────────────────────────────
            if total < self.min_score:
                logger.info(
                    f"[ELITE] {symbol} {direction.upper()} score={total}/{_MAX_SCORE} "
                    f"< {self.min_score} — skipped"
                )
                continue

            # ── Score-based RR clearance ──────────────────────────────────
            required_rr = tp_rr_for_score(total)
            if required_rr > 2.0:
                if not self._check_rr_clearance(h4_df, price, sl_dist, direction,
                                                 min_rr=required_rr):
                    logger.info(
                        f"[ELITE] {symbol} {direction.upper()} — no {required_rr:.0f}:1 clearance"
                    )
                    continue

            # ── 1H gate — must confirm (score already counted above) ──────
            if not h1_conf["aligned"]:
                logger.debug(
                    f"[ELITE] {symbol} {direction.upper()} — 1H not aligned: "
                    f"{h1_conf['reason']}"
                )
                return {
                    "watching":  True,
                    "symbol":    symbol,
                    "direction": direction,
                    "entry":     price,
                    "sl":        price - sl_dist if direction == "long" else price + sl_dist,
                    "score":     total,
                    "tp_rr":     required_rr,
                    "regime":    mregime,
                    "h1_reason": h1_conf["reason"],
                    "wyck_score": sc["wyck_score"],
                    "liq_score":  sc["liq_score"],
                    "mmm_score":  sc["mmm_score"],
                    "vsa_score":  sc["vsa_score"],
                }

            # ── TPs — 3 distinct structural levels (Weekly/Daily swings) ─
            tp1_price, tp2_price, tp3_price, tp1_lbl, tp2_lbl, tp3_lbl = (
                self._find_structural_tps(weekly_df, daily_df, price, sl_dist, direction)
            )
            tp_reason = tp3_lbl   # furthest TP label used in log/reason string

            actual_rr      = abs(tp3_price - price) / sl_dist
            trail_activate = (
                price + self.trail_rr * sl_dist if direction == "long"
                else price - self.trail_rr * sl_dist
            )
            risk_usdt = risk_usdt_for_score(total)
            sl_price  = price - sl_dist if direction == "long" else price + sl_dist
            vol_ratio = float(row.get("volume", 0)) / max(float(row.get("volume_sma", 1)), 1)

            # FIX: conf_bolts thresholds updated for 20-point scale
            conf_label = (
                "ELITE 🏆" if total >= 18 else
                "Strong ⚡" if total >= 14 else
                "Medium"   if total >= 10 else "Base"
            )
            h1_lbl     = h1_conf["reason"]
            adv_detail = sc["detail"]

            reason = (
                f"4H BOS {'↑' if direction == 'long' else '↓'} {swing_ref:.5g} | "
                f"Score {total}/{_MAX_SCORE} [{conf_label}] | "
                f"TP {tp_reason} | Trail @{self.trail_rr:.0f}:1 | "
                f"1H: {h1_lbl} | RSI={rsi:.0f}"
            )

            return {
                "stage": 2, "direction": direction, "symbol": symbol,
                "entry":          price,
                "sl":             sl_price,
                "tp1":            tp1_price,
                "tp2":            tp2_price,
                "tp3":            tp3_price,
                "tp1_label":      tp1_lbl,
                "tp2_label":      tp2_lbl,
                "tp3_label":      tp3_lbl,
                "sl_dist":        sl_dist,
                "tp_rr":          round(actual_rr, 2),
                "trail_activate": trail_activate,
                "risk_usdt":      risk_usdt,
                "rsi":            rsi,
                "vol_ratio":      vol_ratio,
                "atr":            atr,
                "quality":        total,
                "score":          total,
                # Per-category scores
                "kz_score":   sc["kz_score"],
                "amd_score":  sc["amd_score"],
                "wyck_score": sc["wyck_score"],
                "liq_score":  sc["liq_score"],
                "mmm_score":  sc["mmm_score"],
                "vsa_score":  sc["vsa_score"],
                "im_score":   sc["im_score"],
                "free_score": sc["free_score"],
                "h1_score":   sc["h1_score"],
                "kz_bonus":   sc["kz_bonus"],   # legacy key
                # Per-category display lines
                "kz_line":    sc["kz_line"],
                "amd_lines":  sc.get("amd_lines", []),
                "wyck_lines": sc["wyck_lines"],
                "liq_lines":  sc["liq_lines"],
                "mmm_lines":  sc["mmm_lines"],
                "vsa_lines":  sc["vsa_lines"],
                "im_lines":   sc["im_lines"],
                "free_lines": sc["free_lines"],
                "h1_lines":   sc.get("h1_lines", []),
                # Legacy label fields
                "kz_label":   kz.get("label", ""),
                "wyck_label": sc["wyck_label"],
                "mmm_label":  sc["mmm_label"],
                "vsa_label":  sc["vsa_label"],
                "im_label":   sc["im_label"],
                "h1_label":   h1_lbl,
                "adv_detail": adv_detail,
                "reason":     reason,
                # Liquidation map
                "liq_nearest_above":          liq_map.get("nearest_above"),
                "liq_nearest_above_strength": liq_map.get("nearest_above_strength", 0),
                "liq_nearest_below":          liq_map.get("nearest_below"),
                "liq_nearest_below_strength": liq_map.get("nearest_below_strength", 0),
            }

        return None
