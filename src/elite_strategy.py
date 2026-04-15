"""
Elite 4H Confluence Strategy  —  18-point scoring
---------------------------------------------------
Fires on 4H Break of Structure (BOS) when score >= min_score (default 5/18).

Hard requirements (ALL must be true — score irrelevant):
  1. Kill zone active — London 07:00-09:00 UTC OR NY 12:00-14:00 UTC only
  2. 4H candle CLOSED (never on open candle)
  3. Market regime clear — Bull=longs only, Bear=shorts only, Neutral=no signals
  4. 4H BOS: close breaks above last swing high (long) / below swing low (short)
  5. RSI in valid momentum zone: 40–75 (long), 25–60 (short)
  6. 1H must confirm direction (FVG / MSS / sweep)
  7. Not Friday after 17:00 UTC / Not Sunday before 22:00 UTC

Category scoring (per-category caps, total capped at 18):
  WYCKOFF         (max 5): Spring/Upthrust=3, SOS/SOW=2, Phase=1
  LIQUIDITY       (max 4): EQL/EQH swept=2, Stop hunt=2, Clear target=1, Swing swept=1
  MARKET MAKER    (max 4): Manipulation=3, AMD NY=2, Consolidation=1
  VSA             (max 3): No Supply/Demand/Stopping Vol/Climactic=2, Test=1
  INTERMARKET     (max 2): 3-4/5 aligned=1, 5/5=2
  FREE DATA       (max 4): Funding=1, OI=1, L/S ratio=1, Fear/Greed=1
  KILL ZONE BONUS (max 1): Active=1

Category gates (≥1pt required from EACH of: Wyckoff, Liquidity, MMM, VSA)

RR by score:
  5–7   → 2:1 min RR, $5 risk
  8–11  → 3:1 min RR, $7/$10 risk
  12+   → 5:1 min RR, $10/$12/$15 risk

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

logger = logging.getLogger("futures_bot.elite")

_MAX_SCORE = 18


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

    def _check_rr_clearance(self, h4_df, entry, sl_dist, direction,
                             min_rr: float | None = None) -> bool:
        """Return True if no 4H structure blocks the required RR target."""
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

    def _find_structural_tp(self, weekly_df, daily_df, entry, sl_dist, direction,
                             required_rr: float = 5.0) -> tuple[float | None, str]:
        """
        Find the nearest Weekly or Daily swing level at or beyond required_rr.
        Daily levels are checked first (more precise); weekly as fallback.
        Returns (price, reason_label) or (None, "").
        """
        candidates: list[tuple[float, str]] = []   # (price, label)

        # ── Daily levels ──────────────────────────────────────────────────
        if daily_df is not None and len(daily_df) >= 10:
            if direction == "long":
                swings = find_swing_highs_idx(daily_df.iloc[:-2], 3, 1)
                for _, lvl in swings:
                    if lvl > entry:
                        candidates.append((lvl, f"Daily resistance {lvl:.5g}"))
            else:
                swings = find_swing_lows_idx(daily_df.iloc[:-2], 3, 1)
                for _, lvl in swings:
                    if lvl < entry:
                        candidates.append((lvl, f"Daily support {lvl:.5g}"))

        # ── Weekly levels ─────────────────────────────────────────────────
        if weekly_df is not None and len(weekly_df) >= 4:
            if direction == "long":
                swings = find_swing_highs_idx(weekly_df.iloc[:-2], 2, 1)
                for _, lvl in swings:
                    if lvl > entry:
                        candidates.append((lvl, f"Weekly resistance {lvl:.5g}"))
            else:
                swings = find_swing_lows_idx(weekly_df.iloc[:-2], 2, 1)
                for _, lvl in swings:
                    if lvl < entry:
                        candidates.append((lvl, f"Weekly support {lvl:.5g}"))

        # Sort by proximity to entry
        if direction == "long":
            candidates.sort(key=lambda x: x[0])
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)

        for lvl, lbl in candidates:
            rr = abs(lvl - entry) / sl_dist
            if rr >= required_rr:
                return lvl, f"{lbl} ({rr:.1f}:1 RR)"

        return None, ""

    # ── Free data scoring (max 4pts) ─────────────────────────────────────────

    def _score_free_data(self, symbol: str, direction: str) -> tuple[int, list[str]]:
        """Funding rate, OI, L/S ratio, Fear/Greed — 1pt each, max 4."""
        score = 0
        lines: list[str] = []
        base  = symbol.split("/")[0]

        # Funding rate
        rate = get_funding_rate(None, symbol)
        if rate is None:
            score += 1; lines.append(f"✅ Funding Rate N/A   +1")
        elif direction == "long"  and rate <= 0:
            score += 1; lines.append(f"✅ Funding Rate {rate:.4f}%   +1")
        elif direction == "short" and rate >  0:
            score += 1; lines.append(f"✅ Funding Rate +{rate:.4f}%   +1")
        else:
            lines.append(f"⬜ Funding Rate {rate:.4f}%   +0")

        # Open interest — award if data present (confirms activity)
        oi = get_open_interest(None, symbol)
        if oi is not None:
            score += 1; lines.append(f"✅ OI ${oi:,.0f}   +1")
        else:
            lines.append(f"⬜ Open Interest N/A   +0")

        # Long/short ratio
        ls = get_long_short_ratio(base)
        if ls is None:
            score += 1; lines.append(f"✅ Long/Short Ratio N/A   +1")
        elif direction == "long"  and ls <= 2.5:
            score += 1; lines.append(f"✅ Long/Short Ratio {ls:.2f}   +1")
        elif direction == "short" and ls >= 0.4:
            score += 1; lines.append(f"✅ Long/Short Ratio {ls:.2f}   +1")
        else:
            lines.append(f"⬜ Long/Short Ratio {ls:.2f}   +0")

        # Fear & Greed: below 50 = fear = good for longs; above 50 = greed = good for shorts
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
        """True if the last closed candle swept a recent structural swing and closed back."""
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

    # ── Category scoring (18-point system) ───────────────────────────────────

    def _score_all_categories(
        self,
        h4_df: pd.DataFrame,
        weekly_df,
        symbol: str,
        direction: str,
        exchange=None,
        kz: dict | None = None,
    ) -> dict:
        """
        Score all 7 categories with per-category caps.  Total capped at 18.
        Returns a dict with per-category scores, line lists, and detail objects.
        """
        kz = kz or get_kill_zone()
        detail: dict = {"kz": kz}

        # ── WYCKOFF (cap 5) ───────────────────────────────────────────────
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
                wyck_raw = wyck["score"]
                wyck_lines.append(f"✅ {wyck['label'].split(': ', 1)[-1].split(' +')[0]}   +{wyck['score']}")
            else:
                wyck_lines.append(f"⬜ Wyckoff {wyck['phase'].title()} (mismatch)   +0")
        elif wyck["event"] in ("sc_detected", "bc_detected"):
            phase_aligned = (
                (direction == "long"  and wyck["phase"] == "accumulation") or
                (direction == "short" and wyck["phase"] == "distribution")
            )
            if phase_aligned:
                wyck_raw = 1
                wyck_lines.append(f"✅ {'Accumulation' if direction == 'long' else 'Distribution'} Phase   +1")
            else:
                wyck_lines.append(f"⬜ Wyckoff {wyck['phase'].title()} (mismatch)   +0")
        else:
            wyck_lines.append("⬜ Wyckoff: N/A   +0")
        wyck_score = min(5, wyck_raw)

        # ── LIQUIDITY (cap 4) ─────────────────────────────────────────────
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
            liq_raw += 2
            hunt_lbl = next((l for l in liq["label"] if "Hunt" in l or "hunt" in l), "Stop Hunt Detected")
            liq_lines.append(f"✅ {hunt_lbl.split(': ')[-1].split(' +')[0]}   +2")
        # Clear liquidity target
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
        liq_score = min(4, liq_raw)

        # ── MMM (cap 4) ───────────────────────────────────────────────────
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
                mmm_raw += 3
                mmm_lines.append(f"✅ Manipulation Confirmed   +3")
            else:
                mmm_lines.append(f"⬜ MMM Manipulation (mismatch)   +0")
        elif mmm["phase"] == "consolidation":
            mmm_raw += 1
            mmm_lines.append(f"✅ Consolidation Detected   +1")
        # AMD NY bonus — distribution phase during NY open
        if kz["amd_score"] >= 2:
            mmm_raw += 2
            mmm_lines.append(f"✅ AMD NY Confirmation   +2")
        if not mmm_lines:
            mmm_lines.append("⬜ MMM: N/A   +0")
        mmm_score = min(4, mmm_raw)

        # ── VSA (cap 3) ───────────────────────────────────────────────────
        vsa = detect_vsa(h4_df)
        detail["vsa"] = vsa
        vsa_raw   = 0
        vsa_lines = []
        if vsa["score"] > 0:   # no penalties in new system
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
        vsa_score = min(3, vsa_raw)

        # ── INTERMARKET (cap 2) ───────────────────────────────────────────
        try:
            im = get_intermarket_score(exchange=exchange, direction=direction)
        except Exception:
            im = {"score": 0, "label": "Intermarket: N/A", "n_align": 0, "n_total": 5}
        detail["intermarket"] = im
        im_score  = min(2, im["score"])
        n_align   = im.get("n_align", 0)
        n_total   = im.get("n_total", 5)
        if im_score > 0:
            im_lines = [f"✅ {n_align}/{n_total} Factors Aligned   +{im_score}"]
        else:
            im_lines = [f"⬜ Intermarket {n_align}/{n_total}   +0"]

        # ── FREE DATA (cap 4) ─────────────────────────────────────────────
        free_score, free_lines = self._score_free_data(symbol, direction)
        free_score = min(4, free_score)

        # ── KILL ZONE BONUS (cap 1) ───────────────────────────────────────
        kz_bonus  = 1 if kz["active"] else 0
        kz_name   = kz.get("name", "").upper().replace("_", " ")
        kz_line   = (f"✅ {kz_name} Active   +1" if kz_bonus
                     else f"⬜ Kill Zone   +0")

        total = min(_MAX_SCORE,
                    wyck_score + liq_score + mmm_score + vsa_score
                    + im_score + free_score + kz_bonus)

        return {
            "total":      total,
            "wyck_score": wyck_score, "liq_score":  liq_score,
            "mmm_score":  mmm_score,  "vsa_score":  vsa_score,
            "im_score":   im_score,   "free_score": free_score,
            "kz_bonus":   kz_bonus,
            "wyck_lines": wyck_lines, "liq_lines":  liq_lines,
            "mmm_lines":  mmm_lines,  "vsa_lines":  vsa_lines,
            "im_lines":   im_lines,   "free_lines": free_lines,
            "kz_line":    kz_line,
            "detail":     detail,
            # Legacy label fields used by bot.py for public channel message
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
        Scan for a 4H BOS signal with 18-point confluence scoring.

        Timeframe stack:
          weekly_df  — bias + major levels (20 candles)
          daily_df   — structure + TP targets (50 candles)
          h4_df      — primary entry TF, CLOSED candles only (100 candles)
          h1_df      — confirmation gate: FVG / MSS / sweep (100 candles)

        Returns a signal dict or None.
        """
        min_len = self.lookback + self.swing_left + self.swing_right + 5
        if len(h4_df) < min_len:
            return None

        # ── Kill Zone — bonus point only, not a hard gate ─────────────────
        # Detected during scoring; active session adds +1 to total score.
        kz = get_kill_zone()

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
                if mregime in ("bear", "neutral"):
                    continue
                sl_dist   = max(price - (float(row["low"]) - _sl_buf), atr * 0.5)
                swing_ref = swing_high
            else:
                bos = price < swing_low and float(prev["close"]) >= swing_low
                if not bos:
                    continue
                if rsi > 60 or rsi < 25:
                    continue
                if mregime in ("bull", "neutral"):
                    continue
                sl_dist   = max((float(row["high"]) + _sl_buf) - price, atr * 0.5)
                swing_ref = swing_low

            # ── Liquidation map — block if unswept cluster between entry & SL ─
            # Equal highs/lows between entry and SL are stop magnets.
            # Price will sweep them first, triggering our SL before the real
            # move begins. Skip the trade until that liquidity is cleared.
            sl_price_check = price - sl_dist if direction == "long" else price + sl_dist
            liq_map = calculate_liquidation_map(h4_df)
            clear, liq_reason = liq_map_clear_to_entry(liq_map, price, sl_price_check, direction)
            if not clear:
                logger.debug(f"[ELITE] {symbol} {direction.upper()} — {liq_reason}")
                continue

            # ── Fast 2:1 clearance gate (minimum possible RR) ─────────────
            if not self._check_rr_clearance(h4_df, price, sl_dist, direction, min_rr=2.0):
                logger.debug(f"[ELITE] {symbol} {direction.upper()} — no 2:1 RR clearance")
                continue

            # ── Score all 7 categories (18-point system) ───────────────────
            sc = self._score_all_categories(h4_df, weekly_df, symbol, direction, exchange, kz)
            total = sc["total"]

            # ── Minimum score gate ─────────────────────────────────────────
            # No per-category requirement — bot grabs whatever confluence is
            # present and fires if total ≥ 5. Spring + MMM alone = 6pts which
            # always co-occur anyway, so category gates are redundant friction.
            if total < self.min_score:
                logger.debug(
                    f"[ELITE] {symbol} {direction.upper()} score={total}/{_MAX_SCORE} "
                    f"< {self.min_score} — skipped"
                )
                continue

            # ── Score-based RR clearance (2:1 / 3:1 / 5:1) ───────────────
            required_rr = tp_rr_for_score(total)
            if required_rr > 2.0:
                if not self._check_rr_clearance(h4_df, price, sl_dist, direction,
                                                 min_rr=required_rr):
                    logger.debug(
                        f"[ELITE] {symbol} {direction.upper()} — no {required_rr:.0f}:1 clearance"
                    )
                    continue

            # ── 1H confirmation gate ───────────────────────────────────────
            h1_conf = confirm_1h_alignment(h1_df, direction)
            if not h1_conf["aligned"]:
                logger.debug(
                    f"[ELITE] {symbol} {direction.upper()} — 1H not aligned: "
                    f"{h1_conf['reason']}"
                )
                continue

            # ── TP — Weekly/Daily structural level at required RR ──────────
            struct_tp, struct_label = self._find_structural_tp(
                weekly_df, daily_df, price, sl_dist, direction,
                required_rr=required_rr,
            )
            if struct_tp:
                tp_price  = struct_tp
                tp_reason = struct_label
            else:
                tp_price  = (
                    price + required_rr * sl_dist if direction == "long"
                    else price - required_rr * sl_dist
                )
                tp_reason = f"Fixed {required_rr:.0f}:1 RR (no structural level)"

            actual_rr      = abs(tp_price - price) / sl_dist
            trail_activate = (
                price + self.trail_rr * sl_dist if direction == "long"
                else price - self.trail_rr * sl_dist
            )
            risk_usdt = risk_usdt_for_score(total)
            sl_price  = price - sl_dist if direction == "long" else price + sl_dist
            vol_ratio = float(row.get("volume", 0)) / max(float(row.get("volume_sma", 1)), 1)

            conf_label = (
                "ELITE 🏆" if total >= 15 else
                "Strong ⚡" if total >= 12 else
                "Medium"   if total >= 8  else "Base"
            )
            h1_lbl = h1_conf["reason"]
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
                "tp1":            tp_price,
                "tp2":            tp_price,
                "tp3":            tp_price,
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
                "wyck_score":  sc["wyck_score"],
                "liq_score":   sc["liq_score"],
                "mmm_score":   sc["mmm_score"],
                "vsa_score":   sc["vsa_score"],
                "im_score":    sc["im_score"],
                "free_score":  sc["free_score"],
                "kz_bonus":    sc["kz_bonus"],
                # Per-category display lines
                "wyck_lines":  sc["wyck_lines"],
                "liq_lines":   sc["liq_lines"],
                "mmm_lines":   sc["mmm_lines"],
                "vsa_lines":   sc["vsa_lines"],
                "im_lines":    sc["im_lines"],
                "free_lines":  sc["free_lines"],
                "kz_line":     sc["kz_line"],
                # Legacy label fields
                "kz_label":    kz.get("label", ""),
                "wyck_label":  sc["wyck_label"],
                "mmm_label":   sc["mmm_label"],
                "vsa_label":   sc["vsa_label"],
                "im_label":    sc["im_label"],
                "h1_label":    h1_lbl,
                "adv_detail":  adv_detail,
                "reason":      reason,
                # Liquidation map — nearest clusters above/below entry
                "liq_nearest_above":          liq_map.get("nearest_above"),
                "liq_nearest_above_strength": liq_map.get("nearest_above_strength", 0),
                "liq_nearest_below":          liq_map.get("nearest_below"),
                "liq_nearest_below_strength": liq_map.get("nearest_below_strength", 0),
            }

        return None
