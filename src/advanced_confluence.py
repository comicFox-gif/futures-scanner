"""
Advanced Confluence Detectors — v2.0
--------------------------------------
Rebuilt scoring system:
  - Wyckoff SC detection fixed (removed bad open>close filter, lowered thresholds)
  - Kill zone reweighted +1 → +3 (most impactful change)
  - 1H confirmation now scores points (was binary gate only)
  - Stop hunt reweighted +2 → +1 (was overweighted vs kill zone)
  - MMM reweighted +3 → +2
  - Wyckoff reweighted +3 → +2 (now that it fires reliably)

New max score = 20
Recommended minimum to trade = 12

Detectors:
  get_kill_zone()             → session + AMD phase + score (max +5)
  detect_wyckoff(df)          → Wyckoff phase/event + score (max +2)
  detect_liquidity(df)        → EQH/EQL sweep + stop hunt + score (max +5)
  detect_mmm(df)              → Market Maker Model phase + score (max +2)
  detect_vsa(df)              → Volume Spread Analysis + score (max +2)
  detect_delta_divergence(df) → price vs volume-delta alignment (info only)
  get_intermarket_score()     → macro score (max +2)
  confirm_1h_alignment()      → 1H gate + score (max +3)
  risk_usdt_for_score()       → position sizing by score
  tp_rr_for_score()           → TP RR by score
  calculate_liquidation_map() → stop cluster map
  liq_map_clear_to_entry()    → entry safety check
"""

from __future__ import annotations
import logging
import time as _time
from datetime import datetime

import pandas as pd

logger = logging.getLogger("futures_bot.advanced")


# ══════════════════════════════════════════════════════════════════════════════
# SCORE REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
#
#  Kill Zone active (London / NY)          +3
#  AMD distribution phase (12–14 UTC)      +2
#  AMD distribution phase (14–17 UTC)      +1
#  Wyckoff spring / SOS / upthrust / SOW   +2
#  Wyckoff SC/BC detected (phase only)     +1
#  EQL swept                               +2
#  EQH swept                               +2
#  Stop hunt (bullish or bearish)          +1
#  MMM manipulation                        +2
#  MMM consolidation                       +1
#  VSA signal                              +2
#  Intermarket 3+/5 aligned                +1
#  Intermarket 5/5 aligned                 +2
#  1H FVG present                          +2
#  1H MSS present                          +1
#  1H sweep present                        +0  (gate only, no extra score)
#
#  MAX POSSIBLE                            ~20
#  MINIMUM TO TRADE                        12
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Kill Zone + AMD Phase   (max +5)
# ══════════════════════════════════════════════════════════════════════════════

def get_kill_zone() -> dict:
    """
    Return the active session/kill-zone for the current UTC time.

    Kill zones (reweighted to +3 — most impactful filter):
      London          07:00–09:00 UTC  → score +3
      NY              12:00–14:00 UTC  → score +3
      No trade zone                   → score  0

    AMD phase (unchanged):
      12–14 UTC  Distribution prime   → amd_score +2
      14–17 UTC  Distribution late    → amd_score +1
      Other                           → amd_score  0

    Returns dict:
      active     bool
      name       str
      score      int   (0 or 3)
      label      str
      amd_phase  str
      amd_score  int   (0, 1, or 2)
    """
    now = datetime.utcnow()
    h   = now.hour
    wd  = now.weekday()   # 0=Mon … 6=Sun

    # Weekend hard blocks
    if wd == 4 and h >= 17:
        return _kz(False, "weekend", 0, "⏸ Weekend — No Trade Zone", "none", 0)
    if wd == 6 and h < 22:
        return _kz(False, "sunday", 0, "⏸ Sunday — No Trade Zone", "none", 0)

    # AMD phase
    if 12 <= h < 14:
        amd = "distribution"; amd_score = 2   # NY open — best window
    elif 14 <= h < 17:
        amd = "distribution"; amd_score = 1   # distribution late
    elif 8 <= h < 12:
        amd = "manipulation"; amd_score = 0
    elif 0 <= h < 8:
        amd = "accumulation"; amd_score = 0
    else:
        amd = "none"; amd_score = 0

    # Kill zones — reweighted to +3
    if 7 <= h < 9:
        return _kz(True,  "london", 3, "⏰ London Kill Zone 07–09 UTC ✅ +3", amd, amd_score)
    if 12 <= h < 14:
        return _kz(True,  "ny",     3, "⏰ NY Kill Zone 12–14 UTC ✅ +3",     amd, amd_score)

    return _kz(False, "no_trade", 0, f"⏸ No Trade Zone ({h:02d}:00 UTC)", amd, 0)


def _kz(active, name, score, label, amd_phase, amd_score):
    return dict(
        active=active, name=name, score=score, label=label,
        amd_phase=amd_phase, amd_score=amd_score,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Wyckoff Phase Detection   (max +2)
# ══════════════════════════════════════════════════════════════════════════════

def detect_wyckoff(df: pd.DataFrame, lookback: int = 80) -> dict:
    """
    Detect Wyckoff accumulation/distribution phases and key events.

    FIX v2: Removed pk_open > pk_close requirement from SC detection.
    A selling climax often prints as a long lower-wick bullish reversal candle —
    requiring a bearish body was filtering out the exact event we want.
    Volume and close-position thresholds also lowered slightly for real data.

    FIX v2: Max score reduced from +3 to +2 (now that Wyckoff fires reliably,
    it shouldn't dominate over kill zone).

    Events and scores:
      spring / sos / upthrust / sow   +2
      sc_detected / bc_detected        +1
    """
    NONE = dict(phase="unknown", event=None, score=0, label="Wyckoff: N/A")
    try:
        if len(df) < lookback + 5:
            return NONE

        win   = df.iloc[-(lookback + 5):]
        vol   = win["volume"].values
        close = win["close"].values
        high  = win["high"].values
        low   = win["low"].values
        open_ = win["open"].values

        vol_sma = pd.Series(vol).rolling(20).mean()
        vol_avg = float(vol_sma.iloc[-2]) if not pd.isna(vol_sma.iloc[-2]) else float(vol.mean())
        if vol_avg == 0:
            return NONE

        # Highest-volume candle (excluding last two — unconfirmed)
        peak_idx = int(pd.Series(vol[:-2]).idxmax())
        peak_vol = float(vol[peak_idx])
        pk_close = float(close[peak_idx])
        pk_high  = float(high[peak_idx])
        pk_low   = float(low[peak_idx])
        pk_open  = float(open_[peak_idx])
        pk_rng   = pk_high - pk_low

        cur_idx   = len(win) - 2
        cur_close = float(close[cur_idx])
        cur_high  = float(high[cur_idx])
        cur_low   = float(low[cur_idx])
        cur_vol   = float(vol[cur_idx])

        lookback_start = max(0, peak_idx - 10)

        # ── Selling Climax (SC) ────────────────────────────────────────────
        # FIX: removed pk_open > pk_close (bearish body requirement)
        # FIX: lowered volume threshold 2.0 → 1.5, close ratio 0.5 → 0.4
        sc_close_ratio = (pk_close - pk_low) / max(pk_rng, 1e-10)
        is_downtrend   = pk_close < float(close[lookback_start])
        is_sc = (
            peak_vol > vol_avg * 1.5      # was 2.0 — too strict for real data
            and sc_close_ratio > 0.4      # was 0.5 — SC can close mid-range
            and is_downtrend
            # REMOVED: pk_open > pk_close — SC often prints as wick reversal
        )

        # ── Buying Climax (BC) ────────────────────────────────────────────
        # FIX: same threshold relaxations applied
        bc_close_ratio = (pk_high - pk_close) / max(pk_rng, 1e-10)
        is_uptrend     = pk_close > float(close[lookback_start])
        is_bc = (
            peak_vol > vol_avg * 1.5      # was 2.0
            and bc_close_ratio > 0.4      # was 0.5
            and is_uptrend
            # REMOVED: pk_open < pk_close — BC can print as a wick reversal too
        )

        sc_low  = float(low[peak_idx])  if is_sc else None
        bc_high = float(high[peak_idx]) if is_bc else None

        # ── ACCUMULATION events ───────────────────────────────────────────
        if is_sc:
            post_sc = win.iloc[peak_idx + 1: cur_idx]
            ar_high = float(post_sc["high"].max()) if len(post_sc) >= 2 else None

            # Spring: wicks below SC low on low volume, closes back above
            if sc_low and cur_low < sc_low and cur_close > sc_low:
                if cur_vol / vol_avg < 1.5:
                    return dict(phase="accumulation", event="spring", score=2,
                                label="Wyckoff: ACCUMULATION — Spring 🌱 +2")

            # Secondary Test: near SC zone, lower volume than SC
            if sc_low and abs(cur_close - sc_low) / sc_low < 0.02:
                if cur_vol < peak_vol * 0.7:
                    return dict(phase="accumulation", event="secondary_test", score=1,
                                label="Wyckoff: ACCUMULATION — Secondary Test +1")

            # Sign of Strength: breaks above AR level with high volume
            if ar_high and cur_close > ar_high and cur_vol > vol_avg * 1.5:
                return dict(phase="accumulation", event="sos", score=2,
                            label="Wyckoff: ACCUMULATION — Sign of Strength ✅ +2")

            return dict(phase="accumulation", event="sc_detected", score=1,
                        label="Wyckoff: Accumulation Phase +1")

        # ── DISTRIBUTION events ───────────────────────────────────────────
        if is_bc:
            # Upthrust: wicks above BC high on low volume, closes back below
            if bc_high and cur_high > bc_high and cur_close < bc_high:
                if cur_vol / vol_avg < 1.5:
                    return dict(phase="distribution", event="upthrust", score=2,
                                label="Wyckoff: DISTRIBUTION — Upthrust ⬆️ +2")

            # Sign of Weakness: breaks below recent support with high volume
            post_bc    = win.iloc[peak_idx + 1: cur_idx]
            recent_low = float(post_bc["low"].min()) if len(post_bc) >= 2 else pk_low
            if cur_close < recent_low and cur_vol > vol_avg * 1.5:
                return dict(phase="distribution", event="sow", score=2,
                            label="Wyckoff: DISTRIBUTION — Sign of Weakness ✅ +2")

            return dict(phase="distribution", event="bc_detected", score=1,
                        label="Wyckoff: Distribution Phase +1")

    except Exception as e:
        logger.debug(f"[WYCKOFF] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Equal Highs / Equal Lows + Stop Hunt Detection   (max +5)
# ══════════════════════════════════════════════════════════════════════════════

def detect_liquidity(
    df: pd.DataFrame,
    lookback: int = 50,
    threshold_pct: float = 0.003,
) -> dict:
    """
    Detect Equal Highs (EQH), Equal Lows (EQL), and stop hunts.

    FIX v2: Stop hunt reweighted +2 → +1.
    It was equal to EQL/EQH swept which is wrong — a full sweep of a
    multi-touch level is stronger confluence than a single prev-candle spike.

    Scores:
      EQH swept   +2
      EQL swept   +2
      Stop hunt   +1  (was +2)
    """
    NONE = dict(
        eqh_levels=[], eql_levels=[],
        eqh_swept=False, eql_swept=False, stop_hunt=False,
        sweep_vol_confirmed=False,
        score=0, label=[],
    )
    try:
        if len(df) < lookback + 5:
            return NONE

        win   = df.iloc[-(lookback + 5):-2]
        cur   = df.iloc[-2]
        prev  = df.iloc[-3]

        highs   = win["high"].values
        lows    = win["low"].values
        vol_avg = float(win["volume"].mean())

        cur_high  = float(cur["high"])
        cur_low   = float(cur["low"])
        cur_close = float(cur["close"])
        cur_vol   = float(cur["volume"])
        prev_high = float(prev["high"])
        prev_low  = float(prev["low"])

        # ── EQH clusters ─────────────────────────────────────────────────
        eqh_levels: list[float] = []
        for i, h in enumerate(highs):
            cluster = [h]
            for j in range(i + 1, len(highs)):
                if abs(highs[j] - h) / max(h, 1e-10) < threshold_pct:
                    cluster.append(highs[j])
            if len(cluster) >= 2:
                lvl = sum(cluster) / len(cluster)
                if not any(abs(lvl - e) / max(e, 1e-10) < threshold_pct for e in eqh_levels):
                    eqh_levels.append(lvl)

        # ── EQL clusters ─────────────────────────────────────────────────
        eql_levels: list[float] = []
        for i, lo in enumerate(lows):
            cluster = [lo]
            for j in range(i + 1, len(lows)):
                if abs(lows[j] - lo) / max(lo, 1e-10) < threshold_pct:
                    cluster.append(lows[j])
            if len(cluster) >= 2:
                lvl = sum(cluster) / len(cluster)
                if not any(abs(lvl - e) / max(e, 1e-10) < threshold_pct for e in eql_levels):
                    eql_levels.append(lvl)

        labels: list[str] = []
        score     = 0
        eqh_swept = False
        eql_swept = False
        stop_hunt = False
        # Whale confirmation: a real sweep transacts size. A thin-volume wick
        # through equal highs/lows is more likely the trap itself, not whales.
        sweep_vol_confirmed = cur_vol > vol_avg

        # EQH swept: wick above EQH, close below
        for lvl in eqh_levels:
            if cur_high > lvl * (1 + threshold_pct * 0.3) and cur_close < lvl:
                eqh_swept = True
                score += 2
                vtag = " + whale vol 🐋" if sweep_vol_confirmed else " (thin vol ⚠️)"
                labels.append(f"Liquidity: EQH Swept ${lvl:,.4g} +2 💧{vtag}")
                break

        # EQL swept: wick below EQL, close above
        for lvl in eql_levels:
            if cur_low < lvl * (1 - threshold_pct * 0.3) and cur_close > lvl:
                eql_swept = True
                score += 2
                vtag = " + whale vol 🐋" if sweep_vol_confirmed else " (thin vol ⚠️)"
                labels.append(f"Liquidity: EQL Swept ${lvl:,.4g} +2 💧{vtag}")
                break

        # Stop hunt: spike beyond previous candle + reversal close
        # FIX v2: reduced from +2 → +1 (less weight than full EQL/EQH sweep)
        if (cur_low < prev_low * (1 - threshold_pct)
                and cur_close > prev_low
                and cur_vol > vol_avg * 0.9):
            stop_hunt = True
            score += 1          # was +2
            labels.append("Liquidity: Sellside Stop Hunt ✅ +1")
        elif (cur_high > prev_high * (1 + threshold_pct)
              and cur_close < prev_high
              and cur_vol > vol_avg * 0.9):
            stop_hunt = True
            score += 1          # was +2
            labels.append("Liquidity: Buyside Stop Hunt ✅ +1")

        return dict(
            eqh_levels=eqh_levels, eql_levels=eql_levels,
            eqh_swept=eqh_swept,   eql_swept=eql_swept,
            stop_hunt=stop_hunt,
            sweep_vol_confirmed=sweep_vol_confirmed,
            score=score, label=labels,
        )

    except Exception as e:
        logger.debug(f"[LIQUIDITY] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Market Maker Model (MMM)   (max +2)
# ══════════════════════════════════════════════════════════════════════════════

def detect_mmm(df: pd.DataFrame) -> dict:
    """
    Detect Market Maker Model phases on 4H candles.

    FIX v2: Max score reduced +3 → +2.
    MMM manipulation is useful but shouldn't outscore kill zone presence.

    Scores:
      Manipulation below/above range   +2  (was +3)
      Consolidation only               +1  (unchanged)
    """
    NONE = dict(phase="unknown", manipulation=False, direction=None, score=0, label="MMM: N/A")
    try:
        if len(df) < 20:
            return NONE

        win = df.iloc[-25:-2]
        cur = df.iloc[-2]

        atr_col = "atr" if "atr" in win.columns else None
        if atr_col:
            avg_range = float(win[atr_col].mean())
        else:
            avg_range = float((win["high"] - win["low"]).mean())

        consol = win.iloc[-6:]
        c_high = float(consol["high"].max())
        c_low  = float(consol["low"].min())
        c_rng  = c_high - c_low

        is_consolidating = c_rng < avg_range * 1.8 and len(consol) >= 3

        cur_high  = float(cur["high"])
        cur_low   = float(cur["low"])
        cur_close = float(cur["close"])
        cur_vol   = float(cur["volume"])
        vol_avg   = float(win["volume"].mean())

        if not is_consolidating:
            return dict(phase="trending", manipulation=False, direction=None,
                        score=0, label="MMM: Trending — no consolidation detected")

        # Manipulation DOWN → expect LONG
        if cur_low < c_low and cur_close > c_low and cur_vol < vol_avg * 1.3:
            return dict(
                phase="manipulation", manipulation=True, direction="long",
                score=2,    # was +3
                label="MMM: Manipulation ↓ below range — Real move UP expected 🎯 +2",
            )

        # Manipulation UP → expect SHORT
        if cur_high > c_high and cur_close < c_high and cur_vol < vol_avg * 1.3:
            return dict(
                phase="manipulation", manipulation=True, direction="short",
                score=2,    # was +3
                label="MMM: Manipulation ↑ above range — Real move DOWN expected 🎯 +2",
            )

        return dict(
            phase="consolidation", manipulation=False, direction=None,
            score=1,
            label=f"MMM: Consolidation {c_low:.5g}–{c_high:.5g} +1",
        )

    except Exception as e:
        logger.debug(f"[MMM] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Volume Spread Analysis (VSA)   (max +2)
# ══════════════════════════════════════════════════════════════════════════════

def detect_vsa(df: pd.DataFrame) -> dict:
    """
    Analyse price spread vs volume for institutional footprints.
    Unchanged from v1 — scoring was already appropriate.

    Bullish signals:
      no_supply          +2
      stopping_volume    +2
      test_bar           +1

    Bearish signals:
      no_demand          +2
      climactic          +2
      upthrust_bar       +2

    Warning:
      effort_no_result   -1
    """
    NONE = dict(signal=None, score=0, label="VSA: N/A", bullish=None)
    try:
        if len(df) < 22:
            return NONE

        win = df.iloc[-22:-2]
        cur = df.iloc[-2]

        vol_avg    = float(win["volume"].mean())
        spread_avg = float((win["high"] - win["low"]).mean())
        if vol_avg == 0 or spread_avg == 0:
            return NONE

        c_high = float(cur["high"])
        c_low  = float(cur["low"])
        c_cls  = float(cur["close"])
        c_opn  = float(cur["open"])
        c_vol  = float(cur["volume"])

        spread    = c_high - c_low
        body      = abs(c_cls - c_opn)
        close_pos = (c_cls - c_low) / max(spread, 1e-10)
        is_bull   = c_cls > c_opn

        # ── Bullish VSA ───────────────────────────────────────────────────
        if spread < spread_avg * 0.8 and c_vol < vol_avg * 0.8 and close_pos > 0.6:
            return dict(signal="no_supply", score=2,
                        label="VSA: No Supply detected ✅ +2", bullish=True)

        if c_vol > vol_avg * 2.0 and not is_bull and close_pos > 0.6:
            return dict(signal="stopping_volume", score=2,
                        label="VSA: Stopping Volume confirmed 🛑 +2", bullish=True)

        if c_vol < vol_avg * 0.7 and not is_bull and close_pos > 0.65:
            return dict(signal="test_bar", score=1,
                        label="VSA: Test Bar — no supply found +1", bullish=True)

        # ── Bearish VSA ───────────────────────────────────────────────────
        if spread < spread_avg * 0.8 and c_vol < vol_avg * 0.7 and is_bull and close_pos < 0.5:
            return dict(signal="no_demand", score=2,
                        label="VSA: No Demand — weakness ⚠️ +2", bullish=False)

        if c_vol > vol_avg * 2.5 and spread > spread_avg * 1.5 and close_pos < 0.4:
            return dict(signal="climactic", score=2,
                        label="VSA: Climactic Action — reversal near ⚠️ +2", bullish=False)

        if not is_bull and c_vol > vol_avg * 1.5 and close_pos < 0.35 and spread > spread_avg:
            return dict(signal="upthrust_bar", score=2,
                        label="VSA: Upthrust Bar — distribution ⚠️ +2", bullish=False)

    except Exception as e:
        logger.debug(f"[VSA] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Delta Divergence   (informational only — no score)
# ══════════════════════════════════════════════════════════════════════════════

def detect_delta_divergence(df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Volume-delta proxy: compare cumulative buy/sell pressure vs price direction.
    Informational only — score = 0. Used as a warning flag in signal output.
    """
    NONE = dict(divergence=False, price_dir=0, delta_dir=0, score=0, label="Delta: N/A")
    try:
        if len(df) < lookback + 3:
            return NONE

        win       = df.iloc[-(lookback + 3):-2]
        cur       = df.iloc[-2]
        price_dir = 1 if float(cur["close"]) > float(win.iloc[0]["close"]) else -1

        buy_vol = sell_vol = 0.0
        for _, row in win.iterrows():
            rng = float(row["high"]) - float(row["low"])
            cp  = (float(row["close"]) - float(row["low"])) / max(rng, 1e-10)
            v   = float(row["volume"])
            buy_vol  += v * cp
            sell_vol += v * (1.0 - cp)

        delta_dir = 1 if buy_vol > sell_vol else -1
        divergent = price_dir != delta_dir

        return dict(
            divergence=divergent,
            price_dir=price_dir,
            delta_dir=delta_dir,
            score=0,
            label=(
                f"Delta: Divergence ⚠️ price={'↑' if price_dir > 0 else '↓'} "
                f"delta={'↑' if delta_dir > 0 else '↓'}"
                if divergent
                else f"Delta: Aligned {'↑' if delta_dir > 0 else '↓'} ✅"
            ),
        )

    except Exception as e:
        logger.debug(f"[DELTA] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Intermarket Analysis   (max +2)
# ══════════════════════════════════════════════════════════════════════════════

_im_cache: dict  = {}
_im_ts:    float = 0.0
_IM_TTL          = 3600.0   # 1-hour cache


def get_intermarket_score(exchange=None, direction: str = "long") -> dict:
    """
    Fetch and score macro intermarket factors for the given trade direction.
    Unchanged from v1 — scoring was already correct.

    3+/5 aligned → +1
    5/5 aligned  → +2
    """
    global _im_cache, _im_ts

    if _time.time() - _im_ts < _IM_TTL and _im_cache:
        raw = _im_cache
    else:
        raw       = _fetch_intermarket(exchange)
        _im_cache = raw
        _im_ts    = _time.time()

    checks = {
        "Funding": raw.get("funding_bearish"),
        "MktCap":  raw.get("mktcap_rising"),
        "BTC.D":   raw.get("btcd_low"),
        "ETH/BTC": raw.get("ethbtc_rising"),
        "F/G":     raw.get("fg_bullish"),
    }

    if direction == "short":
        checks = {k: (not v if v is not None else None) for k, v in checks.items()}

    filled  = {k: v for k, v in checks.items() if v is not None}
    n_total = len(filled)
    n_align = sum(1 for v in filled.values() if v)

    if n_total >= 5 and n_align == 5:
        score = 2
    elif n_total >= 3 and n_align >= 3:
        score = 1
    else:
        score = 0

    tag   = {True: "✅", False: "❌", None: "—"}
    lines = [f"  {k}: {tag[v]}" for k, v in checks.items()]
    score_sfx = f" +{score}" if score else ""
    label = (
        f"Intermarket: {n_align}/{n_total} aligned{score_sfx}\n"
        + "\n".join(lines)
    )

    return dict(score=score, label=label, n_align=n_align, n_total=n_total, checks=checks)


def _fetch_intermarket(exchange=None) -> dict:
    """Internal: fetch raw intermarket data — crypto-native sources only."""
    data: dict = {
        "funding_bearish": None,
        "mktcap_rising":   None,
        "btcd_low":        None,
        "ethbtc_rising":   None,
        "fg_bullish":      None,
    }

    # BTC Funding Rate
    try:
        from src.sentiment import get_funding_rate
        rate = get_funding_rate(None, "BTC/USDT:USDT")
        if rate is not None:
            data["funding_bearish"] = rate <= 0
    except Exception as e:
        logger.debug(f"[INTERMARKET] Funding rate: {e}")

    # Total crypto market cap + BTC Dominance
    try:
        import requests as _req
        resp = _req.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=6,
            headers={"User-Agent": "futures-bot/1.0"},
        )
        if resp.status_code == 200:
            gdata = resp.json().get("data", {})
            btcd  = gdata.get("market_cap_percentage", {}).get("btc")
            if btcd is not None:
                data["btcd_low"] = float(btcd) < 55.0
            mktcap_chg = gdata.get("market_cap_change_percentage_24h_usd")
            if mktcap_chg is not None:
                data["mktcap_rising"] = float(mktcap_chg) > 0
    except Exception as e:
        logger.debug(f"[INTERMARKET] CoinGecko global: {e}")

    # ETH/BTC ratio
    if exchange is not None:
        try:
            ticker  = exchange.fetch_ticker("ETH/BTC")
            eth_now = float(ticker.get("last") or 0)
            ohlcv   = exchange.fetch_ohlcv("ETH/BTC", "1d", limit=3)
            if ohlcv and len(ohlcv) >= 2:
                prev = float(ohlcv[-2][4])
                data["ethbtc_rising"] = eth_now > prev
        except Exception as e:
            logger.debug(f"[INTERMARKET] ETH/BTC fetch: {e}")

    # Fear & Greed
    try:
        from src.sentiment import get_fear_greed
        fg = get_fear_greed()
        data["fg_bullish"] = fg["value"] < 50
    except Exception as e:
        logger.debug(f"[INTERMARKET] F&G fetch: {e}")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# 8.  1H Confirmation Gate + Score   (max +3)
# ══════════════════════════════════════════════════════════════════════════════

def confirm_1h_alignment(h1_df: pd.DataFrame, direction: str) -> dict:
    """
    Check if the 1H timeframe confirms the 4H signal direction.

    FIX v2: Now returns a score instead of just a boolean gate.
    Previously fvg_present and mss_present contributed 0 to the total score —
    these are strong confluence signals and should be rewarded.

    Scoring:
      FVG present   +2  (3-candle imbalance = institutional order block)
      MSS present   +1  (structure break = momentum confirmation)
      Sweep present  0  (gate only — already captured in liquidity score)

    Passes if ANY one condition is present (gate logic unchanged).
    """
    FAIL = dict(aligned=False, reason="1H: Insufficient data",
                fvg_present=False, mss_present=False, sweep_present=False,
                score=0)
    try:
        if h1_df is None or len(h1_df) < 20:
            return FAIL

        cur     = h1_df.iloc[-2]
        prev    = h1_df.iloc[-3]
        vol_win = h1_df.iloc[-15:-2]
        vol_avg = float(vol_win["volume"].mean()) if len(vol_win) else 1.0

        fvg_present   = False
        mss_present   = False
        sweep_present = False
        reasons: list[str] = []
        score = 0

        # ── 1. FVG ───────────────────────────────────────────────────────
        for i in range(-8, -3):
            try:
                c0 = h1_df.iloc[i - 1]
                c2 = h1_df.iloc[i + 1]
                if direction == "long":
                    if float(c2["low"]) > float(c0["high"]):
                        fvg_present = True
                        score += 2      # FIX: was 0
                        reasons.append("1H FVG bullish ✅ +2")
                        break
                else:
                    if float(c2["high"]) < float(c0["low"]):
                        fvg_present = True
                        score += 2      # FIX: was 0
                        reasons.append("1H FVG bearish ✅ +2")
                        break
            except IndexError:
                pass

        # ── 2. MSS — 1H structure break ──────────────────────────────────
        lookback  = h1_df.iloc[-15:-3]
        cur_close = float(cur["close"])
        if direction == "long":
            sh = float(lookback["high"].max())
            if cur_close > sh:
                mss_present = True
                score += 1          # FIX: was 0
                reasons.append(f"1H MSS ↑ above {sh:,.4g} ✅ +1")
        else:
            sl = float(lookback["low"].min())
            if cur_close < sl:
                mss_present = True
                score += 1          # FIX: was 0
                reasons.append(f"1H MSS ↓ below {sl:,.4g} ✅ +1")

        # ── 3. Liquidity Sweep on 1H ──────────────────────────────────────
        cur_vol = float(cur["volume"])
        if direction == "long":
            if (float(cur["low"])  < float(prev["low"])
                    and cur_close  > float(prev["low"])
                    and cur_vol    > vol_avg * 0.8):
                sweep_present = True
                reasons.append("1H sell-side sweep ✅")   # no extra score — in liquidity already
        else:
            if (float(cur["high"]) > float(prev["high"])
                    and cur_close  < float(prev["high"])
                    and cur_vol    > vol_avg * 0.8):
                sweep_present = True
                reasons.append("1H buy-side sweep ✅")

        aligned = fvg_present or mss_present or sweep_present
        reason  = (
            " | ".join(reasons)
            if aligned
            else f"1H disagrees — no FVG/MSS/sweep for {direction}"
        )
        return dict(
            aligned=aligned, reason=reason,
            fvg_present=fvg_present,
            mss_present=mss_present,
            sweep_present=sweep_present,
            score=score,    # FIX: now contributes to total confluence score
        )

    except Exception as e:
        logger.debug(f"[1H CONFIRM] {e}")
        return FAIL


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Position sizing + TP RR by score
# ══════════════════════════════════════════════════════════════════════════════

def risk_usdt_for_score(score: int) -> float:
    """
    Dollar risk based on 20-point confluence score.

    Updated thresholds to match new max score of 20.

    7–9   → $5   (marginal — consider skipping)
    10–12 → $7
    13–15 → $10
    16–18 → $12
    19–20 → $15
    """
    if score >= 19:  return 15.0
    if score >= 16:  return 12.0
    if score >= 13:  return 10.0
    if score >= 10:  return 7.0
    return 5.0


def tp_rr_for_score(score: int) -> float:
    """
    Minimum TP RR based on 20-point confluence score.

    <10   → 2:1  (weak signal, tight target)
    10–13 → 3:1  (medium confidence)
    14+   → 5:1  (strong / elite)
    """
    if score >= 14:  return 5.0
    if score >= 10:  return 3.0
    return 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 10.  Liquidation Map   (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_liquidation_map(
    df: pd.DataFrame,
    lookback: int = 60,
    threshold_pct: float = 0.003,
) -> dict:
    """
    Estimate where stop/liquidation clusters sit using pure price action.
    Unchanged from v1 — logic was solid.
    """
    NONE = dict(
        above=[], below=[],
        nearest_above=None, nearest_below=None,
        nearest_above_strength=0, nearest_below_strength=0,
    )
    try:
        if len(df) < lookback + 5:
            return NONE

        win   = df.iloc[-(lookback + 5):-2]
        price = float(df.iloc[-2]["close"])
        highs = win["high"].values
        lows  = win["low"].values

        def _cluster(values):
            seen   = [False] * len(values)
            groups = []
            for i, v in enumerate(values):
                if seen[i]:
                    continue
                cluster = [v]
                seen[i] = True
                for j in range(i + 1, len(values)):
                    if not seen[j] and abs(values[j] - v) / max(v, 1e-10) < threshold_pct:
                        cluster.append(values[j])
                        seen[j] = True
                if len(cluster) >= 2:
                    groups.append((sum(cluster) / len(cluster), len(cluster)))
            return groups

        high_clusters = _cluster(highs)
        low_clusters  = _cluster(lows)

        above = sorted(
            [(lvl, cnt) for lvl, cnt in high_clusters if lvl > price],
            key=lambda x: x[0],
        )
        below = sorted(
            [(lvl, cnt) for lvl, cnt in low_clusters if lvl < price],
            key=lambda x: x[0], reverse=True,
        )

        nearest_above          = above[0][0] if above else None
        nearest_above_strength = above[0][1] if above else 0
        nearest_below          = below[0][0] if below else None
        nearest_below_strength = below[0][1] if below else 0

        logger.debug(
            f"[LIQ MAP] price={price:.5g} | "
            f"above={len(above)} clusters (nearest={nearest_above:.5g} str={nearest_above_strength}) | "
            f"below={len(below)} clusters (nearest={nearest_below:.5g} str={nearest_below_strength})"
            if nearest_above and nearest_below else
            f"[LIQ MAP] price={price:.5g} | above={len(above)} | below={len(below)}"
        )

        return dict(
            above=above, below=below,
            nearest_above=nearest_above,
            nearest_below=nearest_below,
            nearest_above_strength=nearest_above_strength,
            nearest_below_strength=nearest_below_strength,
        )

    except Exception as e:
        logger.debug(f"[LIQ MAP] {e}")
        return NONE


def liq_map_clear_to_entry(
    liq_map: dict,
    entry: float,
    sl_price: float,
    direction: str,
) -> tuple[bool, str]:
    """
    Return (clear, reason).
    Blocks entry if an unswept stop cluster sits between entry and SL.
    Unchanged from v1.
    """
    if direction == "long":
        danger = [
            (lvl, cnt) for lvl, cnt in liq_map.get("below", [])
            if sl_price < lvl < entry
        ]
        if danger:
            worst = max(danger, key=lambda x: x[0])
            return False, (
                f"Unswept sell-side liquidity at {worst[0]:.5g} "
                f"(strength {worst[1]}) between SL and entry — skip"
            )
    else:
        danger = [
            (lvl, cnt) for lvl, cnt in liq_map.get("above", [])
            if entry < lvl < sl_price
        ]
        if danger:
            worst = min(danger, key=lambda x: x[0])
            return False, (
                f"Unswept buy-side liquidity at {worst[0]:.5g} "
                f"(strength {worst[1]}) between entry and SL — skip"
            )

    return True, "clear"
