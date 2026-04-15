"""
Advanced Confluence Detectors
-------------------------------
All functions are non-blocking and fail-safe — return neutral defaults on any error.
Designed to add extra confluence layers on top of the existing Elite 4H BOS strategy.

Detectors:
  get_kill_zone()            → current UTC session + AMD phase + score
  detect_wyckoff(df)         → Wyckoff phase/event + score
  detect_liquidity(df)       → EQH / EQL sweep + stop hunt + score
  detect_mmm(df)             → Market Maker Model manipulation phase + score
  detect_vsa(df)             → Volume Spread Analysis signal + score
  detect_delta_divergence(df) → price vs volume-delta alignment
  get_intermarket_score()    → DXY + SPX + BTC.D + ETH/BTC + F&G macro score
"""

from __future__ import annotations
import logging
import time as _time
from datetime import datetime

import pandas as pd

logger = logging.getLogger("futures_bot.advanced")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Kill Zone + AMD Phase
# ══════════════════════════════════════════════════════════════════════════════

def get_kill_zone() -> dict:
    """
    Return the active session/kill-zone for the current UTC time.

    kill_zones:
      London          07:00–09:00 UTC  → score +1
      NY              12:00–14:00 UTC  → score +1
      London Close    15:00–17:00 UTC  → score +1
      Asian           23:00–01:00 UTC  → score +0 (lower prob)

    AMD phase:
      00–08 UTC  Accumulation
      08–12 UTC  Manipulation  (London fake move)
      12–17 UTC  Distribution  (NY real move)

    Returns dict:
      active     bool
      name       str   ('london' | 'ny' | 'london_close' | 'asian' | 'no_trade')
      score      int   (0 or 1)
      label      str
      amd_phase  str   ('accumulation' | 'manipulation' | 'distribution' | 'none')
      amd_score  int   (0, 1, or 2)
    """
    now = datetime.utcnow()
    h   = now.hour
    wd  = now.weekday()   # 0=Mon … 6=Sun

    # Weekend blocks
    if wd == 4 and h >= 17:
        return _kz(False, "weekend", 0, "⏸ Weekend — No Trade Zone", "none", 0)
    if wd == 6 and h < 22:
        return _kz(False, "sunday", 0, "⏸ Sunday — No Trade Zone", "none", 0)

    # AMD phase
    if 0 <= h < 8:
        amd = "accumulation"; amd_score = 0
    elif 8 <= h < 12:
        amd = "manipulation"; amd_score = 0
    elif 12 <= h < 17:
        amd = "distribution"
        amd_score = 2 if 12 <= h < 14 else 1   # NY open = best window
    else:
        amd = "none"; amd_score = 0

    # Session windows
    if 7 <= h < 9:
        return _kz(True,  "london",       1, "⏰ London Kill Zone 07–09 UTC ✅",   amd, amd_score)
    if 12 <= h < 14:
        return _kz(True,  "ny",           1, "⏰ NY Kill Zone 12–14 UTC ✅",       amd, amd_score)
    if 14 <= h < 17:
        return _kz(True,  "ny_ext",       1, "⏰ NY Session 14–17 UTC",            amd, amd_score)
    if 15 <= h < 17:
        return _kz(True,  "london_close", 1, "⏰ London Close 15–17 UTC",          amd, amd_score)
    if h >= 23 or h < 1:
        return _kz(True,  "asian",        0, "⏰ Asian Kill Zone 23–01 UTC",       amd, 0)

    # No-trade zone
    return _kz(False, "no_trade", 0, f"⏸ No Trade Zone ({h:02d}:00 UTC) ⏸", "none", 0)


def _kz(active, name, score, label, amd_phase, amd_score):
    return dict(
        active=active, name=name, score=score, label=label,
        amd_phase=amd_phase, amd_score=amd_score,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Wyckoff Phase Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_wyckoff(df: pd.DataFrame, lookback: int = 80) -> dict:
    """
    Detect Wyckoff accumulation/distribution phases and key events.

    Events and their scores:
      spring      (accumulation) +3
      sos         (sign of strength) +2
      secondary_test +1
      upthrust    (distribution) +3
      sow         (sign of weakness) +2
      bc_detected  0  (distribution phase identified, no event yet)
      sc_detected  0  (accumulation phase identified, no event yet)
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

        # Ten candles before peak — used to gauge trend direction
        lookback_start = max(0, peak_idx - 10)

        # ── Selling Climax (SC) ────────────────────────────────────────────
        sc_close_ratio = (pk_close - pk_low) / max(pk_rng, 1e-10)
        is_downtrend   = pk_close < float(close[lookback_start])
        is_sc = (
            peak_vol > vol_avg * 2.0
            and sc_close_ratio > 0.5     # closes near high after a big drop
            and is_downtrend
            and pk_open > pk_close       # was a bearish candle
        )

        # ── Buying Climax (BC) ────────────────────────────────────────────
        bc_close_ratio = (pk_high - pk_close) / max(pk_rng, 1e-10)
        is_uptrend     = pk_close > float(close[lookback_start])
        is_bc = (
            peak_vol > vol_avg * 2.0
            and bc_close_ratio > 0.5     # closes near low of the move
            and is_uptrend
            and pk_open < pk_close       # was a bullish candle into the climax
        )

        sc_low  = float(low[peak_idx])  if is_sc else None
        bc_high = float(high[peak_idx]) if is_bc else None

        # ── ACCUMULATION events ───────────────────────────────────────────
        if is_sc:
            # Automatic Rally high (first bounce after SC)
            post_sc  = win.iloc[peak_idx + 1: cur_idx]
            ar_high  = float(post_sc["high"].max()) if len(post_sc) >= 2 else None

            # Spring: breaks below SC low on low volume, closes back above
            if sc_low and cur_low < sc_low and cur_close > sc_low:
                if cur_vol / vol_avg < 1.5:
                    return dict(phase="accumulation", event="spring", score=3,
                                label="Wyckoff: ACCUMULATION - Spring Detected 🌱 +3")

            # Secondary Test: near SC zone, lower volume than SC
            if sc_low and abs(cur_close - sc_low) / sc_low < 0.02:
                if cur_vol < peak_vol * 0.7:
                    return dict(phase="accumulation", event="secondary_test", score=1,
                                label="Wyckoff: ACCUMULATION - Secondary Test +1")

            # Sign of Strength: breaks above AR level with high volume
            if ar_high and cur_close > ar_high and cur_vol > vol_avg * 1.5:
                return dict(phase="accumulation", event="sos", score=2,
                            label="Wyckoff: ACCUMULATION - Sign of Strength ✅ +2")

            return dict(phase="accumulation", event="sc_detected", score=0,
                        label="Wyckoff: ACCUMULATION phase")

        # ── DISTRIBUTION events ───────────────────────────────────────────
        if is_bc:
            # Upthrust: breaks above BC high on low volume, closes back below
            if bc_high and cur_high > bc_high and cur_close < bc_high:
                if cur_vol / vol_avg < 1.5:
                    return dict(phase="distribution", event="upthrust", score=3,
                                label="Wyckoff: DISTRIBUTION - Upthrust Detected ⬆️ +3")

            # Sign of Weakness: breaks below recent support with high volume
            post_bc      = win.iloc[peak_idx + 1: cur_idx]
            recent_low   = float(post_bc["low"].min()) if len(post_bc) >= 2 else pk_low
            if cur_close < recent_low and cur_vol > vol_avg * 1.5:
                return dict(phase="distribution", event="sow", score=2,
                            label="Wyckoff: DISTRIBUTION - Sign of Weakness ✅ +2")

            return dict(phase="distribution", event="bc_detected", score=0,
                        label="Wyckoff: DISTRIBUTION phase")

    except Exception as e:
        logger.debug(f"[WYCKOFF] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Equal Highs / Equal Lows + Stop Hunt Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_liquidity(
    df: pd.DataFrame,
    lookback: int = 50,
    threshold_pct: float = 0.003,
) -> dict:
    """
    Detect Equal Highs (EQH), Equal Lows (EQL), and stop hunts.

    Scores:
      EQH swept (price spiked above, closed below)   → +2
      EQL swept (price spiked below, closed above)   → +2
      Stop hunt (spike + reversal + volume)           → +2
    """
    NONE = dict(
        eqh_levels=[], eql_levels=[],
        eqh_swept=False, eql_swept=False, stop_hunt=False,
        score=0, label=[],
    )
    try:
        if len(df) < lookback + 5:
            return NONE

        win   = df.iloc[-(lookback + 5):-2]    # confirmed history
        cur   = df.iloc[-2]
        prev  = df.iloc[-3]

        highs = win["high"].values
        lows  = win["low"].values
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
        score  = 0
        eqh_swept = False
        eql_swept = False
        stop_hunt = False

        # EQH swept: current wick above EQH but close below it
        for lvl in eqh_levels:
            if cur_high > lvl * (1 + threshold_pct * 0.3) and cur_close < lvl:
                eqh_swept = True
                score += 2
                labels.append(f"Liquidity: EQH Swept ${lvl:,.4g} +2 💧")
                break

        # EQL swept: current wick below EQL but close above it
        for lvl in eql_levels:
            if cur_low < lvl * (1 - threshold_pct * 0.3) and cur_close > lvl:
                eql_swept = True
                score += 2
                labels.append(f"Liquidity: EQL Swept ${lvl:,.4g} +2 💧")
                break

        # Stop hunt: spike beyond previous candle's extreme then reversal
        # Bullish stop hunt (swept lows → long)
        if (cur_low < prev_low * (1 - threshold_pct)
                and cur_close > prev_low
                and cur_vol > vol_avg * 0.9):
            stop_hunt = True
            score += 2
            labels.append("Liquidity: Sellside Stop Hunt ✅ +2")
        # Bearish stop hunt (swept highs → short)
        elif (cur_high > prev_high * (1 + threshold_pct)
              and cur_close < prev_high
              and cur_vol > vol_avg * 0.9):
            stop_hunt = True
            score += 2
            labels.append("Liquidity: Buyside Stop Hunt ✅ +2")

        return dict(
            eqh_levels=eqh_levels, eql_levels=eql_levels,
            eqh_swept=eqh_swept,   eql_swept=eql_swept,
            stop_hunt=stop_hunt,   score=score, label=labels,
        )

    except Exception as e:
        logger.debug(f"[LIQUIDITY] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Market Maker Model (MMM)
# ══════════════════════════════════════════════════════════════════════════════

def detect_mmm(df: pd.DataFrame) -> dict:
    """
    Detect Market Maker Model phases on 4H candles.

    Consolidation → Manipulation (fake break) → Distribution (real move).
    Scores:
      Manipulation below range (long setup)  +3
      Manipulation above range (short setup) +3
      Consolidation only                      0
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

        # Consolidation zone: last 3–6 candles
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
        # Price broke below c_low on moderate volume, closed back inside
        if cur_low < c_low and cur_close > c_low and cur_vol < vol_avg * 1.3:
            return dict(
                phase="manipulation", manipulation=True, direction="long",
                score=3,
                label=f"MMM: Manipulation ↓ below range — Real move UP expected 🎯 +3",
            )

        # Manipulation UP → expect SHORT
        # Price broke above c_high on moderate volume, closed back inside
        if cur_high > c_high and cur_close < c_high and cur_vol < vol_avg * 1.3:
            return dict(
                phase="manipulation", manipulation=True, direction="short",
                score=3,
                label=f"MMM: Manipulation ↑ above range — Real move DOWN expected 🎯 +3",
            )

        return dict(
            phase="consolidation", manipulation=False, direction=None,
            score=0,
            label=f"MMM: Consolidation {c_low:.5g}–{c_high:.5g}",
        )

    except Exception as e:
        logger.debug(f"[MMM] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Volume Spread Analysis (VSA)
# ══════════════════════════════════════════════════════════════════════════════

def detect_vsa(df: pd.DataFrame) -> dict:
    """
    Analyse price spread vs volume for institutional footprints.

    Bullish signals (positive score):
      no_supply          +2   narrow spread, low vol, close upper half
      stopping_volume    +2   very high vol, down candle, closes near high
      test_bar           +1   low vol down candle, closes near high

    Bearish signals (positive score for shorts):
      no_demand          +2   narrow spread, low vol, up candle, weak close
      climactic          +2   extreme vol, wide spread, closes near low
      upthrust_bar       +2   down close after vol spike above resistance

    Warning (negative score — reduce conviction):
      effort_no_result   -1  high vol but tiny body (selling into move)
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
        close_pos = (c_cls - c_low) / max(spread, 1e-10)  # 0=bottom, 1=top
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

        # ── Warning ───────────────────────────────────────────────────────

        if c_vol > vol_avg * 1.8 and body < spread * 0.3 and is_bull:
            return dict(signal="effort_no_result", score=-1,
                        label="VSA: Effort/No Result — selling into move ⚠️ -1", bullish=False)

    except Exception as e:
        logger.debug(f"[VSA] {e}")

    return NONE


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Delta Divergence (proxy via close-position volume)
# ══════════════════════════════════════════════════════════════════════════════

def detect_delta_divergence(df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Volume-delta proxy: compare cumulative buy/sell pressure vs price direction.
    True tick delta isn't available; we approximate from close position within range.
    Divergence = price rising but delta negative (or vice versa) → warning only.
    """
    NONE = dict(divergence=False, price_dir=0, delta_dir=0, score=0,
                label="Delta: N/A")
    try:
        if len(df) < lookback + 3:
            return NONE

        win   = df.iloc[-(lookback + 3):-2]
        cur   = df.iloc[-2]

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
            score=0,   # informational only — no score awarded
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
# 7.  Intermarket Analysis (DXY, SPX, BTC.D, ETH/BTC, Fear&Greed)
# ══════════════════════════════════════════════════════════════════════════════

_im_cache: dict  = {}
_im_ts:    float = 0.0
_IM_TTL            = 3600.0   # 1-hour cache


def get_intermarket_score(exchange=None, direction: str = "long") -> dict:
    """
    Fetch and score macro intermarket factors for the given trade direction.

    Bullish alignment (for longs):  DXY below MA20, SPX rising, BTC.D < 55%,
                                     ETH/BTC rising, Fear&Greed < 50.
    Bearish alignment (for shorts): inverse of the above.

    Scores:
      3+ factors aligned  → +1
      all 5 aligned       → +2
    Returns dict with score, label, and per-factor breakdown.
    """
    global _im_cache, _im_ts

    if _time.time() - _im_ts < _IM_TTL and _im_cache:
        raw = _im_cache
    else:
        raw = _fetch_intermarket(exchange)
        _im_cache = raw
        _im_ts    = _time.time()

    # Factor booleans: True = bullish for crypto
    checks = {
        "DXY":    raw.get("dxy_bearish"),    # below MA = bullish for crypto
        "SPX":    raw.get("spx_bullish"),
        "BTC.D":  raw.get("btcd_low"),       # <55% = alt friendly
        "ETH/BTC": raw.get("ethbtc_rising"),
        "F/G":    raw.get("fg_bullish"),     # fear = good for longs
    }

    if direction == "short":
        # Invert each boolean
        checks = {k: (not v if v is not None else None)
                  for k, v in checks.items()}

    filled  = {k: v for k, v in checks.items() if v is not None}
    n_total = len(filled)
    n_align = sum(1 for v in filled.values() if v)

    if n_total >= 5 and n_align == 5:
        score = 2
    elif n_total >= 3 and n_align >= 3:
        score = 1
    else:
        score = 0

    # Build label lines
    tag = {True: "✅", False: "❌", None: "—"}
    lines = [f"  {k}: {tag[v]}" for k, v in checks.items()]
    score_sfx = f" +{score}" if score else ""
    label = (
        f"Intermarket: {n_align}/{n_total} aligned{score_sfx}\n"
        + "\n".join(lines)
    )

    return dict(score=score, label=label, n_align=n_align, n_total=n_total,
                checks=checks)


def _fetch_intermarket(exchange=None) -> dict:
    """Internal: fetch raw intermarket data. All failures return None fields."""
    data: dict = {
        "dxy_bearish":  None,
        "spx_bullish":  None,
        "btcd_low":     None,
        "ethbtc_rising": None,
        "fg_bullish":   None,
    }

    # ── DXY & SPX via yfinance ────────────────────────────────────────────
    try:
        import yfinance as yf
        dxy = yf.download("DX-Y.NYB", period="30d", interval="1d",
                          progress=False, auto_adjust=True)
        if dxy is not None and len(dxy) >= 21:
            ma20 = float(dxy["Close"].rolling(20).mean().iloc[-1])
            last = float(dxy["Close"].iloc[-1])
            data["dxy_bearish"] = last < ma20
    except Exception as e:
        logger.debug(f"[INTERMARKET] DXY fetch: {e}")

    try:
        import yfinance as yf
        spx = yf.download("^GSPC", period="5d", interval="1d",
                          progress=False, auto_adjust=True)
        if spx is not None and len(spx) >= 2:
            data["spx_bullish"] = float(spx["Close"].iloc[-1]) > float(spx["Close"].iloc[-2])
    except Exception as e:
        logger.debug(f"[INTERMARKET] SPX fetch: {e}")

    # ── BTC Dominance via CoinGecko (free) ───────────────────────────────
    try:
        import requests as _req
        resp = _req.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=6,
            headers={"User-Agent": "futures-bot/1.0"},
        )
        if resp.status_code == 200:
            btcd = resp.json().get("data", {}).get(
                "market_cap_percentage", {}
            ).get("btc")
            if btcd is not None:
                data["btcd_low"] = float(btcd) < 55.0
    except Exception as e:
        logger.debug(f"[INTERMARKET] BTC.D fetch: {e}")

    # ── ETH/BTC ratio via ccxt exchange ──────────────────────────────────
    if exchange is not None:
        try:
            ticker = exchange.fetch_ticker("ETH/BTC")
            eth_now = float(ticker.get("last") or 0)
            ohlcv   = exchange.fetch_ohlcv("ETH/BTC", "1d", limit=3)
            if ohlcv and len(ohlcv) >= 2:
                prev = float(ohlcv[-2][4])
                data["ethbtc_rising"] = eth_now > prev
        except Exception as e:
            logger.debug(f"[INTERMARKET] ETH/BTC fetch: {e}")

    # ── Fear & Greed ──────────────────────────────────────────────────────
    try:
        from src.sentiment import get_fear_greed
        fg = get_fear_greed()
        data["fg_bullish"] = fg["value"] < 50
    except Exception as e:
        logger.debug(f"[INTERMARKET] F&G fetch: {e}")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Position sizing by confluence score
# ══════════════════════════════════════════════════════════════════════════════

def risk_usdt_for_score(score: int) -> float:
    """
    Return the dollar risk amount based on the 20-point confluence score.

    5–7   → $5
    8–11  → $7
    12–14 → $10
    15–17 → $12
    18–20 → $15
    """
    if score >= 18:  return 15.0
    if score >= 15:  return 12.0
    if score >= 12:  return 10.0
    if score >= 8:   return 7.0
    return 5.0


def tp_rr_for_score(score: int) -> float:
    """
    Minimum TP RR based on confluence score.

    5–7   → 2:1
    8–11  → 3:1
    12+   → 5:1
    """
    if score >= 12:  return 5.0
    if score >= 8:   return 3.0
    return 2.0
