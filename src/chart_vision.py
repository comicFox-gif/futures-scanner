"""
Chart Vision — Claude visual confirmation gate
----------------------------------------------
Generates a 4H candlestick chart from OHLCV data, annotates key liquidity
levels, then asks Claude to visually confirm the setup before a signal fires.

Usage:
    from src.chart_vision import ChartVision
    vision = ChartVision(api_key=os.environ["ANTHROPIC_API_KEY"])
    ok, reason = vision.confirm(h4_df, symbol, direction, liq_trigger)
"""
from __future__ import annotations

import io
import base64
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger("futures_bot.vision")

# How many 4H candles to show in the chart (last N bars)
_CHART_CANDLES = 80


def _encode_chart(df: pd.DataFrame, symbol: str, direction: str, liq: dict) -> Optional[str]:
    """
    Draw a candlestick chart with EQL/EQH levels marked.
    Returns base64-encoded PNG or None on failure.
    """
    try:
        import mplfinance as mpf
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend

        plot_df = df.iloc[-_CHART_CANDLES:][["open", "high", "low", "close", "volume"]].copy()
        plot_df.index = pd.to_datetime(plot_df.index)

        # Collect horizontal level lines
        eql = liq.get("eql_levels", [])
        eqh = liq.get("eqh_levels", [])
        hline_vals   = eql + eqh
        hline_colors = ["#ff4444"] * len(eql) + ["#44ff44"] * len(eqh)

        hlines_cfg = dict(
            hlines=hline_vals,
            colors=hline_colors,
            linestyle="--",
            linewidths=0.8,
            alpha=0.85,
        ) if hline_vals else {}

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            gridstyle=":",
            gridcolor="#333333",
            facecolor="#0d0d0d",
            figcolor="#0d0d0d",
            edgecolor="#333333",
            rc={"axes.labelcolor": "#cccccc", "xtick.color": "#cccccc", "ytick.color": "#cccccc"},
        )

        buf = io.BytesIO()
        mpf.plot(
            plot_df,
            type="candle",
            volume=True,
            style=style,
            title=f"  {symbol}  4H  |  {direction.upper()} setup",
            figsize=(14, 8),
            tight_layout=True,
            savefig=dict(fname=buf, format="png", dpi=120, bbox_inches="tight"),
            **hlines_cfg,
        )
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception as e:
        logger.warning(f"[VISION] Chart generation failed: {e}")
        return None


_PROMPT_TEMPLATE = """\
You are a professional crypto trader analyzing a 4H candlestick chart.

Symbol: {symbol}
Proposed trade direction: {direction_upper}
Red dashed lines = Equal Lows (EQL) — sell-side liquidity
Green dashed lines = Equal Highs (EQH) — buy-side liquidity

Analyze this chart and answer ONLY these questions:

1. Has a liquidity sweep completed?
   - For LONG: did price wick BELOW an EQL level and CLOSE back above it?
   - For SHORT: did price wick ABOVE an EQH level and CLOSE back below it?

2. Is market structure clean for this {direction_upper}?
   (Higher highs / higher lows for long. Lower highs / lower lows for short.)

3. Is there obvious unswept liquidity sitting directly between current price
   and the next logical target that would stop this trade?

Reply with exactly one of:
  CONFIRM — sweep complete, structure supports {direction_upper}, path is clear
  REJECT  — sweep not complete, structure wrong, or major obstacle in path
  NEUTRAL — chart is ambiguous, cannot confirm or deny

Then write ONE sentence explaining your decision."""


class ChartVision:
    def __init__(self, api_key: str | None = None):
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._key)
            except ImportError:
                logger.error("[VISION] anthropic package not installed — pip install anthropic")
                raise
        return self._client

    def confirm(
        self,
        h4_df: pd.DataFrame,
        symbol: str,
        direction: str,
        liq_trigger: dict,
    ) -> tuple[bool, str]:
        """
        Generate chart + ask Claude to visually confirm the setup.

        Returns:
            (confirmed: bool, reason: str)
        """
        if not self._key:
            logger.warning("[VISION] No ANTHROPIC_API_KEY — skipping visual gate")
            return True, "Vision skipped (no API key)"

        img_b64 = _encode_chart(h4_df, symbol, direction, liq_trigger)
        if img_b64 is None:
            return True, "Vision skipped (chart error)"

        prompt = _PROMPT_TEMPLATE.format(
            symbol=symbol,
            direction_upper=direction.upper(),
        )

        try:
            client = self._get_client()
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = resp.content[0].text.strip()
            first = text.split()[0].upper().rstrip("—-.,")
            confirmed = first == "CONFIRM"
            logger.info(f"[VISION] {symbol} {direction.upper()} → {text[:120]}")
            return confirmed, text

        except Exception as e:
            logger.warning(f"[VISION] Claude API call failed: {e}")
            return True, f"Vision skipped ({e})"
