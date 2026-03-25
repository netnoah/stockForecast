from __future__ import annotations

"""Market modifier calculation based on broad index trends."""

import logging
from datetime import datetime

import pandas as pd

from data_source import get_index_history, get_index_realtime_quote
from indicators import calc_ma, calc_macd
from scoring import _safe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_NAMES = {
    "sh000001": "上证综指",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000905": "中证500",
}

_TREND_IDX_BULLISH = "bullish"
_TREND_IDX_BEARISH = "bearish"
_TREND_IDX_NEUTRAL = "neutral"

_TREND_IDX_LABELS = {
    _TREND_IDX_BULLISH: "看涨",
    _TREND_IDX_BEARISH: "看跌",
    _TREND_IDX_NEUTRAL: "中性",
}


# ---------------------------------------------------------------------------
# Index trend classification
# ---------------------------------------------------------------------------


def classify_index_trend(df: pd.DataFrame) -> tuple[str, float]:
    """Classify the trend of a market index with strength score.

    Returns:
        (trend, strength) where trend is one of _TREND_IDX_* constants
        and strength is a float in [-1, +1].
    """
    if len(df) < 6:
        return (_TREND_IDX_NEUTRAL, 0.0)

    latest = df.iloc[-1]
    close = _safe(latest.get("close"))
    ma20 = _safe(latest.get("ma20"))
    dif = _safe(latest.get("dif"))
    dea = _safe(latest.get("dea"))

    if any(v is None for v in (close, ma20, dif, dea)):
        return (_TREND_IDX_NEUTRAL, 0.0)

    # --- Mid-term trend (lagging, stable direction) ---
    bias = (close - ma20) / ma20
    bias_score = max(-1.0, min(1.0, bias * 20))

    macd_score = 0.0
    macd_gap = (dif - dea) / close * 1000
    macd_score = max(-1.0, min(1.0, macd_gap * 2))

    slope_score = 0.0
    if len(df) >= 25:
        ma20_5ago = _safe(df.iloc[-6].get("ma20"))
        if ma20_5ago is not None and ma20_5ago != 0:
            slope = (ma20 - ma20_5ago) / ma20_5ago
            slope_score = max(-1.0, min(1.0, slope * 50))

    mid_term = bias_score * 0.4 + macd_score * 0.35 + slope_score * 0.25

    # --- Short-term momentum ---
    momentum = 0.0
    close_1d = _safe(df.iloc[-2].get("close")) if len(df) >= 2 else None
    close_2d = _safe(df.iloc[-3].get("close")) if len(df) >= 3 else None

    if close_1d is not None and close_1d > 0:
        m1d = (close - close_1d) / close_1d
        momentum += max(-1.0, min(1.0, m1d * 50)) * 0.5

    if close_2d is not None and close_2d > 0:
        m2d = (close - close_2d) / close_2d
        momentum += max(-1.0, min(1.0, m2d * 25)) * 0.5

    strength = mid_term * 0.65 + momentum * 0.35
    strength = max(-1.0, min(1.0, strength))

    if strength > 0.15:
        return (_TREND_IDX_BULLISH, strength)
    elif strength < -0.15:
        return (_TREND_IDX_BEARISH, strength)
    return (_TREND_IDX_NEUTRAL, strength)


# ---------------------------------------------------------------------------
# Market modifier
# ---------------------------------------------------------------------------


def calculate_market_modifier(config: dict, intraday: bool = False) -> tuple[int, list[dict]]:
    """Calculate a market-wide modifier based on broad index trends.

    Returns:
        (modifier, results) where:
        - modifier: Integer adjustment in [-max_impact, +max_impact].
        - results: List of dicts with keys: code, name, trend, strength.
    """
    modifier_config = config.get("market_modifier", {})
    enabled = modifier_config.get("enabled", False)
    max_impact = modifier_config.get("max_impact", 15)
    index_codes = modifier_config.get("indices", [])

    if not enabled or not index_codes:
        return (0, [])

    results = []
    for code in index_codes:
        try:
            index_df = get_index_history(code)
            if intraday:
                rt = get_index_realtime_quote(code)
                if rt and rt.get("close") is not None:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    last_date = str(index_df.iloc[-1]["date"])[:10]
                    if last_date != today_str:
                        new_row = {
                            "date": today_str,
                            "open": rt["open"],
                            "high": rt["high"],
                            "low": rt["low"],
                            "close": rt["close"],
                            "volume": rt.get("volume", 0),
                        }
                        index_df = pd.concat(
                            [index_df, pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
            index_df = calc_ma(index_df)
            index_df = calc_macd(index_df)
            trend, strength = classify_index_trend(index_df)

            if intraday and len(index_df) >= 2:
                prev_close = _safe(index_df.iloc[-2].get("close"))
                curr_close = _safe(index_df.iloc[-1].get("close"))
                if prev_close is not None and curr_close is not None and prev_close > 0:
                    day_change = (curr_close - prev_close) / prev_close
                    if abs(day_change) > 0.01:
                        momentum = max(-1.0, min(1.0, day_change * 15))
                        weight = min(0.2 + abs(day_change) * 15, 0.6)
                        strength = max(-1.0, min(1.0, strength * (1 - weight) + momentum * weight))
                        if strength > 0.15:
                            trend = _TREND_IDX_BULLISH
                        elif strength < -0.15:
                            trend = _TREND_IDX_BEARISH
                        else:
                            trend = _TREND_IDX_NEUTRAL

            name = INDEX_NAMES.get(code, code)
            results.append({"code": code, "name": name, "trend": trend, "strength": strength})
        except Exception:
            logger.warning("Failed to fetch/calculate index trend for %s, defaulting to neutral", code)
            results.append({"code": code, "name": INDEX_NAMES.get(code, code), "trend": _TREND_IDX_NEUTRAL, "strength": 0.0})

    total_strength = sum(r["strength"] for r in results)
    avg_strength = total_strength / len(results) if results else 0.0

    modifier = int(avg_strength * max_impact)
    modifier = max(-max_impact, min(max_impact, modifier))

    logger.info("Market modifier: %d (indices: %s)", modifier,
                ", ".join(f"{r['name']}={r['trend']}({r['strength']:+.2f})" for r in results))

    return (modifier, results)
