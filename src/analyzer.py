from __future__ import annotations

"""Facade module — re-exports public symbols from scoring, market, and report.

After the Phase 3 refactor, the scoring engine, market modifier, and report
formatter each live in their own module.  This file keeps the original
import paths working so that existing consumers (e.g. forecast.py) do not
need to change their import statements.
"""

import json
import logging
import math
import os

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (kept here — it has no dependency on scoring/market/report)
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load config.json from the project root.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Re-export from scoring
# ---------------------------------------------------------------------------

from .scoring import (  # noqa: E402
    _clamp,
    _safe,
    _classify_trend,
    score_to_signal,
    calculate_signal,
    calculate_stock_score,
    SIGNAL_RATINGS,
)

# ---------------------------------------------------------------------------
# Re-export from market
# ---------------------------------------------------------------------------

from .market import (  # noqa: E402
    classify_index_trend,
    calculate_market_modifier,
    INDEX_NAMES,
)

# ---------------------------------------------------------------------------
# Re-export from report
# ---------------------------------------------------------------------------

from .report import format_report  # noqa: E402


# ---------------------------------------------------------------------------
# Key levels, position advice, risk alerts (kept here)
# ---------------------------------------------------------------------------


def calculate_key_levels(df: pd.DataFrame) -> dict:
    """Calculate support and resistance levels with validity markers."""
    last = df.iloc[-1]
    close = _safe(last.get("close"))
    support = []
    resistance = []

    ma5 = _safe(last.get("ma5"))
    if ma5 is not None and close is not None:
        if close >= ma5 and (close - ma5) / ma5 <= 0.02:
            support.append(("MA5 ✓", ma5))
        else:
            support.append(("MA5", ma5))

    ma10 = _safe(last.get("ma10"))
    if ma10 is not None and close is not None:
        if close >= ma10 and (close - ma10) / ma10 <= 0.02:
            support.append(("MA10 ✓", ma10))
        else:
            support.append(("MA10", ma10))

    ma20 = _safe(last.get("ma20"))
    if ma20 is not None and close is not None:
        if close >= ma20:
            support.append(("MA20 ✓", ma20))
        else:
            support.append(("MA20", ma20))

    boll_lower = _safe(last.get("boll_lower"))
    if boll_lower is not None:
        support.append(("布林下轨", boll_lower))

    boll_upper = _safe(last.get("boll_upper"))
    if boll_upper is not None:
        resistance.append(("布林上轨", boll_upper))

    recent_high = df["high"].tail(20).max()
    resistance.append(("20日高点", recent_high))

    return {"support": support, "resistance": resistance}


def calculate_position_advice(score: int, key_levels: dict) -> str:
    """Calculate suggested position size based on composite score."""
    if score >= 50:
        pct = "60%"
        signal = "强烈买入"
    elif score >= 15:
        pct = "40%"
        signal = "买入信号"
    elif score >= -14:
        pct = "20%"
        signal = "中性"
    elif score >= -49:
        pct = "10%"
        signal = "卖出信号"
    else:
        pct = "0%"
        signal = "强烈卖出"

    return f"建议仓位: {pct} ({signal})"


def generate_risk_alerts(df: pd.DataFrame, score: int) -> list[str]:
    """Generate risk alert messages based on indicator extremes and score."""
    alerts = []
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    rsi = _safe(latest.get("rsi"))
    if rsi is not None and rsi > 65:
        alerts.append(f"RSI={rsi:.1f} 接近超买，注意回调风险")
    elif rsi is not None and rsi < 35:
        alerts.append(f"RSI={rsi:.1f} 接近超卖，可能反弹")

    close = _safe(latest.get("close"))
    boll_upper = _safe(latest.get("boll_upper"))
    boll_lower = _safe(latest.get("boll_lower"))
    if all(v is not None for v in (close, boll_upper, boll_lower)):
        band_range = boll_upper - boll_lower
        if band_range > 0:
            boll_position = (close - boll_lower) / band_range
            if boll_position > 0.85:
                alerts.append("接近布林上轨，上方空间有限")
            elif boll_position < 0.15:
                alerts.append("接近布林下轨，注意支撑破位")

    if prev is not None:
        prev_close = _safe(prev.get("close"))
        if all(v is not None for v in (close, prev_close)) and prev_close != 0:
            daily_change = (close - prev_close) / prev_close * 100
            if abs(daily_change) > 5:
                alerts.append(f"日涨跌幅较大 ({daily_change:.1f}%)，注意波动")

    if score >= 40:
        alerts.append("多头得分较高 — 已有持仓可考虑部分获利了结")
    elif score <= -40:
        alerts.append("得分偏低 — 避免加仓，等待反转信号")

    return alerts
