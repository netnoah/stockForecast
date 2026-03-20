from __future__ import annotations

"""
Scoring engine for A-Share stock quantitative analyzer.

Provides weighted indicator evaluation that produces a composite score
and signal rating. Each indicator is scored independently in [-20, +20],
then combined using configurable weights from config.json.
"""

import json
import math
import os
from datetime import datetime

import pandas as pd

from data_source import get_index_history
from indicators import calc_ma, calc_macd


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_BULLISH = "\033[31m"   # 红色 = A股涨
_BEARISH = "\033[32m"   # 绿色 = A股跌
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"


def _signal_color(signal: str) -> str:
    """Return ANSI color for a signal label."""
    if signal in ("强烈买入",):
        return _BULLISH + _BOLD + signal + _RST
    if signal in ("买入",):
        return _BULLISH + _BOLD + signal + _RST
    if signal in ("卖出",):
        return _BEARISH + signal + _RST
    if signal in ("强烈卖出",):
        return _BEARISH + signal + _RST
    return _WHITE + signal + _RST


def _score_color(score: int) -> str:
    """Return ANSI color for a numeric score."""
    if score >= 15:
        return _BULLISH + _BOLD + str(score) + _RST
    if score <= -15:
        return _BEARISH + str(score) + _RST
    return _YELLOW + str(score) + _RST


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_RATINGS = [
    (50, 100, "强烈买入"),
    (15, 49, "买入"),
    (-14, 14, "观望"),
    (-49, -15, "卖出"),
    (-100, -50, "强烈卖出"),
]

# Backward compatibility: English signal names from old CSV data
_SIGNAL_EN_MAP = {
    "Strong Buy": "强烈买入",
    "Buy": "买入",
    "Hold": "观望",
    "Sell": "卖出",
    "Strong Sell": "强烈卖出",
}

_SCORE_MIN = -100
_SCORE_MAX = 100

INDEX_NAMES = {
    "sh000001": "上证综指",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000905": "中证500",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load config.json from the project root.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Signal mapping
# ---------------------------------------------------------------------------


def score_to_signal(score: int) -> str:
    """Convert a numeric score to a human-readable signal label.

    Args:
        score: Integer score in the range [-100, 100].

    Returns:
        Signal label. Falls back to "观望" if no range matches.
    """
    for low, high, label in SIGNAL_RATINGS:
        if low <= score <= high:
            return label
    return "观望"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    """Clamp a numeric value to the valid indicator score range [-20, +20]."""
    return max(-20.0, min(20.0, value))


def _safe(val, default=None):
    """Return the numeric value or a default if it is NaN/None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------


def score_ma(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on MA alignment, MA60 position, and price deviation.

    Continuous scoring:
    - Trend strength: (MA5 - MA20) / MA20, scaled to [-6, +6]
    - MA alignment: bonus for bullish/bearish alignment, clamped [-6, +6]
    - Price position vs MA20: (close - MA20) / MA20, scaled to [-8, +8]

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "数据不足")

    latest = df.iloc[-1]
    ma5 = _safe(latest.get("ma5"))
    ma10 = _safe(latest.get("ma10"))
    ma20 = _safe(latest.get("ma20"))
    ma60 = _safe(latest.get("ma60"))
    close = _safe(latest.get("close"))

    if any(v is None for v in (ma5, ma10, ma20, close)):
        return (0, "MA数据不完整 (NaN)")

    # 1) Trend strength: MA5 vs MA20 spread, normalized by price
    trend = (ma5 - ma20) / ma20 * 200
    trend = max(-6, min(6, trend))

    # 2) MA alignment bonus
    #    多头排列 MA5 > MA10 > MA20 > MA60 → +6
    #    空头排列 MA5 < MA10 < MA20 < MA60 → -6
    #    Mixed → proportional score
    parts = []
    has_ma60 = ma60 is not None

    if ma5 > ma10 > ma20:
        parts.append(1)
    elif ma5 < ma10 < ma20:
        parts.append(-1)
    else:
        parts.append(0)

    if has_ma60:
        if ma20 > ma60:
            parts.append(1)
        elif ma20 < ma60:
            parts.append(-1)
        else:
            parts.append(0)

    alignment = sum(parts) / len(parts) * 6
    alignment = max(-6, min(6, alignment))

    # 3) Price position vs MA20
    position = (close - ma20) / ma20 * 200
    position = max(-8, min(8, position))

    total = trend + alignment + position

    # Build reason text
    alignment_labels = {1: "多头", -1: "空头", 0: "混乱"}
    short_label = alignment_labels.get(parts[0], "混乱")
    long_label = alignment_labels.get(parts[1], "无MA60") if has_ma60 else "无MA60"
    reason = f"趋势={trend:+.1f}; 排列={short_label}/{long_label}={alignment:+.1f}; 偏离={position:+.1f}"
    return (_clamp(total), reason)


def score_macd(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on MACD momentum and histogram trend.

    Continuous scoring:
    - Momentum: (DIF - DEA) / close, scaled to [-15, +15]
    - Histogram trend: change rate of histogram, scaled to [-5, +5]

    Note: The MACD histogram column in indicators.py is named "macd".

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "数据不足")

    prev = df.iloc[-2]
    latest = df.iloc[-1]

    prev_dif = _safe(prev.get("dif"))
    prev_dea = _safe(prev.get("dea"))
    prev_hist = _safe(prev.get("macd"))
    last_dif = _safe(latest.get("dif"))
    last_dea = _safe(latest.get("dea"))
    last_hist = _safe(latest.get("macd"))
    close = _safe(latest.get("close"))

    if any(v is None for v in (prev_dif, prev_dea, prev_hist, last_dif, last_dea, last_hist, close)):
        return (0, "MACD数据不完整 (NaN)")

    # Momentum: DIF vs DEA normalized by close price
    momentum = (last_dif - last_dea) / close * 2000
    momentum = max(-15, min(15, momentum))

    # Histogram trend: rate of change
    if abs(prev_hist) > 1e-10:
        hist_trend = (last_hist - prev_hist) / abs(prev_hist) * 5
    else:
        hist_trend = 0
    hist_trend = max(-5, min(5, hist_trend))

    total = momentum + hist_trend
    reason = f"动量={momentum:+.1f}; 柱状图趋势={hist_trend:+.1f}"
    return (_clamp(total), reason)


def _rsi分段映射(rsi: float) -> float:
    """Map RSI to score using a segmented function.

    Compresses sensitivity in the neutral zone (30-70) and amplifies
    signals in extreme zones (<20, >80).

    Breakpoints:
        RSI=0 → +20   (deeply oversold)
        RSI=20 → +15  (oversold)
        RSI=30 → +8   (mildly oversold)
        RSI=50 → 0    (neutral)
        RSI=70 → -8   (mildly overbought)
        RSI=80 → -15  (overbought)
        RSI=100 → -20 (deeply overbought)
    """
    breakpoints = [
        (0, 20), (20, 15), (30, 8), (50, 0), (70, -8), (80, -15), (100, -20),
    ]
    for i in range(len(breakpoints) - 1):
        lo_rsi, lo_score = breakpoints[i]
        hi_rsi, hi_score = breakpoints[i + 1]
        if rsi <= hi_rsi:
            t = (rsi - lo_rsi) / (hi_rsi - lo_rsi) if hi_rsi != lo_rsi else 0
            return lo_score + t * (hi_score - lo_score)
    return breakpoints[-1][1]


def score_rsi(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on RSI using a segmented mapping.

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 1:
        return (0, "数据不足")

    latest = df.iloc[-1]
    rsi = _safe(latest.get("rsi"))

    if rsi is None:
        return (0, "RSI数据不完整 (NaN)")

    score = _rsi分段映射(rsi)
    return (_clamp(score), f"RSI={rsi:.1f} → {score:+.1f}")


def score_bollinger(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on Bollinger Band position and squeeze detection.

    - Position score [-14, +14]: price location within bands, scaled down
      in narrow bands (squeeze) to avoid false signals.
    - Squeeze bonus [-6, +6]: bandwidth narrowing vs 20-day average indicates
      impending breakout direction from price trend.

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "数据不足")

    latest = df.iloc[-1]
    close = _safe(latest.get("close"))
    boll_upper = _safe(latest.get("boll_upper"))
    boll_lower = _safe(latest.get("boll_lower"))
    boll_width = _safe(latest.get("boll_width"))

    if any(v is None for v in (close, boll_upper, boll_lower)):
        return (0, "布林带数据不完整 (NaN)")

    band_range = boll_upper - boll_lower
    if band_range == 0:
        return (0, "布林带宽度为零，无法计算位置")

    # 1) Position score, attenuated by squeeze
    position = (close - boll_lower) / band_range
    position_score = (0.5 - position) * 28  # max ±14

    # Squeeze: narrow band → position signals are unreliable, scale down
    if boll_width is not None and boll_width > 0:
        # Typical A-share boll_width range: 0.02-0.15
        # Below 0.04 = tight squeeze, attenuate position score
        squeeze_factor = min(boll_width / 0.04, 1.0)
        position_score *= squeeze_factor
        squeeze_label = "收窄" if squeeze_factor < 0.8 else "正常"
    else:
        squeeze_factor = 1.0
        squeeze_label = "未知"

    position_score = max(-14, min(14, position_score))

    # 2) Squeeze bonus: compare current width to 20-day average width
    squeeze_bonus = 0.0
    widths = df["boll_width"].tail(21).dropna()
    if len(widths) >= 10 and boll_width is not None:
        avg_width = widths.iloc[:-1].mean()  # exclude current day
        if avg_width > 0:
            width_ratio = boll_width / avg_width
            # width_ratio < 0.7 = significant squeeze → score based on recent
            # price direction (squeeze resolves in trend direction)
            if width_ratio < 0.7:
                prev_close = _safe(df.iloc[-2].get("close"))
                if prev_close is not None and prev_close != 0:
                    direction = 1 if close > prev_close else -1
                    intensity = (0.7 - width_ratio) / 0.7  # 0→0, 0→1
                    squeeze_bonus = direction * intensity * 6
                else:
                    squeeze_bonus = 0

    squeeze_bonus = max(-6, min(6, squeeze_bonus))

    total = position_score + squeeze_bonus
    reason = (
        f"位置={position:.2f}({squeeze_label}); "
        f"squeeze={squeeze_bonus:+.1f} → {total:+.1f}"
    )
    return (_clamp(total), reason)


def score_kdj(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on KDJ values as a continuous function.

    Continuous scoring:
    - K-D momentum: (K - D) scaled to [-10, +10]
    - J overbought/oversold: (50 - J) scaled to [-10, +10]

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "数据不足")

    latest = df.iloc[-1]

    last_k = _safe(latest.get("k"))
    last_d = _safe(latest.get("d"))
    last_j = _safe(latest.get("j"))

    if any(v is None for v in (last_k, last_d, last_j)):
        return (0, "KDJ数据不完整 (NaN)")

    # K-D momentum
    kd_momentum = (last_k - last_d) / 10
    kd_momentum = max(-10, min(10, kd_momentum))

    # J value: J can exceed [0,100] (often 110-120 or negative)
    # Use segmented mapping: compress neutral zone, amplify extremes
    # Breakpoints: J=-20→+10, J=20→+5, J=50→0, J=80→-5, J=100→-8, J=120→-10
    j_breakpoints = [
        (-20, 10), (20, 5), (50, 0), (80, -5), (100, -8), (120, -10),
    ]
    j_score = 0.0
    for i in range(len(j_breakpoints) - 1):
        lo_j, lo_s = j_breakpoints[i]
        hi_j, hi_s = j_breakpoints[i + 1]
        if last_j <= hi_j:
            t = (last_j - lo_j) / (hi_j - lo_j) if hi_j != lo_j else 0
            j_score = lo_s + t * (hi_s - lo_s)
            break
    else:
        j_score = j_breakpoints[-1][1]
    j_score = max(-10, min(10, j_score))

    total = kd_momentum + j_score
    reason = f"K-D动量={kd_momentum:+.1f}; J值={last_j:.1f}→{j_score:+.1f}"
    return (_clamp(total), reason)


def score_volume(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on volume-price relationship as a continuous function.

    score = (vol_ratio - 1) * price_direction * 20
   放量上涨正分，放量下跌负分，缩量下跌正分（卖盘衰竭），缩量上涨负分（量价背离）。

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "数据不足")

    prev = df.iloc[-2]
    latest = df.iloc[-1]

    vol_ratio = _safe(latest.get("vol_ratio"))
    prev_close = _safe(prev.get("close"))
    last_close = _safe(latest.get("close"))

    if any(v is None for v in (vol_ratio, prev_close, last_close)):
        return (0, "成交量数据不完整 (NaN)")

    # 涨跌幅过小视为平盘，不产生量价信号
    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close != 0 else 0.0
    if abs(change_pct) < 0.5:
        return (0, f"量比={vol_ratio:.2f} 平盘(涨跌{change_pct:+.2f}%) → 无信号")

    # 涨跌幅作为方向权重: |change|越大, 量价信号越可信
    direction_weight = min(abs(change_pct) / 3.0, 1.0)  # 3%涨跌幅达到满权重
    price_direction = 1 if change_pct > 0 else -1
    vol_deviation = vol_ratio - 1  # >0 = above average, <0 = below average
    score = vol_deviation * price_direction * 20 * direction_weight

    direction_text = f"{'上涨' if price_direction > 0 else '下跌'}{abs(change_pct):.1f}%"

    return (_clamp(score), f"量比={vol_ratio:.2f} {direction_text} → {score:+.1f}")


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

_INDICATOR_SCORE_FUNCS = {
    "ma": score_ma,
    "macd": score_macd,
    "rsi": score_rsi,
    "bollinger": score_bollinger,
    "kdj": score_kdj,
    "volume": score_volume,
}


def calculate_stock_score(df: pd.DataFrame, config: dict) -> tuple[int, list[dict]]:
    """Calculate a composite stock score based on weighted indicator evaluation.

    Reads indicator configuration from config.json to determine which
    indicators are enabled and their respective weights.

    Raw score = 50 + (weighted_sum / total_weight), producing a value
    in the approximate range [10, 90].

    Args:
        df: DataFrame with all indicator columns already calculated.
        config: Parsed config.json dictionary.

    Returns:
        (raw_score, results) where:
        - raw_score: Integer score in [10, 90] range.
        - results: List of dicts with keys: name, score, reason.
    """
    indicator_config = config.get("indicators", {})

    weighted_sum = 0.0
    total_weight = 0
    results = []

    for name, score_func in _INDICATOR_SCORE_FUNCS.items():
        cfg = indicator_config.get(name, {})

        if not cfg.get("enabled", False):
            continue

        weight = cfg.get("weight", 0)
        if weight <= 0:
            continue

        score, reason = score_func(df)
        weighted_sum += score * weight
        total_weight += weight

        results.append({
            "name": name,
            "score": score * 5,
            "reason": reason,
        })

    # Calculate raw score in [-100, +100] range
    if total_weight > 0:
        raw_score = (weighted_sum / total_weight) * 5
    else:
        raw_score = 0.0

    # Clamp to valid signal range
    raw_score = max(-100, min(100, raw_score))

    return (int(raw_score), results)


# ---------------------------------------------------------------------------
# Market modifier
# ---------------------------------------------------------------------------


def classify_index_trend(df: pd.DataFrame) -> str:
    """Classify the trend of a market index based on MA20 and MACD.

    Args:
        df: DataFrame with at least close, ma20, dif, dea columns.

    Returns:
        "bullish", "bearish", or "neutral".
    """
    if len(df) < 1:
        return "neutral"

    latest = df.iloc[-1]
    close = _safe(latest.get("close"))
    ma20 = _safe(latest.get("ma20"))
    dif = _safe(latest.get("dif"))
    dea = _safe(latest.get("dea"))

    if any(v is None for v in (ma20, dif, dea)):
        return "neutral"

    if close is None:
        return "neutral"

    if close > ma20 and dif > dea:
        return "看涨"
    if close < ma20 and dif < dea:
        return "看跌"
    return "中性"


def calculate_market_modifier(config: dict) -> tuple[int, list[dict]]:
    """Calculate a market-wide modifier based on broad index trends.

    Reads market_modifier config (enabled, max_impact, indices list).
    For each index, fetches history, calculates MA + MACD, classifies trend.
    Returns a modifier value and list of per-index results.

    Args:
        config: Parsed config.json dictionary.

    Returns:
        (modifier, results) where:
        - modifier: Integer adjustment in [-max_impact, +max_impact].
        - results: List of dicts with keys: code, name, trend.
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
            index_df = calc_ma(index_df)
            index_df = calc_macd(index_df)
            trend = classify_index_trend(index_df)
            name = INDEX_NAMES.get(code, code)
            results.append({"code": code, "name": name, "trend": trend})
        except Exception:
            # A failed index fetch should not crash the entire analysis
            results.append({"code": code, "name": INDEX_NAMES.get(code, code), "trend": "中性"})

    bullish_count = sum(1 for r in results if r["trend"] == "看涨")
    bearish_count = sum(1 for r in results if r["trend"] == "看跌")
    total_count = len(results)

    modifier = 0
    if bullish_count >= 3:
        ratio = bullish_count / total_count
        modifier = int(max_impact * (0.67 + 0.33 * ratio))
        modifier = min(modifier, max_impact)
    elif bullish_count == 2:
        modifier = int(max_impact * 0.35)
    elif bearish_count >= 3:
        ratio = bearish_count / total_count
        modifier = int(-max_impact * (0.67 + 0.33 * ratio))
        modifier = max(modifier, -max_impact)
    elif bearish_count == 2:
        modifier = int(-max_impact * 0.4)

    return (modifier, results)


# ---------------------------------------------------------------------------
# Key levels
# ---------------------------------------------------------------------------


def calculate_key_levels(df: pd.DataFrame) -> dict:
    """Calculate support and resistance levels from technical indicators.

    Args:
        df: DataFrame with indicator columns (ma20, boll_lower, boll_upper, high).

    Returns:
        Dict with "support" and "resistance" keys, each containing a list
        of (name, value) tuples.
    """
    last = df.iloc[-1]
    support = []
    resistance = []

    if not pd.isna(last.get("ma20")):
        support.append(("MA20", last["ma20"]))

    if not pd.isna(last.get("boll_lower")):
        support.append(("布林下轨", last["boll_lower"]))

    if not pd.isna(last.get("boll_upper")):
        resistance.append(("布林上轨", last["boll_upper"]))

    recent_high = df["high"].tail(20).max()
    resistance.append(("20日高点", recent_high))

    return {"support": support, "resistance": resistance}


# ---------------------------------------------------------------------------
# Position advice
# ---------------------------------------------------------------------------


def calculate_position_advice(score: int, key_levels: dict) -> str:
    """Calculate suggested position size based on composite score.

    Args:
        score: Integer score in [10, 90] range.
        key_levels: Dict from calculate_key_levels (unused but kept for
            future enhancement).

    Returns:
        Human-readable position advice string.
    """
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


# ---------------------------------------------------------------------------
# Risk alerts
# ---------------------------------------------------------------------------


def generate_risk_alerts(df: pd.DataFrame, score: int) -> list[str]:
    """Generate risk alert messages based on indicator extremes and score.

    Args:
        df: DataFrame with indicator columns.
        score: Final composite score (after market modifier).

    Returns:
        List of risk alert strings.
    """
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


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def _trend_icon(trend: str) -> str:
    """Return a single-character icon for a trend classification."""
    if trend == "看涨":
        return "+"
    if trend == "看跌":
        return "-"
    return "~"


def _score_icon(score: int) -> str:
    """Return a single-character icon for an indicator score."""
    if score > 0:
        return "+"
    if score < 0:
        return "-"
    return "~"


def format_report(
    symbol: str,
    stock_name: str,
    df: pd.DataFrame,
    score: int,
    indicator_results: list[dict],
    market_modifier: int,
    market_results: list[dict],
    key_levels: dict,
    risk_alerts: list[str],
    position_advice: str,
    is_intraday: bool = False,
    realtime_data: dict | None = None,
    session_label: str = "",
) -> str:
    """Format all analysis data into a human-readable text report.

    Args:
        symbol: Stock symbol code.
        df: DataFrame with all indicator columns.
        score: Final composite score.
        indicator_results: List of indicator result dicts from calculate_stock_score.
        market_modifier: Integer market modifier value.
        market_results: List of market index result dicts.
        key_levels: Dict with support and resistance levels.
        risk_alerts: List of risk alert strings.
        position_advice: Position advice string from calculate_position_advice.
        is_intraday: Whether this is an intraday analysis.
        realtime_data: Optional dict with real-time quote data.
        session_label: Label describing current market session.

    Returns:
        Formatted multi-line report string.
    """
    separator = "=" * 50
    lines = []

    # --- Header ---
    latest = df.iloc[-1]
    if is_intraday and realtime_data is not None:
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        raw_date = latest.get("date", "")
        if isinstance(raw_date, pd.Timestamp):
            date_str = raw_date.strftime("%Y-%m-%d")
        else:
            date_str = str(raw_date)[:10] if raw_date else ""

    signal = score_to_signal(score)
    prob = int((score + 100) / 2)  # Map [-100,+100] to [0,100]

    lines.append(_CYAN + separator + _RST)
    lines.append(f"  {_BOLD}{stock_name} ({symbol}){_RST} | {date_str}")
    lines.append(_CYAN + separator + _RST)
    lines.append("")

    # --- Price info ---
    if is_intraday and realtime_data is not None:
        close_val = float(realtime_data["close"])
        prev_close_val = float(realtime_data.get("prev_close", 0))
        change_pct = (close_val / prev_close_val - 1) * 100 if prev_close_val != 0 else 0.0
        pct_color = _BULLISH if change_pct >= 0 else _BEARISH
        lines.append(f"当前价格: {_BOLD}{close_val:.2f}{_RST}  今日涨跌: {pct_color}{change_pct:+.2f}%{_RST}")
        label = session_label or "盘中实时分析"
        lines.append(_DIM + f"[交易时段] {label}" + _RST)
    else:
        last_close = _safe(latest.get("close"))
        prev = df.iloc[-2] if len(df) >= 2 else None
        if prev is not None:
            prev_close = _safe(prev.get("close"))
            if all(v is not None for v in (last_close, prev_close)) and prev_close != 0:
                change_pct = (last_close - prev_close) / prev_close * 100
                pct_color = _BULLISH if change_pct >= 0 else _BEARISH
                lines.append(f"当前价格: {_BOLD}{last_close:.2f}{_RST}  涨跌幅: {pct_color}{change_pct:+.2f}%{_RST}")
            else:
                lines.append(f"当前价格: {_BOLD}{last_close:.2f}{_RST}  涨跌幅: N/A")
        else:
            lines.append(f"当前价格: {_BOLD}{last_close:.2f}{_RST}")

    lines.append("")

    # --- Signal + Probability ---
    colored_signal = _signal_color(signal)
    colored_prob = _score_color(prob)
    if is_intraday:
        direction = "高收" if prob >= 50 else "低收"
        lines.append(f"[信号评级] {colored_signal}")
        lines.append(f"[今日收盘预测] 倾向于收{direction} ({colored_prob}%)")
    else:
        lines.append(f"[信号评级] {colored_signal}")
        lines.append(f"[次日上涨概率] {colored_prob}%")
    lines.append("")

    # --- Intraday Real-Time Status ---
    if is_intraday and realtime_data is not None:
        rt_close = float(realtime_data["close"])
        rt_open = float(realtime_data["open"])
        rt_high = float(realtime_data["high"])
        rt_low = float(realtime_data["low"])
        lines.append(_BOLD + "--- 实时行情 ---" + _RST)
        lines.append(f"现价: {_BOLD}{rt_close:.2f}{_RST} | 开盘: {rt_open:.2f} | 最高: {_BULLISH}{rt_high:.2f}{_RST} | 最低: {_BEARISH}{rt_low:.2f}{_RST}")
        daily_range = rt_high - rt_low
        if daily_range > 0:
            pct_in_range = (rt_close - rt_low) / daily_range
            pct_int = int(pct_in_range * 100)
            if pct_in_range < 0.33:
                region = "低位"
            elif pct_in_range < 0.66:
                region = "中位"
            else:
                region = "高位"
            lines.append(f"盘中位置: {region} (日内振幅第{pct_int}百分位)")
        else:
            lines.append("盘中位置: 平盘 (无振幅)")
        lines.append("")

    # --- Broad Market Environment ---
    if market_results:
        lines.append(_BOLD + "--- 大盘环境 ---" + _RST)
        for r in market_results:
            icon = _trend_icon(r["trend"])
            trend_color = _BULLISH if r["trend"] == "看涨" else (_BEARISH if r["trend"] == "看跌" else _WHITE)
            lines.append(f"  [{icon}] {r['name']}: {trend_color}{r['trend']}{_RST}")
        bullish_count = sum(1 for r in market_results if r["trend"] == "看涨")
        bearish_count = sum(1 for r in market_results if r["trend"] == "看跌")
        modifier_sign = "+" if market_modifier >= 0 else ""
        lines.append(f"大盘修正: {modifier_sign}{market_modifier} ({bullish_count}看涨, {bearish_count}看跌)")
        lines.append("")

    # --- Technical Indicators ---
    lines.append(_BOLD + "--- 技术指标 ---" + _RST)
    for ind in indicator_results:
        icon = _score_icon(ind["score"])
        display_score = round(ind["score"])
        score_sign = "+" if display_score >= 0 else ""
        sc = _BULLISH if display_score > 0 else (_BEARISH if display_score < 0 else _DIM)
        lines.append(f"  [{icon}] {ind['name'].upper()}: {ind['reason']} ({score_sign}{sc}{display_score}{_RST})")
    if is_intraday:
        lines.append(_DIM + "  [!] 注意: 最新K线为盘中数据(未完成)，指标值为近似值" + _RST)
    lines.append("")

    # --- Risk Alerts ---
    if risk_alerts:
        lines.append(_BOLD + "--- 风险警示 ---" + _RST)
        for alert in risk_alerts:
            lines.append(f"  {_YELLOW}[!]{_RST} {alert}")
        lines.append("")

    # --- Key Levels ---
    lines.append(_BOLD + "--- 关键价位 ---" + _RST)
    support_parts = [f"{val:.2f} ({name})" for name, val in key_levels.get("support", [])]
    resistance_parts = [f"{val:.2f} ({name})" for name, val in key_levels.get("resistance", [])]
    if support_parts:
        lines.append(f"支撑位: {' / '.join(support_parts)}")
    if resistance_parts:
        lines.append(f"压力位: {' / '.join(resistance_parts)}")
    lines.append("")

    # --- Position Advice ---
    lines.append(_BOLD + "--- 仓位建议 ---" + _RST)
    lines.append(position_advice)
    lines.append(_CYAN + separator + _RST)

    return "\n".join(lines)
