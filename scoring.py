from __future__ import annotations

"""Indicator scoring engine and composite score calculation."""

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


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

_SCORE_MIN = -100
_SCORE_MAX = 100

# 7-level stock trend classification (for individual stock scoring)
_TREND_STRONG_BULL = "strong_bull"
_TREND_BULL = "bull"
_TREND_WEAK_BULL = "weak_bull"
_TREND_CONSOLIDATION = "consolidation"
_TREND_WEAK_BEAR = "weak_bear"
_TREND_BEAR = "bear"
_TREND_STRONG_BEAR = "strong_bear"

_TREND_STOCK_LABELS = {
    _TREND_STRONG_BULL: "强势多头",
    _TREND_BULL: "多头",
    _TREND_WEAK_BULL: "弱多头",
    _TREND_CONSOLIDATION: "震荡",
    _TREND_WEAK_BEAR: "弱空头",
    _TREND_BEAR: "空头",
    _TREND_STRONG_BEAR: "强势空头",
}


# ---------------------------------------------------------------------------
# Signal mapping
# ---------------------------------------------------------------------------


def score_to_signal(score: int) -> str:
    """Convert a numeric score to a human-readable signal label."""
    for low, high, label in SIGNAL_RATINGS:
        if low <= score <= high:
            return label
    return "观望"


def calculate_signal(score: int, trend_status: str) -> str:
    """Generate signal with trend filtering to reduce false signals."""
    bullish_trends = {_TREND_STRONG_BULL, _TREND_BULL, _TREND_WEAK_BULL, _TREND_CONSOLIDATION}
    bearish_trends = {_TREND_STRONG_BEAR, _TREND_BEAR}
    strong_bearish_trends = {_TREND_STRONG_BEAR, _TREND_BEAR, _TREND_WEAK_BEAR}

    if score >= 50 and trend_status in {_TREND_STRONG_BULL, _TREND_BULL}:
        return "强烈买入"
    if score >= 15 and trend_status not in bearish_trends:
        return "买入"
    if score <= -50 and trend_status in bearish_trends:
        return "强烈卖出"
    if score <= -15 and trend_status in strong_bearish_trends:
        return "卖出"
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


def _classify_trend(df: pd.DataFrame) -> tuple[str, int]:
    """7-level trend classification based on MA alignment and spacing."""
    if len(df) < 6:
        return (_TREND_CONSOLIDATION, 50)

    latest = df.iloc[-1]
    ma5 = _safe(latest.get("ma5"))
    ma10 = _safe(latest.get("ma10"))
    ma20 = _safe(latest.get("ma20"))

    if any(v is None for v in (ma5, ma10, ma20)):
        return (_TREND_CONSOLIDATION, 50)

    prev = df.iloc[-6]
    p_ma5 = _safe(prev.get("ma5"))
    p_ma10 = _safe(prev.get("ma10"))

    def _spacing_widening(p5: float, p10: float) -> bool:
        if any(v is None for v in (p5, p10)) or p10 == 0:
            return False
        prev_gap = (p5 - p10) / abs(p10)
        curr_gap = (ma5 - ma10) / abs(ma10)
        if abs(prev_gap) < 1e-6:
            return abs(curr_gap) > 0.005
        return (curr_gap - prev_gap) / abs(prev_gap) > 0.05

    if ma5 > ma10 > ma20:
        if _spacing_widening(p_ma5, p_ma10):
            return (_TREND_STRONG_BULL, 90)
        return (_TREND_BULL, 75)
    if ma5 < ma10 < ma20:
        if _spacing_widening(p_ma5, p_ma10):
            return (_TREND_STRONG_BEAR, 10)
        return (_TREND_BEAR, 25)
    if ma5 > ma10 and ma10 <= ma20:
        return (_TREND_WEAK_BULL, 55)
    if ma5 < ma10 and ma10 >= ma20:
        return (_TREND_WEAK_BEAR, 40)
    return (_TREND_CONSOLIDATION, 50)


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------


def score_ma(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on MA alignment, trend classification, and price deviation."""
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

    trend_status, trend_strength = _classify_trend(df)

    trend = (ma5 - ma20) / ma20 * 200
    trend = max(-6, min(6, trend))

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

    position = (close - ma20) / ma20 * 200
    position = max(-8, min(8, position))

    trend_bonus = 0.0
    if trend_status == _TREND_STRONG_BULL:
        trend_bonus = 3.0
    elif trend_status == _TREND_STRONG_BEAR:
        trend_bonus = -3.0

    total = trend + alignment + position + trend_bonus

    trend_label = _TREND_STOCK_LABELS.get(trend_status, "未知")
    alignment_labels = {1: "多头", -1: "空头", 0: "混乱"}
    short_label = alignment_labels.get(parts[0], "混乱")
    long_label = alignment_labels.get(parts[1], "无MA60") if has_ma60 else "无MA60"
    reason = (
        f"趋势={trend_label}({trend_strength}); "
        f"排列={short_label}/{long_label}={alignment:+.1f}; "
        f"偏离={position:+.1f}"
    )
    return (_clamp(total), reason)


def score_macd(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on MACD momentum (DIF trend) and histogram position."""
    if len(df) < 2:
        return (0, "数据不足")

    prev = df.iloc[-2]
    latest = df.iloc[-1]

    prev_dif = _safe(prev.get("dif"))
    last_dif = _safe(latest.get("dif"))
    last_dea = _safe(latest.get("dea"))
    close = _safe(latest.get("close"))

    if any(v is None for v in (prev_dif, last_dif, last_dea, close)):
        return (0, "MACD数据不完整 (NaN)")

    dif_momentum = (last_dif - prev_dif) / close * 2000
    dif_momentum = max(-15, min(15, dif_momentum))

    hist_value = (last_dif - last_dea) / close * 2000
    hist_position = max(-5, min(5, hist_value))

    total = dif_momentum + hist_position
    reason = f"DIF动量={dif_momentum:+.1f}; 柱状图位置={hist_position:+.1f}"
    return (_clamp(total), reason)


def _rsi_segment_map(rsi: float) -> float:
    """Map RSI to score using a segmented function."""
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
    """Score based on RSI using a segmented mapping."""
    if len(df) < 1:
        return (0, "数据不足")

    latest = df.iloc[-1]
    rsi = _safe(latest.get("rsi"))

    if rsi is None:
        return (0, "RSI数据不完整 (NaN)")

    score = _rsi_segment_map(rsi)
    return (_clamp(score), f"RSI={rsi:.1f} → {score:+.1f}")


def score_bollinger(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on Bollinger Band position and squeeze detection."""
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

    position = (close - boll_lower) / band_range
    position_score = (0.5 - position) * 28

    if boll_width is not None and boll_width > 0:
        squeeze_factor = min(boll_width / 0.04, 1.0)
        position_score *= squeeze_factor
        squeeze_label = "收窄" if squeeze_factor < 0.8 else "正常"
    else:
        squeeze_factor = 1.0
        squeeze_label = "未知"

    position_score = max(-14, min(14, position_score))

    squeeze_bonus = 0.0
    widths = df["boll_width"].tail(21).dropna()
    if len(widths) >= 10 and boll_width is not None:
        avg_width = widths.iloc[:-1].mean()
        if avg_width > 0:
            width_ratio = boll_width / avg_width
            if width_ratio < 0.7:
                prev_close = _safe(df.iloc[-2].get("close"))
                if prev_close is not None and prev_close != 0:
                    direction = 1 if close > prev_close else -1
                    intensity = (0.7 - width_ratio) / 0.7
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
    """Score based on KDJ values as a continuous function."""
    if len(df) < 2:
        return (0, "数据不足")

    latest = df.iloc[-1]

    last_k = _safe(latest.get("k"))
    last_d = _safe(latest.get("d"))
    last_j = _safe(latest.get("j"))

    if any(v is None for v in (last_k, last_d, last_j)):
        return (0, "KDJ数据不完整 (NaN)")

    kd_momentum = (last_k - last_d) / 10
    kd_momentum = max(-10, min(10, kd_momentum))

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
    """Score based on volume-price relationship as a continuous function."""
    if len(df) < 2:
        return (0, "数据不足")

    prev = df.iloc[-2]
    latest = df.iloc[-1]

    vol_ratio = _safe(latest.get("vol_ratio"))
    prev_close = _safe(prev.get("close"))
    last_close = _safe(latest.get("close"))

    if any(v is None for v in (vol_ratio, prev_close, last_close)):
        return (0, "成交量数据不完整 (NaN)")

    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close != 0 else 0.0
    if abs(change_pct) < 0.5:
        return (0, f"量比={vol_ratio:.2f} 平盘(涨跌{change_pct:+.2f}%) → 无信号")

    direction_weight = min(abs(change_pct) / 3.0, 1.0)
    price_direction = 1 if change_pct > 0 else -1
    vol_deviation = vol_ratio - 1
    score = vol_deviation * price_direction * 20 * direction_weight

    direction_text = f"{'上涨' if price_direction > 0 else '下跌'}{abs(change_pct):.1f}%"

    return (_clamp(score), f"量比={vol_ratio:.2f} {direction_text} → {score:+.1f}")


def score_bias(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on price deviation from MA5 (bias)."""
    if len(df) < 2:
        return (0, "数据不足")

    latest = df.iloc[-1]
    close = _safe(latest.get("close"))
    ma5 = _safe(latest.get("ma5"))

    if any(v is None for v in (close, ma5)) or ma5 == 0:
        return (0, "BIAS数据不完整 (NaN)")

    bias_pct = (close - ma5) / ma5 * 100

    trend_status, trend_strength = _classify_trend(df)
    if trend_status == _TREND_STRONG_BULL and trend_strength >= 70:
        bias_pct_adjusted = bias_pct / 1.5
    else:
        bias_pct_adjusted = bias_pct

    if bias_pct_adjusted < -5:
        score = 8
        label = "乖离过大"
    elif bias_pct_adjusted < -3:
        score = 16
        label = "回踩MA5"
    elif bias_pct_adjusted < 0:
        score = 20
        label = "略低于MA5"
    elif bias_pct_adjusted < 2:
        score = 18
        label = "贴近MA5"
    elif bias_pct_adjusted < 5:
        score = 8
        label = "略高于MA5"
    else:
        score = -20
        label = "乖离过高(追高风险)"

    reason = f"BIAS={bias_pct:+.2f}%; {label} → {score:+.1f}"
    return (_clamp(score), reason)


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
    "bias": score_bias,
}


def calculate_stock_score(df: pd.DataFrame, config: dict) -> tuple[int, list[dict], str]:
    """Calculate a composite stock score based on weighted indicator evaluation."""
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

    if total_weight > 0:
        raw_score = (weighted_sum / total_weight) * 5
    else:
        raw_score = 0.0

    raw_score = max(-100, min(100, raw_score))

    trend_status, _ = _classify_trend(df)

    logger.info("Composite score: raw=%d trend=%s indicators=%s",
                int(raw_score), trend_status,
                ", ".join(f"{r['name']}={r['score']}" for r in results))

    return (int(raw_score), results, trend_status)
