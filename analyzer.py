from __future__ import annotations

"""
Scoring engine for A-Share stock quantitative analyzer.

Provides weighted indicator evaluation that produces a composite score
and signal rating. Each indicator is scored independently in [-20, +20],
then combined using configurable weights from config.json.
"""

import json
import logging
import math
import os
from datetime import datetime

import pandas as pd

from data_source import get_index_history, get_index_realtime_quote
from indicators import calc_ma, calc_macd

logger = logging.getLogger(__name__)


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

# Trend strength levels for market modifier
_TRENT_BULLISH = "bullish"
_TRENT_BEARISH = "bearish"
_TRENT_NEUTRAL = "neutral"

_TRENT_LABELS = {
    _TRENT_BULLISH: "看涨",
    _TRENT_BEARISH: "看跌",
    _TRENT_NEUTRAL: "中性",
}

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


def calculate_signal(score: int, trend_status: str) -> str:
    """Generate signal with trend filtering to reduce false signals.

    Rules:
    - Strong buy: score >= 50 AND trend in (strong_bull, bull)
    - Buy: score >= 15 AND trend NOT in (bear, strong_bear)
    - Sell: score <= -15 AND trend in (bear, strong_bear)
    - Strong sell: score <= -50 AND trend in (bear, strong_bear)
    - Otherwise: Hold (观望)

    Args:
        score: Integer score in [-100, 100].
        trend_status: One of the _TREND_* constants.

    Returns:
        Signal label string.
    """
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
    """7-level trend classification based on MA alignment and spacing.

    Checks MA5 > MA10 > MA20 (bullish) or MA5 < MA10 < MA20 (bearish),
    then examines whether the spacing has widened over the past 5 bars
    to distinguish strong vs regular trends.

    Returns:
        (trend_status, strength) where:
        - trend_status: one of _TREND_STRONG_BULL, _TREND_BULL, etc.
        - strength: integer 0-100 (100 = strongest bull, 0 = strongest bear).
    """
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
        """Check if MA5-MA10 spacing has widened by >5% over 5 bars."""
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
    """Score based on MA alignment, trend classification, and price deviation.

    Continuous scoring:
    - Trend strength: (MA5 - MA20) / MA20, scaled to [-6, +6]
    - MA alignment: bonus for bullish/bearish alignment, clamped [-6, +6]
    - Price position vs MA20: (close - MA20) / MA20, scaled to [-8, +8]
    - Trend bonus: ±3 for strong_bull/strong_bear (trend acceleration)

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

    trend_status, trend_strength = _classify_trend(df)

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

    # 4) Strong trend bonus (trend acceleration signal)
    trend_bonus = 0.0
    if trend_status == _TREND_STRONG_BULL:
        trend_bonus = 3.0
    elif trend_status == _TREND_STRONG_BEAR:
        trend_bonus = -3.0

    total = trend + alignment + position + trend_bonus

    # Build reason text
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
    """Score based on MACD momentum (DIF trend) and histogram position.

    Continuous scoring:
    - Momentum: DIF rate of change (DIF today vs yesterday), scaled to [-15, +15].
      This captures whether DIF is accelerating or decelerating, correctly
      detecting trend improvement even when DIF/DEA are both negative (underwater).
    - Histogram position: (DIF - DEA) normalized by close, scaled to [-5, +5].
      Positive when DIF is above DEA (bullish), negative when below (bearish).

    Note: The MACD histogram column in indicators.py is named "macd".

    Returns:
        (score, reason) tuple.
    """
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

    # Momentum: DIF's rate of change — true directional momentum
    dif_momentum = (last_dif - prev_dif) / close * 2000
    dif_momentum = max(-15, min(15, dif_momentum))

    # Histogram position: DIF vs DEA distance — current bullish/bearish state
    hist_value = (last_dif - last_dea) / close * 2000
    hist_position = max(-5, min(5, hist_value))

    total = dif_momentum + hist_position
    reason = f"DIF动量={dif_momentum:+.1f}; 柱状图位置={hist_position:+.1f}"
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


def score_bias(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on price deviation from MA5 (bias).

    BIAS = (close - MA5) / MA5 * 100%. Penalizes chasing highs and rewards
    pullbacks near MA5. Strong bull trend (strength >= 70) widens thresholds
    by 1.5x to avoid missing super-trend moves.

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "数据不足")

    latest = df.iloc[-1]
    close = _safe(latest.get("close"))
    ma5 = _safe(latest.get("ma5"))

    if any(v is None for v in (close, ma5)) or ma5 == 0:
        return (0, "BIAS数据不完整 (NaN)")

    bias_pct = (close - ma5) / ma5 * 100

    # Strong bull compensation: widen thresholds by 1.5x
    trend_status, trend_strength = _classify_trend(df)
    if trend_status == _TREND_STRONG_BULL and trend_strength >= 70:
        bias_pct_adjusted = bias_pct / 1.5
    else:
        bias_pct_adjusted = bias_pct

    # Scoring table (applied to adjusted bias)
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
    """Calculate a composite stock score based on weighted indicator evaluation.

    Reads indicator configuration from config.json to determine which
    indicators are enabled and their respective weights.

    Raw score = (weighted_sum / total_weight) * 5, producing a value
    in the approximate range [-100, +100].

    Args:
        df: DataFrame with all indicator columns already calculated.
        config: Parsed config.json dictionary.

    Returns:
        (raw_score, results, trend_status) where:
        - raw_score: Integer score in [-100, +100] range.
        - results: List of dicts with keys: name, score, reason.
        - trend_status: One of the _TREND_* constants.
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

    trend_status, _ = _classify_trend(df)

    logger.info("Composite score: raw=%d trend=%s indicators=%s",
                int(raw_score), trend_status,
                ", ".join(f"{r['name']}={r['score']}" for r in results))

    return (int(raw_score), results, trend_status)


# ---------------------------------------------------------------------------
# Market modifier
# ---------------------------------------------------------------------------


def classify_index_trend(df: pd.DataFrame) -> tuple[str, float]:
    """Classify the trend of a market index with strength score.

    Combines three signals:
    - Price vs MA20 position (bias %)
    - MACD direction (DIF vs DEA)
    - MA20 slope (5-day change rate)

    Returns:
        (trend, strength) where trend is one of _TRENT_* constants
        and strength is a float in [-1, +1].
    """
    if len(df) < 6:
        return (_TRENT_NEUTRAL, 0.0)

    latest = df.iloc[-1]
    close = _safe(latest.get("close"))
    ma20 = _safe(latest.get("ma20"))
    dif = _safe(latest.get("dif"))
    dea = _safe(latest.get("dea"))

    if any(v is None for v in (close, ma20, dif, dea)):
        return (_TRENT_NEUTRAL, 0.0)

    # 1) Price bias from MA20
    bias = (close - ma20) / ma20
    bias_score = max(-1.0, min(1.0, bias * 20))  # ±5% bias → ±1

    # 2) MACD direction
    macd_score = 0.0
    macd_gap = (dif - dea) / close * 1000  # normalized
    macd_score = max(-1.0, min(1.0, macd_gap * 2))

    # 3) MA20 slope (5-day change rate)
    slope_score = 0.0
    if len(df) >= 25:
        ma20_5ago = _safe(df.iloc[-6].get("ma20"))
        if ma20_5ago is not None and ma20_5ago != 0:
            slope = (ma20 - ma20_5ago) / ma20_5ago
            slope_score = max(-1.0, min(1.0, slope * 50))

    # Composite strength
    strength = (bias_score * 0.4 + macd_score * 0.35 + slope_score * 0.25)
    strength = max(-1.0, min(1.0, strength))

    if strength > 0.15:
        return (_TRENT_BULLISH, strength)
    elif strength < -0.15:
        return (_TRENT_BEARISH, strength)
    return (_TRENT_NEUTRAL, strength)


def calculate_market_modifier(config: dict, intraday: bool = False) -> tuple[int, list[dict]]:
    """Calculate a market-wide modifier based on broad index trends.

    Each index is classified with a strength score in [-1, +1].
    The modifier is the weighted average of all index strengths, scaled
    to [-max_impact, +max_impact].

    When *intraday* is True, real-time index quotes are fetched and
    appended so that the modifier reflects the current session rather
    than the previous close.

    Args:
        config: Parsed config.json dictionary.
        intraday: Whether the analysis is running during trading hours.

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

            # Intraday momentum override — MACD is a lagging indicator and
            # can remain positive for several bars after a sudden drop.
            # When the intraday change exceeds 1 %, blend in a direct
            # momentum signal so the modifier reacts in real time.
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
                            trend = _TRENT_BULLISH
                        elif strength < -0.15:
                            trend = _TRENT_BEARISH
                        else:
                            trend = _TRENT_NEUTRAL

            name = INDEX_NAMES.get(code, code)
            results.append({"code": code, "name": name, "trend": trend, "strength": strength})
        except Exception:
            logger.warning("Failed to fetch/calculate index trend for %s, defaulting to neutral", code)
            results.append({"code": code, "name": INDEX_NAMES.get(code, code), "trend": _TRENT_NEUTRAL, "strength": 0.0})

    # Weighted average of strengths
    total_strength = sum(r["strength"] for r in results)
    avg_strength = total_strength / len(results) if results else 0.0

    modifier = int(avg_strength * max_impact)
    modifier = max(-max_impact, min(max_impact, modifier))

    logger.info("Market modifier: %d (indices: %s)", modifier,
                ", ".join(f"{r['name']}={r['trend']}({r['strength']:+.2f})" for r in results))

    return (modifier, results)


# ---------------------------------------------------------------------------
# Key levels
# ---------------------------------------------------------------------------


def calculate_key_levels(df: pd.DataFrame) -> dict:
    """Calculate support and resistance levels with validity markers.

    Support validity (✓ marker):
    - MA5: price within 2% above MA5
    - MA10: price within 2% above MA10
    - MA20: price >= MA20

    Args:
        df: DataFrame with indicator columns (ma5, ma10, ma20, boll_lower, boll_upper, high).

    Returns:
        Dict with "support" and "resistance" keys, each containing a list
        of (name, value) tuples. Valid supports include ✓ in the name.
    """
    last = df.iloc[-1]
    close = _safe(last.get("close"))
    support = []
    resistance = []

    # MA5 support: price within 2% above MA5
    ma5 = _safe(last.get("ma5"))
    if ma5 is not None and close is not None:
        if close >= ma5 and (close - ma5) / ma5 <= 0.02:
            support.append(("MA5 ✓", ma5))
        else:
            support.append(("MA5", ma5))

    # MA10 support: price within 2% above MA10
    ma10 = _safe(last.get("ma10"))
    if ma10 is not None and close is not None:
        if close >= ma10 and (close - ma10) / ma10 <= 0.02:
            support.append(("MA10 ✓", ma10))
        else:
            support.append(("MA10", ma10))

    # MA20 support: price >= MA20
    ma20 = _safe(last.get("ma20"))
    if ma20 is not None and close is not None:
        if close >= ma20:
            support.append(("MA20 ✓", ma20))
        else:
            support.append(("MA20", ma20))

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
    if trend == _TRENT_BULLISH:
        return "+"
    if trend == _TRENT_BEARISH:
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
    trend_status: str = "",
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
        trend_status: Stock trend status from _classify_trend().

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

    if trend_status:
        signal = calculate_signal(score, trend_status)
    else:
        signal = score_to_signal(score)

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

    # --- Signal + Score ---
    colored_signal = _signal_color(signal)
    colored_score = _score_color(score)
    lines.append(f"[信号评级] {colored_signal}")
    lines.append(f"[综合评分] {colored_score}分")
    if trend_status:
        trend_label = _TREND_STOCK_LABELS.get(trend_status, trend_status)
        trend_color = _BULLISH if trend_status in (_TREND_STRONG_BULL, _TREND_BULL, _TREND_WEAK_BULL) else (_BEARISH if trend_status in (_TREND_STRONG_BEAR, _TREND_BEAR, _TREND_WEAK_BEAR) else _WHITE)
        lines.append(f"[趋势状态] {trend_color}{trend_label}{_RST}")
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
            trend_label = _TRENT_LABELS.get(r["trend"], r["trend"])
            trend_color = _BULLISH if r["trend"] == _TRENT_BULLISH else (_BEARISH if r["trend"] == _TRENT_BEARISH else _WHITE)
            strength_str = f"({r['strength']:+.2f})" if "strength" in r else ""
            lines.append(f"  [{icon}] {r['name']}: {trend_color}{trend_label}{strength_str}{_RST}")
        bullish_count = sum(1 for r in market_results if r["trend"] == _TRENT_BULLISH)
        bearish_count = sum(1 for r in market_results if r["trend"] == _TRENT_BEARISH)
        modifier_sign = "+" if market_modifier >= 0 else ""
        lines.append(f"大盘修正: {modifier_sign}{market_modifier} ({bullish_count}看涨, {bearish_count}看跌)")
        lines.append(_DIM + "  (括号内为趋势强度 [-1,+1], 由价格偏离MA20/MACD方向/MA20斜率综合计算)" + _RST)
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

    # --- Score Reference ---
    lines.append(_DIM + "评分参考: 强烈买入[50,100]+多头趋势 买入[15,49]+非空头趋势 卖出[-49,-15]+空头趋势 强烈卖出[-100,-50]+空头趋势" + _RST)

    return "\n".join(lines)
