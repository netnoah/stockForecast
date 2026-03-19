"""
Scoring engine for A-Share stock quantitative analyzer.

Provides weighted indicator evaluation that produces a composite score
and signal rating. Each indicator is scored independently in [-20, +20],
then combined using configurable weights from config.json.
"""

import json
import math
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_RATINGS = [
    (75, 90, "Strong Buy"),
    (60, 74, "Buy"),
    (45, 59, "Hold"),
    (30, 44, "Sell"),
    (10, 29, "Strong Sell"),
]

_SCORE_MIN = -20
_SCORE_MAX = 20


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
        score: Integer score in the range [10, 90].

    Returns:
        Signal label: "Strong Buy", "Buy", "Hold", "Sell", or "Strong Sell".
        Falls back to "Hold" if no range matches.
    """
    for low, high, label in SIGNAL_RATINGS:
        if low <= score <= high:
            return label
    return "Hold"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _clamp(value: float) -> int:
    """Clamp a numeric value to the valid score range [-20, +20]."""
    return int(max(_SCORE_MIN, min(_SCORE_MAX, value)))


def _safe(val, default=None):
    """Return the numeric value or a default if it is NaN/None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------


def score_ma(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on moving average alignment and price position.

    Checks:
    - MA alignment: MA5 > MA10 > MA20 => +10, MA5 < MA10 < MA20 => -10
    - Price vs MA20: close > MA20 => +5, else => -5

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "insufficient data")

    latest = df.iloc[-1]
    ma5 = _safe(latest.get("ma5"))
    ma10 = _safe(latest.get("ma10"))
    ma20 = _safe(latest.get("ma20"))
    close = _safe(latest.get("close"))

    if any(v is None for v in (ma5, ma10, ma20, close)):
        return (0, "MA data incomplete (NaN)")

    parts = []
    total = 0

    # MA alignment
    if ma5 > ma10 > ma20:
        total += 10
        parts.append("bullish alignment (MA5>MA10>MA20) +10")
    elif ma5 < ma10 < ma20:
        total -= 10
        parts.append("bearish alignment (MA5<MA10<MA20) -10")
    else:
        parts.append("mixed MA alignment 0")

    # Price vs MA20
    if close > ma20:
        total += 5
        parts.append("close above MA20 +5")
    else:
        total -= 5
        parts.append("close below MA20 -5")

    return (_clamp(total), "; ".join(parts))


def score_macd(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on MACD crossover and histogram trends.

    Checks:
    - Golden cross (DIF crosses above DEA): +15
    - Death cross (DIF crosses below DEA): -15
    - Histogram turns positive: +8
    - Histogram turns negative: -8
    - Histogram increasing while positive: +5
    - Histogram decreasing while negative: -5

    Note: The MACD histogram column in indicators.py is named "macd".

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "insufficient data")

    prev = df.iloc[-2]
    latest = df.iloc[-1]

    prev_dif = _safe(prev.get("dif"))
    prev_dea = _safe(prev.get("dea"))
    prev_hist = _safe(prev.get("macd"))
    last_dif = _safe(latest.get("dif"))
    last_dea = _safe(latest.get("dea"))
    last_hist = _safe(latest.get("macd"))

    if any(v is None for v in (prev_dif, prev_dea, prev_hist, last_dif, last_dea, last_hist)):
        return (0, "MACD data incomplete (NaN)")

    parts = []
    total = 0

    # Golden cross
    if prev_dif <= prev_dea and last_dif > last_dea:
        total += 15
        parts.append("golden cross (DIF crosses above DEA) +15")

    # Death cross
    if prev_dif >= prev_dea and last_dif < last_dea:
        total -= 15
        parts.append("death cross (DIF crosses below DEA) -15")

    # Histogram turns positive
    if prev_hist <= 0 and last_hist > 0:
        total += 8
        parts.append("histogram turns positive +8")

    # Histogram turns negative
    if prev_hist >= 0 and last_hist < 0:
        total -= 8
        parts.append("histogram turns negative -8")

    # Histogram increasing while positive
    if last_hist > 0 and last_hist > prev_hist:
        total += 5
        parts.append("histogram increasing while positive +5")

    # Histogram decreasing while negative
    if last_hist < 0 and last_hist < prev_hist:
        total -= 5
        parts.append("histogram decreasing while negative -5")

    if not parts:
        parts.append("no MACD signal 0")

    return (_clamp(total), "; ".join(parts))


def score_rsi(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on RSI levels.

    Checks:
    - RSI < 30: +12 (oversold)
    - RSI 30-40: +5 (approaching oversold)
    - RSI > 70: -12 (overbought)
    - RSI 60-70: -3 (approaching overbought)
    - Otherwise: 0 (neutral)

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 1:
        return (0, "insufficient data")

    latest = df.iloc[-1]
    rsi = _safe(latest.get("rsi"))

    if rsi is None:
        return (0, "RSI data incomplete (NaN)")

    if rsi < 30:
        return (12, f"RSI={rsi:.1f} oversold, rebound expected +12")
    if rsi < 40:
        return (5, f"RSI={rsi:.1f} approaching oversold +5")
    if rsi > 70:
        return (-12, f"RSI={rsi:.1f} overbought, pullback risk -12")
    if rsi > 60:
        return (-3, f"RSI={rsi:.1f} approaching overbought -3")
    return (0, f"RSI={rsi:.1f} neutral")


def score_bollinger(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on Bollinger Band position.

    Calculates position within bands: (close - lower) / (upper - lower).
    - Position < 0.1: +12 (near lower band, support)
    - Position 0.1-0.3: +5
    - Position > 0.9: -10 (near upper band, resistance)
    - Position 0.7-0.9: -3
    - Otherwise: 0

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 1:
        return (0, "insufficient data")

    latest = df.iloc[-1]
    close = _safe(latest.get("close"))
    boll_upper = _safe(latest.get("boll_upper"))
    boll_lower = _safe(latest.get("boll_lower"))

    if any(v is None for v in (close, boll_upper, boll_lower)):
        return (0, "Bollinger data incomplete (NaN)")

    band_range = boll_upper - boll_lower
    if band_range == 0:
        return (0, "Bollinger band width is zero, cannot calculate position")

    position = (close - boll_lower) / band_range

    if position < 0.1:
        return (12, f"position={position:.2f} near lower band (support) +12")
    if position < 0.3:
        return (5, f"position={position:.2f} in lower region +5")
    if position > 0.9:
        return (-10, f"position={position:.2f} near upper band (resistance) -10")
    if position > 0.7:
        return (-3, f"position={position:.2f} in upper region -3")
    return (0, f"position={position:.2f} neutral")


def score_kdj(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on KDJ crossover and extreme J values.

    Checks:
    - KDJ golden cross (K crosses above D): +12
    - KDJ death cross (K crosses below D): -12
    - J < 20 for two consecutive days: +10 (oversold)
    - J > 80 for two consecutive days: -10 (overbought)
    - Otherwise: 0

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "insufficient data")

    prev = df.iloc[-2]
    latest = df.iloc[-1]

    prev_k = _safe(prev.get("k"))
    prev_d = _safe(prev.get("d"))
    prev_j = _safe(prev.get("j"))
    last_k = _safe(latest.get("k"))
    last_d = _safe(latest.get("d"))
    last_j = _safe(latest.get("j"))

    if any(v is None for v in (prev_k, prev_d, prev_j, last_k, last_d, last_j)):
        return (0, "KDJ data incomplete (NaN)")

    parts = []
    total = 0

    # KDJ golden cross
    if prev_k < prev_d and last_k > last_d:
        total += 12
        parts.append("KDJ golden cross (K crosses above D) +12")

    # KDJ death cross
    if prev_k > prev_d and last_k < last_d:
        total -= 12
        parts.append("KDJ death cross (K crosses below D) -12")

    # J oversold
    if last_j < 20 and prev_j < 20:
        total += 10
        parts.append(f"J={last_j:.1f} oversold for 2 days +10")

    # J overbought
    if last_j > 80 and prev_j > 80:
        total -= 10
        parts.append(f"J={last_j:.1f} overbought for 2 days -10")

    if not parts:
        parts.append("no KDJ signal 0")

    return (_clamp(total), "; ".join(parts))


def score_volume(df: pd.DataFrame) -> tuple[int, str]:
    """Score based on volume-price relationship.

    Checks:
    - vol_ratio > 1.5 AND price up: +12 (strong buying)
    - vol_ratio > 1.5 AND price down: -10 (distribution)
    - vol_ratio < 0.5 AND price up: -5 (volume-price divergence)
    - vol_ratio < 0.5 AND price down: +5 (selling exhaustion)
    - vol_ratio > 1.0 AND price up: +5
    - Otherwise: 0

    Returns:
        (score, reason) tuple.
    """
    if len(df) < 2:
        return (0, "insufficient data")

    prev = df.iloc[-2]
    latest = df.iloc[-1]

    vol_ratio = _safe(latest.get("vol_ratio"))
    prev_close = _safe(prev.get("close"))
    last_close = _safe(latest.get("close"))

    if any(v is None for v in (vol_ratio, prev_close, last_close)):
        return (0, "Volume data incomplete (NaN)")

    price_up = last_close > prev_close

    if vol_ratio > 1.5 and price_up:
        return (12, f"vol_ratio={vol_ratio:.2f} volume spike + price up = strong buying +12")
    if vol_ratio > 1.5 and not price_up:
        return (-10, f"vol_ratio={vol_ratio:.2f} volume spike + price down = distribution -10")
    if vol_ratio < 0.5 and price_up:
        return (-5, f"vol_ratio={vol_ratio:.2f} low volume + price up = divergence -5")
    if vol_ratio < 0.5 and not price_up:
        return (5, f"vol_ratio={vol_ratio:.2f} low volume + price down = selling exhaustion +5")
    if vol_ratio > 1.0 and price_up:
        return (5, f"vol_ratio={vol_ratio:.2f} above-average volume + price up +5")
    return (0, f"vol_ratio={vol_ratio:.2f} no volume signal")


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
            "score": score,
            "reason": reason,
        })

    # Calculate raw score centered at 50
    if total_weight > 0:
        raw_score = 50 + (weighted_sum / total_weight)
    else:
        raw_score = 50.0

    # Clamp to valid signal range
    raw_score = max(10, min(90, raw_score))

    return (int(raw_score), results)
