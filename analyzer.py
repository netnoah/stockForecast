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

INDEX_NAMES = {
    "sh000001": "Shanghai Composite",
    "sz399001": "Shenzhen Component",
    "sz399006": "ChiNext",
    "sh000905": "CSI 500",
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
        return "bullish"
    if close < ma20 and dif < dea:
        return "bearish"
    return "neutral"


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
            results.append({"code": code, "name": INDEX_NAMES.get(code, code), "trend": "neutral"})

    bullish_count = sum(1 for r in results if r["trend"] == "bullish")
    bearish_count = sum(1 for r in results if r["trend"] == "bearish")
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
        support.append(("Boll Lower", last["boll_lower"]))

    if not pd.isna(last.get("boll_upper")):
        resistance.append(("Boll Upper", last["boll_upper"]))

    recent_high = df["high"].tail(20).max()
    resistance.append(("20D High", recent_high))

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
    if score >= 75:
        pct = "60%"
        signal = "strong buy"
    elif score >= 60:
        pct = "40%"
        signal = "buy signal"
    elif score >= 45:
        pct = "20%"
        signal = "neutral"
    elif score >= 30:
        pct = "10%"
        signal = "sell signal"
    else:
        pct = "0%"
        signal = "strong sell"

    return f"Suggested: {pct} ({signal})"


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
        alerts.append(f"RSI={rsi:.1f} approaching overbought, watch for pullback")
    elif rsi is not None and rsi < 35:
        alerts.append(f"RSI={rsi:.1f} approaching oversold, may rebound")

    close = _safe(latest.get("close"))
    boll_upper = _safe(latest.get("boll_upper"))
    boll_lower = _safe(latest.get("boll_lower"))
    if all(v is not None for v in (close, boll_upper, boll_lower)):
        band_range = boll_upper - boll_lower
        if band_range > 0:
            boll_position = (close - boll_lower) / band_range
            if boll_position > 0.85:
                alerts.append("Close to Bollinger upper band, limited upside room")
            elif boll_position < 0.15:
                alerts.append("Near Bollinger lower band, watch for support breakdown")

    if prev is not None:
        prev_close = _safe(prev.get("close"))
        if all(v is not None for v in (close, prev_close)) and prev_close != 0:
            daily_change = (close - prev_close) / prev_close * 100
            if abs(daily_change) > 5:
                alerts.append(f"Large daily swing ({daily_change:.1f}%), expect volatility")

    if score >= 70:
        alerts.append("High bullish score - consider taking partial profits on existing positions")
    elif score <= 30:
        alerts.append("Low score - avoid adding positions, wait for reversal signal")

    return alerts


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def _trend_icon(trend: str) -> str:
    """Return a single-character icon for a trend classification."""
    if trend == "bullish":
        return "+"
    if trend == "bearish":
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
    prob = max(10, min(90, score))

    lines.append(separator)
    lines.append(f"  Stock Analysis Report | {symbol} | {date_str}")
    lines.append(separator)
    lines.append("")

    # --- Price info ---
    if is_intraday and realtime_data is not None:
        close_val = float(realtime_data["close"])
        open_val = float(realtime_data["open"])
        change_pct = (close_val / open_val - 1) * 100 if open_val != 0 else 0.0
        lines.append(f"Current Price: {close_val:.2f}  Today's Change: {change_pct:+.2f}%")
        lines.append("[Trading Session] Intraday real-time analysis")
    else:
        last_close = _safe(latest.get("close"))
        prev = df.iloc[-2] if len(df) >= 2 else None
        if prev is not None:
            prev_close = _safe(prev.get("close"))
            if all(v is not None for v in (last_close, prev_close)) and prev_close != 0:
                change_pct = (last_close - prev_close) / prev_close * 100
                lines.append(f"Current Price: {last_close:.2f}  Change: {change_pct:+.2f}%")
            else:
                lines.append(f"Current Price: {last_close:.2f}  Change: N/A")
        else:
            lines.append(f"Current Price: {last_close:.2f}")

    lines.append("")

    # --- Signal + Probability ---
    if is_intraday:
        direction = "higher" if prob >= 50 else "lower"
        lines.append(f"[Signal Rating] {signal}")
        lines.append(f"[Today Close Prediction] Leaning towards close {direction} ({prob}%)")
    else:
        lines.append(f"[Signal Rating] {signal}")
        lines.append(f"[Next-Day Up Probability] {prob}%")
    lines.append("")

    # --- Intraday Real-Time Status ---
    if is_intraday and realtime_data is not None:
        rt_close = float(realtime_data["close"])
        rt_open = float(realtime_data["open"])
        rt_high = float(realtime_data["high"])
        rt_low = float(realtime_data["low"])
        lines.append("--- Real-Time Status ---")
        lines.append(f"Current: {rt_close:.2f} | Open: {rt_open:.2f} | High: {rt_high:.2f} | Low: {rt_low:.2f}")
        daily_range = rt_high - rt_low
        if daily_range > 0:
            pct_in_range = (rt_close - rt_low) / daily_range
            pct_int = int(pct_in_range * 100)
            if pct_in_range < 0.33:
                region = "lower"
            elif pct_in_range < 0.66:
                region = "middle"
            else:
                region = "upper"
            lines.append(f"Intraday position: {region} region ({pct_int}th percentile of daily range)")
        else:
            lines.append("Intraday position: flat (no range)")
        lines.append("")

    # --- Broad Market Environment ---
    if market_results:
        lines.append("--- Broad Market Environment ---")
        for r in market_results:
            icon = _trend_icon(r["trend"])
            lines.append(f"  [{icon}] {r['name']}: {r['trend']}")
        bullish_count = sum(1 for r in market_results if r["trend"] == "bullish")
        bearish_count = sum(1 for r in market_results if r["trend"] == "bearish")
        modifier_sign = "+" if market_modifier >= 0 else ""
        lines.append(f"Market modifier: {modifier_sign}{market_modifier} ({bullish_count} bullish, {bearish_count} bearish)")
        lines.append("")

    # --- Technical Indicators ---
    lines.append("--- Technical Indicators ---")
    for ind in indicator_results:
        icon = _score_icon(ind["score"])
        score_sign = "+" if ind["score"] >= 0 else ""
        lines.append(f"  [{icon}] {ind['name'].upper()}: {ind['reason']} ({score_sign}{ind['score']})")
    if is_intraday:
        lines.append("  [!] Note: Latest candle is intraday (incomplete), indicator values are approximate")
    lines.append("")

    # --- Risk Alerts ---
    if risk_alerts:
        lines.append("--- Risk Alerts ---")
        for alert in risk_alerts:
            lines.append(f"  [!] {alert}")
        lines.append("")

    # --- Key Levels ---
    lines.append("--- Key Levels ---")
    support_parts = [f"{val:.2f} ({name})" for name, val in key_levels.get("support", [])]
    resistance_parts = [f"{val:.2f} ({name})" for name, val in key_levels.get("resistance", [])]
    if support_parts:
        lines.append(f"Support: {' / '.join(support_parts)}")
    if resistance_parts:
        lines.append(f"Resistance: {' / '.join(resistance_parts)}")
    lines.append("")

    # --- Position Advice ---
    lines.append("--- Position Advice ---")
    lines.append(position_advice)
    lines.append(separator)

    return "\n".join(lines)
