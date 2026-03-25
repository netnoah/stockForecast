from __future__ import annotations

"""Report formatting with ANSI color support."""

import logging
from datetime import datetime

import pandas as pd

from .models import AnalysisResult
from .scoring import (
    _safe,
    score_to_signal,
    calculate_signal,
    _TREND_STRONG_BULL,
    _TREND_BULL,
    _TREND_WEAK_BULL,
    _TREND_CONSOLIDATION,
    _TREND_WEAK_BEAR,
    _TREND_BEAR,
    _TREND_STRONG_BEAR,
    _TREND_STOCK_LABELS,
)
from .market import _TREND_IDX_BULLISH, _TREND_IDX_BEARISH, _TREND_IDX_NEUTRAL, _TREND_IDX_LABELS

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
    if signal in ("强烈买入", "买入"):
        return _BULLISH + _BOLD + signal + _RST
    if signal in ("强烈卖出", "卖出"):
        return _BEARISH + signal + _RST
    return _WHITE + signal + _RST


def _score_color(score: int) -> str:
    """Return ANSI color for a numeric score."""
    if score >= 15:
        return _BULLISH + _BOLD + str(score) + _RST
    if score <= -15:
        return _BEARISH + str(score) + _RST
    return _YELLOW + str(score) + _RST


def _trend_icon(trend: str) -> str:
    """Return a single-character icon for a trend classification."""
    if trend == _TREND_IDX_BULLISH:
        return "+"
    if trend == _TREND_IDX_BEARISH:
        return "-"
    return "~"


def _score_icon(score: int) -> str:
    """Return a single-character icon for an indicator score."""
    if score > 0:
        return "+"
    if score < 0:
        return "-"
    return "~"


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_report(
    result: AnalysisResult,
    df: pd.DataFrame,
) -> str:
    """Format all analysis data into a human-readable text report."""
    symbol = result.symbol
    stock_name = result.stock_name
    score = result.score
    indicator_results = result.indicator_results
    market_modifier = result.market_modifier
    market_results = result.market_results
    key_levels = result.key_levels
    risk_alerts = result.risk_alerts
    position_advice = result.position_advice
    is_intraday = result.is_intraday
    realtime_data = result.realtime_data
    session_label = result.session_label
    trend_status = result.trend_status

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
            if pct_int < 0.33:
                region = "低位"
            elif pct_int < 0.66:
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
            trend_label = _TREND_IDX_LABELS.get(r["trend"], r["trend"])
            trend_color = _BULLISH if r["trend"] == _TREND_IDX_BULLISH else (_BEARISH if r["trend"] == _TREND_IDX_BEARISH else _WHITE)
            strength_str = f"({r['strength']:+.2f})" if "strength" in r else ""
            lines.append(f"  [{icon}] {r['name']}: {trend_color}{trend_label}{strength_str}{_RST}")
        bullish_count = sum(1 for r in market_results if r["trend"] == _TREND_IDX_BULLISH)
        bearish_count = sum(1 for r in market_results if r["trend"] == _TREND_IDX_BEARISH)
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
