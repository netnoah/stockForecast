import argparse
import logging
import re
import sys
from datetime import datetime

import pandas as pd

from .logger import setup_logging
from .data_source import get_stock_history, get_realtime_quote, get_stock_name, merge_intraday_row, fetch_actual_closes, is_hk_stock
from .indicators import calc_all_indicators
from .analyzer import (
    load_config,
    calculate_stock_score,
    score_to_signal,
    calculate_signal,
    calculate_market_modifier,
    calculate_key_levels,
    generate_risk_alerts,
    calculate_position_advice,
    format_report,
)
from .tracker import (
    record_prediction,
    backfill_predictions,
    calculate_accuracy,
    format_accuracy_report,
)
from .wecom import push_reports
from .models import AnalysisResult

logger = logging.getLogger(__name__)


def is_trading_hours() -> bool:
    """Check if current time is within or after an A-share trading session on a trading day.

    Returns True from 9:30 through the end of the day on weekdays.  This covers
    intraday, post-close, and evening analysis — the realtime quote APIs
    (Sina/Tencent) remain available well after market close.
    """
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hour_min = now.hour + now.minute / 60
    return hour_min >= 9.5


def _session_label() -> str:
    """Return a label describing the current market session."""
    now = datetime.now()
    hour_min = now.hour + now.minute / 60
    if 9.5 <= hour_min < 11.5:
        return "盘中实时分析"
    if 11.5 < hour_min < 13.0:
        return "午间休市 (上午收盘数据)"
    if 13.0 <= hour_min <= 15.0:
        return "盘中实时分析"
    if 15.0 < hour_min <= 17.0:
        return "盘后分析 (今日收盘数据)"
    return "盘后分析"


def _a_share_traded_minutes() -> int:
    """Return elapsed trading minutes for A-share market today.

    A-share: 9:30-11:30 (120 min) + 13:00-15:00 (120 min) = 240 min/day.
    Morning break 10:15-10:30 is not counted.
    """
    now = datetime.now()
    total_min = now.hour * 60 + now.minute

    morning_open = 9 * 60 + 30     # 9:30
    break_start = 10 * 60 + 15     # 10:15
    break_end = 10 * 60 + 30       # 10:30
    lunch_start = 11 * 60 + 30     # 11:30
    lunch_end = 13 * 60            # 13:00
    market_close = 15 * 60         # 15:00

    if total_min <= morning_open:
        return 0
    if total_min <= break_start:
        return total_min - morning_open
    if total_min <= break_end:
        return break_start - morning_open  # 45 min
    if total_min <= lunch_start:
        return (break_start - morning_open) + (total_min - break_end)  # 45 + (t-630)
    if total_min <= lunch_end:
        return 120  # full morning session
    if total_min <= market_close:
        return 120 + (total_min - lunch_end)
    return 240  # full day


def _hk_traded_minutes() -> int:
    """Return elapsed trading minutes for HK market today.

    HK market: 9:30-12:00 (150 min) + 13:00-16:00 (180 min) = 330 min/day.
    """
    now = datetime.now()
    total_min = now.hour * 60 + now.minute

    morning_open = 9 * 60 + 30    # 9:30
    morning_close = 12 * 60        # 12:00
    afternoon_open = 13 * 60       # 13:00
    afternoon_close = 16 * 60      # 16:00

    if total_min <= morning_open:
        return 0
    if total_min <= morning_close:
        return total_min - morning_open
    if total_min <= afternoon_open:
        return morning_close - morning_open  # 150
    if total_min <= afternoon_close:
        return (morning_close - morning_open) + (total_min - afternoon_open)
    return 330  # full day


def analyze_stock(symbol: str, config: dict, refresh: bool = False,
                  market_modifier: tuple[int, list[dict]] | None = None):
    """Analyze a single stock and return AnalysisResult or None on error."""
    try:
        df = get_stock_history(symbol, refresh=refresh)
    except RuntimeError as e:
        logger.error("Data fetch failed for %s: %s", symbol, e)
        print(f"  [错误] {e}")
        return None

    if df is None or len(df) < 30:
        logger.warning("Insufficient data for %s: %d rows", symbol, len(df) if df is not None else 0)
        print(f"  [错误] {symbol} 数据不足 ({len(df) if df is not None else 0} 行)")
        return None

    intraday = is_trading_hours()
    realtime_data = None
    session_label = ""

    if intraday:
        realtime_data = get_realtime_quote(symbol)
        session_label = _session_label()
        if realtime_data:
            df = merge_intraday_row(symbol, df, realtime_data)

    df = calc_all_indicators(df)

    # During intraday, override calculated vol_ratio with time-adjusted value:
    # - A-share: use Tencent API volume_ratio (field 49, exchange-provided)
    # - HK stock: use Tencent API volume_ratio (field 50), fallback to time-adjusted calc
    if intraday and realtime_data and realtime_data.get("volume_ratio") is not None:
        df = df.copy()
        df.loc[df.index[-1], "vol_ratio"] = realtime_data["volume_ratio"]
    elif intraday and realtime_data:
        rt_vol = realtime_data.get("volume", 0)
        if rt_vol and rt_vol > 0 and len(df) >= 6:
            prev5_avg = df.iloc[-6:-1]["volume"].astype(float).mean()
            if prev5_avg > 0:
                if is_hk_stock(symbol):
                    traded_min = _hk_traded_minutes()
                    total_min = 330
                else:
                    traded_min = _a_share_traded_minutes()
                    total_min = 240
                if traded_min > 0:
                    vol_ratio = (rt_vol / traded_min) / (prev5_avg / total_min)
                    df = df.copy()
                    df.loc[df.index[-1], "vol_ratio"] = round(vol_ratio, 2)
    raw_score, ind_results, trend_status = calculate_stock_score(df, config)
    if market_modifier is not None:
        modifier, market_results = market_modifier
    else:
        modifier, market_results = calculate_market_modifier(config, intraday=intraday)
    final_score = max(-100, min(100, int(raw_score + modifier)))
    signal = calculate_signal(final_score, trend_status)
    key_levels = calculate_key_levels(df)
    risk_alerts = generate_risk_alerts(df, final_score)
    position_advice = calculate_position_advice(final_score, key_levels)

    stock_name = get_stock_name(symbol)

    result = AnalysisResult(
        symbol=symbol,
        stock_name=stock_name,
        score=final_score,
        signal=signal,
        trend_status=trend_status,
        indicator_results=ind_results,
        market_modifier=modifier,
        market_results=market_results,
        key_levels=key_levels,
        risk_alerts=risk_alerts,
        position_advice=position_advice,
        is_intraday=intraday,
        realtime_data=realtime_data,
        session_label=session_label,
    )

    result.report = format_report(result, df)

    record_prediction(symbol, stock_name, float(df.iloc[-1]["close"]), signal, final_score)

    logger.info("Analysis complete: %s(%s) score=%d signal=%s", stock_name, symbol, final_score, signal)

    return result


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="A股量化分析工具")
    parser.add_argument("symbols", nargs="*", help="股票代码 (如 002602)")
    parser.add_argument("-l", "--list", action="store_true", help="读取 config.json 中的 stock_list 配置")
    parser.add_argument("--review", action="store_true", help="仅显示预测自检报告")
    parser.add_argument("--refresh", action="store_true", help="强制刷新缓存数据")
    args = parser.parse_args()

    config = load_config()

    if args.review:
        # Backfill first so review reflects latest data
        logger.info("Running in review mode")
        backfill_predictions(fetch_actual_closes)
        stats = calculate_accuracy()
        print(format_accuracy_report(stats))
        return

    if args.list:
        raw = config.get("stock_list", "")
        symbols = re.split(r"[,\s]+", raw.strip())
        symbols = [s for s in symbols if s]
    else:
        symbols = args.symbols
    if not symbols:
        print("未指定股票代码，请使用参数或在 config.json 中配置 stocks")
        sys.exit(1)

    logger.info("Program started: mode=analyze stocks=%s refresh=%s", symbols, args.refresh)

    multiple = len(symbols) > 1
    reports = []

    if multiple:
        print(f"正在分析 {len(symbols)} 只股票...\n")

    # Compute market modifier once for all stocks (not per-stock)
    intraday = is_trading_hours()
    precomputed_modifier = calculate_market_modifier(config, intraday=intraday)

    for i, symbol in enumerate(symbols):
        if multiple:
            print(f"\n{'=' * 50}")
            print(f"  {i + 1}/{len(symbols)} | {symbol}")
            print(f"{'=' * 50}\n")

        result = analyze_stock(symbol, config, args.refresh, market_modifier=precomputed_modifier)
        if result is None:
            continue
        print(result.report)
        reports.append({"symbol": result.symbol, "name": result.stock_name,
                        "signal": result.signal, "score": result.score, "report": result.report})

    # Multi-stock summary
    push_sections: list[str] = []
    if multiple and reports:
        reports.sort(key=lambda r: r["score"], reverse=True)
        buy_stocks = [r for r in reports if r["signal"] in ("强烈买入", "买入")]
        sell_stocks = [r for r in reports if r["signal"] in ("强烈卖出", "卖出")]
        neutral_stocks = [r for r in reports if r["signal"] == "观望"]

        summary_lines = ["--- 每日汇总 ---"]
        if buy_stocks:
            items = ", ".join(f"{r['name']}({r['symbol']}) {r['signal']} {r['score']}分" for r in buy_stocks)
            summary_lines.append(f"关注: {items}")
        if sell_stocks:
            items = ", ".join(f"{r['name']}({r['symbol']}) {r['signal']} {r['score']}分" for r in sell_stocks)
            summary_lines.append(f"回避: {items}")
        if neutral_stocks:
            items = ", ".join(f"{r['name']}({r['symbol']}) {r['score']}分" for r in neutral_stocks)
            summary_lines.append(f"中性: {items}")

        summary_text = "\n".join(summary_lines)
        print(summary_text)
        push_sections.append(summary_text)

    # Backfill existing predictions after analysis (cache is now fresh)
    logger.info("Running backfill for %d analyzed stocks", len(reports))
    backfill_predictions(fetch_actual_closes)

    # Show accuracy stats after all analyses
    stats = calculate_accuracy()
    if stats["verified"] > 0:
        accuracy_report = format_accuracy_report(stats)
        print("")
        print(accuracy_report)
        push_sections.append(accuracy_report)

    # Push to WeChat Work — ask once, send all collected sections
    if push_sections or reports:
        all_sections = [r["report"] for r in reports] + push_sections
        push_reports(config, all_sections, title="分析报告")


if __name__ == "__main__":
    main()
