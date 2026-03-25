import argparse
import logging
import re
import sys
from datetime import datetime

import pandas as pd

from logger import setup_logging
from data_source import get_stock_history, get_realtime_quote, get_stock_name, _stock_cache_path, _save_cache
from indicators import calc_all_indicators
from analyzer import (
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
from tracker import (
    record_prediction,
    backfill_predictions,
    calculate_accuracy,
    format_accuracy_report,
)
from wecom import push_reports

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
    if 9.5 <= hour_min <= 11.5:
        return "盘中实时分析"
    if 11.5 < hour_min < 13.0:
        return "午间休市 (上午收盘数据)"
    if 13.0 <= hour_min <= 15.0:
        return "盘中实时分析"
    if 15.0 < hour_min <= 17.0:
        return "盘后分析 (今日收盘数据)"
    return "盘后分析"


def fetch_actual_closes(symbol: str, pred_date: str, max_days: int = 14) -> tuple[float | None, list[float | None]]:
    """Fetch base close and subsequent closes for backfill.

    Priority: cached data (via get_stock_history) -> raw API fetch.
    qfq-adjusted prices preserve percentage changes within the same series,
    so they are safe for actual_change calculation.

    Returns (base_close, closes) where base_close is the close on pred_date,
    and closes is a list of length max_days with close prices for subsequent
    trading days (None if unavailable).
    """
    # 1. Try cached data first (qfq, but percentage changes are preserved)
    try:
        df = get_stock_history(symbol)
        if df is not None and not df.empty:
            match = df[df["date"].astype(str).str[:10] == pred_date]
            if len(match) > 0:
                idx = match.index[0]
                base_close = float(df.iloc[idx]["close"])
                closes = []
                for i in range(1, max_days + 1):
                    if idx + i < len(df):
                        closes.append(float(df.iloc[idx + i]["close"]))
                    else:
                        closes.append(None)
                if any(c is not None for c in closes):
                    return (base_close, closes)
    except Exception:
        pass

    # 2. Fallback to raw API (existing logic)
    try:
        from data_source import _fetch_via_akshare, _fetch_via_sina, _dedup_and_sort, _is_hk_stock
        from data_source import _hk_code, _exchange_prefix

        if _is_hk_stock(symbol):
            import akshare as ak
            code = _hk_code(symbol)
            df = ak.stock_hk_daily(symbol=code, adjust="")
        else:
            prefix = _exchange_prefix(symbol)
            full_code = f"{prefix}{symbol}"
            df = _fetch_via_akshare_raw(full_code)
            if df is None or df.empty:
                df = _fetch_via_sina(full_code)
            if df is None or df.empty:
                return (None, [None] * max_days)

        if df is None or df.empty:
            return (None, [None] * max_days)

        df["date_str"] = df["date"].astype(str).str[:10]
        match = df[df["date_str"] == pred_date]
        if len(match) == 0:
            return (None, [None] * max_days)
        idx = match.index[0]
        base_close = float(df.iloc[idx]["close"])
        closes = []
        for i in range(1, max_days + 1):
            if idx + i < len(df):
                closes.append(float(df.iloc[idx + i]["close"]))
            else:
                closes.append(None)
        return (base_close, closes)
    except Exception:
        return (None, [None] * max_days)


def _fetch_via_akshare_raw(full_code: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV data via akshare WITHOUT adjustment (raw prices)."""
    try:
        import akshare as ak
        df = ak.stock_zh_a_daily(symbol=full_code, adjust="")
        if df is None or df.empty:
            return None
        from data_source import _CSV_COLUMNS
        df = df.rename(columns={"day": "date"})
        df = df[_CSV_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return None


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


def analyze_stock(symbol: str, config: dict, refresh: bool = False):
    """Analyze a single stock and return (report_string, signal, score) or None on error."""
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
            today_str = datetime.now().strftime("%Y-%m-%d")
            last_date = str(df.iloc[-1]["date"])[:10]
            if last_date != today_str:
                new_row = {
                    "date": today_str,
                    "open": realtime_data["open"],
                    "high": realtime_data["high"],
                    "low": realtime_data["low"],
                    "close": realtime_data["close"],
                    "volume": realtime_data["volume"],
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                _save_cache(df, _stock_cache_path(symbol))

    df = calc_all_indicators(df)

    # During intraday, override calculated vol_ratio with time-adjusted value:
    # - A-share: use Tencent API volume_ratio (field 49, exchange-provided)
    # - HK stock: use Tencent API volume_ratio (field 50), fallback to time-adjusted calc
    if intraday and realtime_data and realtime_data.get("volume_ratio") is not None:
        df.iloc[-1, df.columns.get_loc("vol_ratio")] = realtime_data["volume_ratio"]
    elif intraday and realtime_data:
        from data_source import _is_hk_stock as _check_hk
        rt_vol = realtime_data.get("volume", 0)
        if rt_vol and rt_vol > 0 and len(df) >= 6:
            prev5_avg = df.iloc[-6:-1]["volume"].astype(float).mean()
            if prev5_avg > 0:
                if _check_hk(symbol):
                    traded_min = _hk_traded_minutes()
                    total_min = 330
                else:
                    traded_min = _a_share_traded_minutes()
                    total_min = 240
                if traded_min > 0:
                    vol_ratio = (rt_vol / traded_min) / (prev5_avg / total_min)
                    df.iloc[-1, df.columns.get_loc("vol_ratio")] = round(vol_ratio, 2)
    raw_score, ind_results, trend_status = calculate_stock_score(df, config)
    modifier, market_results = calculate_market_modifier(config, intraday=intraday)
    final_score = max(-100, min(100, int(raw_score + modifier)))
    signal = calculate_signal(final_score, trend_status)
    key_levels = calculate_key_levels(df)
    risk_alerts = generate_risk_alerts(df, final_score)
    position_advice = calculate_position_advice(final_score, key_levels)

    stock_name = get_stock_name(symbol)

    report = format_report(
        symbol=symbol,
        stock_name=stock_name,
        df=df,
        score=final_score,
        indicator_results=ind_results,
        market_modifier=modifier,
        market_results=market_results,
        key_levels=key_levels,
        risk_alerts=risk_alerts,
        position_advice=position_advice,
        is_intraday=intraday,
        realtime_data=realtime_data,
        session_label=session_label,
        trend_status=trend_status,
    )

    record_prediction(symbol, stock_name, float(df.iloc[-1]["close"]), signal, final_score)

    logger.info("Analysis complete: %s(%s) score=%d signal=%s", stock_name, symbol, final_score, signal)

    return report, signal, final_score


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

    for i, symbol in enumerate(symbols):
        if multiple:
            print(f"\n{'=' * 50}")
            print(f"  {i + 1}/{len(symbols)} | {symbol}")
            print(f"{'=' * 50}\n")

        result = analyze_stock(symbol, config, args.refresh)
        if result is None:
            continue
        report, signal, score = result
        print(report)
        stock_name = get_stock_name(symbol)
        reports.append({"symbol": symbol, "name": stock_name, "signal": signal, "score": score, "report": report})

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
