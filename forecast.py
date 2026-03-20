import argparse
import sys
from datetime import datetime

import pandas as pd

from data_source import get_stock_history, get_realtime_quote, get_stock_name
from indicators import calc_all_indicators
from analyzer import (
    load_config,
    calculate_stock_score,
    score_to_signal,
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


def is_trading_hours() -> bool:
    """Check if current time is within A-share trading hours."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hour_min = now.hour + now.minute / 60
    return (9.5 <= hour_min <= 11.5) or (13.0 <= hour_min <= 15.0)


def fetch_actual_close(symbol: str, pred_date: str) -> float | None:
    """Fetch actual closing price for the next trading day after pred_date (used by tracker backfill)."""
    try:
        df = get_stock_history(symbol)
        # Find the row matching pred_date
        df["date_str"] = df["date"].astype(str).str[:10]
        match = df[df["date_str"] == pred_date]
        if len(match) == 0:
            return None
        idx = match.index[0]
        if idx + 1 < len(df):
            return float(df.iloc[idx + 1]["close"])
        return None
    except Exception:
        return None


def analyze_stock(symbol: str, config: dict, refresh: bool = False):
    """Analyze a single stock and return (report_string, signal, score) or None on error."""
    try:
        df = get_stock_history(symbol, refresh=refresh)
    except RuntimeError as e:
        print(f"  [错误] {e}")
        return None

    if df is None or len(df) < 30:
        print(f"  [错误] {symbol} 数据不足 ({len(df) if df is not None else 0} 行)")
        return None

    intraday = is_trading_hours()
    realtime_data = None

    if intraday:
        realtime_data = get_realtime_quote(symbol)
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

    df = calc_all_indicators(df)
    raw_score, ind_results = calculate_stock_score(df, config)
    modifier, market_results = calculate_market_modifier(config)
    final_score = max(10, min(90, int(raw_score + modifier)))
    signal = score_to_signal(final_score)
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
    )

    record_prediction(symbol, stock_name, float(df.iloc[-1]["close"]), signal, final_score, final_score)

    return report, signal, final_score


def main():
    parser = argparse.ArgumentParser(description="A股量化分析工具")
    parser.add_argument("symbols", nargs="*", help="股票代码 (如 002602)")
    parser.add_argument("--review", action="store_true", help="仅显示预测自检报告")
    parser.add_argument("--refresh", action="store_true", help="强制刷新缓存数据")
    args = parser.parse_args()

    config = load_config()

    # Backfill existing predictions before new analysis
    backfill_predictions(fetch_actual_close)

    if args.review:
        stats = calculate_accuracy()
        print(format_accuracy_report(stats))
        return

    symbols = args.symbols or config.get("stocks", [])
    if not symbols:
        print("未指定股票代码，请使用参数或在 config.json 中配置 stocks")
        sys.exit(1)

    multiple = len(symbols) > 1
    reports = []

    if multiple:
        print(f"正在分析 {len(symbols)} 只股票...\n")

    for i, symbol in enumerate(symbols):
        if multiple:
            print(f"{'=' * 50}")
            print(f"  {i + 1}/{len(symbols)} | {symbol}")
            print(f"{'=' * 50}")

        result = analyze_stock(symbol, config, args.refresh)
        if result is None:
            continue
        report, signal, score = result
        print(report)
        reports.append({"symbol": symbol, "signal": signal, "score": score})

    # Multi-stock summary
    if multiple and reports:
        reports.sort(key=lambda r: r["score"], reverse=True)
        buy_stocks = [r for r in reports if r["signal"] in ("强烈买入", "买入")]
        sell_stocks = [r for r in reports if r["signal"] in ("强烈卖出", "卖出")]
        neutral_stocks = [r for r in reports if r["signal"] == "观望"]

        print("--- 每日汇总 ---")
        if buy_stocks:
            items = ", ".join(f"{r['symbol']} ({r['signal']}, {r['score']}%)" for r in buy_stocks)
            print(f"关注: {items}")
        if sell_stocks:
            items = ", ".join(f"{r['symbol']} ({r['signal']}, {r['score']}%)" for r in sell_stocks)
            print(f"回避: {items}")
        if neutral_stocks:
            items = ", ".join(f"{r['symbol']} ({r['score']}%)" for r in neutral_stocks)
            print(f"中性: {items}")

    # Show accuracy stats after all analyses
    stats = calculate_accuracy()
    if stats["verified"] > 0:
        print("")
        print(format_accuracy_report(stats))


if __name__ == "__main__":
    main()
