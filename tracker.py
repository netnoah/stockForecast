import csv
import os
from datetime import datetime

# A股交易成本: 印花税0.05%(卖出) + 佣金约0.025%(买卖各一次) ≈ 0.1% 单边
# 来回成本约 0.15%, 涨跌超过此阈值才算有效 hit
_ROUND_TRIP_COST = 0.0015  # 0.15%

_PREDICTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.csv")
_FIELDS = ["date", "symbol", "name", "price", "signal", "score", "pred_up_prob", "actual_change", "hit"]


def _ensure_file() -> None:
    """Create directory and file with header if not exists."""
    os.makedirs(os.path.dirname(_PREDICTIONS_FILE), exist_ok=True)

    if not os.path.exists(_PREDICTIONS_FILE):
        with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()


def _write_predictions(predictions: list[dict]) -> None:
    """Write all predictions to CSV, sorted by date descending."""
    _ensure_file()
    sorted_preds = sorted(predictions, key=lambda p: p["date"], reverse=True)
    with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_preds)


def record_prediction(symbol: str, name: str, price: float, signal: str, score: int, prob: int) -> None:
    """
    Write a prediction to predictions.csv.

    Deduplicates: same day + same stock keeps only the latest prediction.
    File is sorted by date descending (newest first).
    """
    _ensure_file()

    today = datetime.now().strftime("%Y-%m-%d")
    new_row = {
        "date": today,
        "symbol": symbol,
        "name": name,
        "price": f"{price:.2f}",
        "signal": signal,
        "score": score,
        "pred_up_prob": prob,
        "actual_change": "",
        "hit": "",
    }

    predictions = read_predictions()

    # Remove existing rows for same day + same stock
    predictions = [p for p in predictions if not (p["date"] == today and p["symbol"] == symbol)]

    predictions.append(new_row)
    _write_predictions(predictions)


def _migrate_predictions(predictions: list[dict], fetch_name_fn) -> list[dict]:
    """Backfill missing 'name' field for old predictions and re-save."""
    from data_source import get_stock_name
    changed = False
    for pred in predictions:
        if not pred.get("name") or not pred["name"].strip():
            pred["name"] = get_stock_name(pred["symbol"])
            changed = True
    if changed:
        _write_predictions(predictions)
    return predictions


def read_predictions() -> list[dict]:
    """
    Read all rows from predictions.csv as list of dicts.

    Deduplicates same-day same-stock entries, keeping the last one.
    Returns:
        List of prediction dictionaries. Empty list if file is empty.
    """
    _ensure_file()

    if not os.path.exists(_PREDICTIONS_FILE):
        return []

    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030'):
        try:
            with open(_PREDICTIONS_FILE, 'r', newline='', encoding=enc) as f:
                reader = csv.DictReader(f)
                predictions = list(reader)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        predictions = []
        reader = csv.DictReader(f)
        predictions = list(reader)

    predictions = _migrate_predictions(predictions, None)

    # Deduplicate: same day + same stock, keep last occurrence
    seen = set()
    deduped = []
    for p in reversed(predictions):
        key = (p["date"], p["symbol"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    deduped.reverse()

    if len(deduped) != len(predictions):
        _write_predictions(deduped)
        return deduped

    return deduped


def backfill_predictions(fetch_actual_fn) -> int:
    """
    Backfill actual results for predictions where actual_change is empty.

    Args:
        fetch_actual_fn: Function that takes (symbol, date) and returns
                        the next-day close price as float, or None if unavailable.

    Returns:
        Count of records that were backfilled.
    """
    predictions = read_predictions()

    if not predictions:
        return 0

    changes_made = False
    backfill_count = 0

    for pred in predictions:
        # Skip if already has actual_change
        if pred.get("actual_change") and pred.get("actual_change").strip():
            continue

        symbol = pred["symbol"]
        pred_date = pred["date"]
        pred_price = float(pred["price"])
        signal = pred["signal"]

        # Fetch next-day close price
        next_close = fetch_actual_fn(symbol, pred_date)

        if next_close is not None:
            # Calculate actual change
            actual_change = (next_close - pred_price) / pred_price

            # Determine hit
            hit = _calculate_hit(signal, actual_change)

            # Update prediction
            pred["actual_change"] = f"{actual_change * 100:+.2f}%"
            pred["hit"] = "1" if hit else "0"

            changes_made = True
            backfill_count += 1

    # Only rewrite file if changes were made
    if changes_made:
        _write_predictions(predictions)

    return backfill_count


def _normalize_signal(signal: str) -> str:
    """Normalize English signal names from old CSV data to Chinese."""
    _en_to_cn = {
        "Strong Buy": "强烈买入",
        "Buy": "买入",
        "Hold": "观望",
        "Sell": "卖出",
        "Strong Sell": "强烈卖出",
    }
    return _en_to_cn.get(signal, signal)


def _calculate_hit(signal: str, actual_change: float) -> bool:
    """
    Determine if a prediction was a hit.

    Buy/sell signals require direction correctness AND sufficient magnitude
    to cover round-trip trading cost (~0.15%). Hold signals count as hit
    when next-day fluctuation is within 1%.

    Args:
        signal: Trading signal
        actual_change: Actual price change (as decimal, e.g., 0.0123 for +1.23%)

    Returns:
        True if prediction was correct, False otherwise.
    """
    actual_up = actual_change > 0
    normalized = _normalize_signal(signal)

    if normalized in ("强烈买入", "买入"):
        return actual_up and abs(actual_change) > _ROUND_TRIP_COST
    elif normalized in ("强烈卖出", "卖出"):
        return not actual_up and abs(actual_change) > _ROUND_TRIP_COST
    else:  # Hold / 观望
        return abs(actual_change) < 0.01  # ±1%


def calculate_accuracy() -> dict:
    """
    Calculate prediction accuracy statistics including profit/loss ratio.

    Returns:
        Dictionary containing:
        - total: Total number of predictions
        - verified: Number of predictions with actual results
        - overall: Overall accuracy percentage (None if no verified predictions)
        - by_signal: Accuracy breakdown by signal type
        - avg_profit: Average return of winning trades (as %)
        - avg_loss: Average loss of losing trades (as %)
        - profit_loss_ratio: avg_profit / abs(avg_loss) (None if no losses)
        - expectancy: Expected return per trade (as %)
    """
    predictions = read_predictions()

    if not predictions:
        return {
            "total": 0,
            "verified": 0,
            "overall": None,
            "by_signal": {},
            "avg_profit": None,
            "avg_loss": None,
            "profit_loss_ratio": None,
            "expectancy": None,
        }

    # Filter predictions with actual results
    verified = [p for p in predictions if p.get("hit") and p.get("hit").strip()]

    total = len(predictions)
    verified_count = len(verified)

    if verified_count == 0:
        return {
            "total": total,
            "verified": 0,
            "overall": None,
            "by_signal": {},
            "avg_profit": None,
            "avg_loss": None,
            "profit_loss_ratio": None,
            "expectancy": None,
        }

    # Calculate overall accuracy
    hits = sum(1 for p in verified if p["hit"] == "1")
    overall = (hits / verified_count) * 100

    # Calculate profit/loss ratio from directional signals (exclude 观望)
    directional = [p for p in verified if _normalize_signal(p["signal"]) != "观望"]

    win_changes = []
    loss_changes = []
    for p in directional:
        change_str = p.get("actual_change", "").strip()
        if not change_str:
            continue
        change = float(change_str.replace("%", ""))  # already in percentage
        is_hit = p["hit"] == "1"
        if is_hit:
            win_changes.append(change)
        else:
            loss_changes.append(change)

    avg_profit = sum(win_changes) / len(win_changes) if win_changes else None
    avg_loss = sum(loss_changes) / len(loss_changes) if loss_changes else None

    if avg_profit is not None and avg_loss is not None and avg_loss != 0:
        profit_loss_ratio = avg_profit / abs(avg_loss)
    elif avg_profit is not None and avg_loss is None:
        profit_loss_ratio = float("inf")
    else:
        profit_loss_ratio = None

    # Expectancy = hit_rate * avg_profit - miss_rate * avg_loss
    if directional and avg_profit is not None:
        dir_hits = sum(1 for p in directional if p["hit"] == "1")
        dir_total = len(directional)
        hit_rate = dir_hits / dir_total
        miss_rate = 1 - hit_rate
        avg_l = abs(avg_loss) if avg_loss is not None else 0
        expectancy = hit_rate * avg_profit - miss_rate * avg_l
    else:
        expectancy = None

    # Calculate by signal
    signals = ["强烈买入", "买入", "观望", "卖出", "强烈卖出"]
    by_signal = {}

    for signal in signals:
        signal_preds = [p for p in verified if _normalize_signal(p["signal"]) == signal]
        if signal_preds:
            signal_total = len(signal_preds)
            signal_hits = sum(1 for p in signal_preds if p["hit"] == "1")
            by_signal[signal] = {
                "total": signal_total,
                "hits": signal_hits,
                "accuracy": (signal_hits / signal_total) * 100,
            }

    return {
        "total": total,
        "verified": verified_count,
        "overall": overall,
        "by_signal": by_signal,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_loss_ratio": profit_loss_ratio,
        "expectancy": expectancy,
    }


def format_accuracy_report(stats: dict) -> str:
    """
    Format accuracy statistics as a readable text report.

    Args:
        stats: Dictionary from calculate_accuracy()

    Returns:
        Formatted string report.
    """
    lines = ["--- 预测自检报告 ---"]

    lines.append(f"总预测次数: {stats['total']}")
    lines.append(f"已验证样本: {stats['verified']}")

    if stats['verified'] == 0:
        lines.append("暂无已验证的预测记录。")
        return "\n".join(lines)

    verified = stats['verified']
    overall = stats['overall']
    hits = sum(s['hits'] for s in stats['by_signal'].values())

    lines.append(f"整体准确率: {hits}/{verified} ({overall:.1f}%)")
    lines.append(f"交易成本阈值: {_ROUND_TRIP_COST * 100:.2f}%")

    # Profit/loss ratio
    avg_profit = stats.get("avg_profit")
    avg_loss = stats.get("avg_loss")
    pl_ratio = stats.get("profit_loss_ratio")
    expectancy = stats.get("expectancy")

    if avg_profit is not None or avg_loss is not None:
        profit_str = f"+{avg_profit:.2f}%" if avg_profit is not None else "N/A"
        loss_str = f"{avg_loss:.2f}%" if avg_loss is not None else "N/A"
        if pl_ratio is not None and pl_ratio != float("inf"):
            ratio_str = f"{pl_ratio:.2f}"
        elif pl_ratio == float("inf"):
            ratio_str = "inf (无亏损)"
        else:
            ratio_str = "N/A"
        lines.append(f"平均盈利: {profit_str} | 平均亏损: {loss_str}")
        lines.append(f"盈亏比: {ratio_str}")
        if expectancy is not None:
            exp_sign = "+" if expectancy >= 0 else ""
            lines.append(f"单笔期望收益: {exp_sign}{expectancy:.3f}%")
    lines.append("")

    # Format each signal
    for signal, signal_stats in stats['by_signal'].items():
        signal_hits = signal_stats['hits']
        signal_total = signal_stats['total']
        accuracy = signal_stats['accuracy']
        lines.append(f"  {signal} 准确率: {signal_hits}/{signal_total} ({accuracy:.1f}%)")

    return "\n".join(lines)
