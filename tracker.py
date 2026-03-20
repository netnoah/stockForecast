import csv
import os
from datetime import datetime

_PREDICTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.csv")
_FIELDS = ["date", "symbol", "name", "price", "signal", "score", "pred_up_prob", "actual_change", "hit"]


def _ensure_file() -> None:
    """Create directory and file with header if not exists."""
    os.makedirs(os.path.dirname(_PREDICTIONS_FILE), exist_ok=True)

    if not os.path.exists(_PREDICTIONS_FILE):
        with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()


def _write_predictions(predictions: list[dict]) -> None:
    """Write all predictions to CSV, sorted by date descending."""
    _ensure_file()
    sorted_preds = sorted(predictions, key=lambda p: p["date"], reverse=True)
    with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
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

    with open(_PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
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
            pred["actual_change"] = f"{actual_change:+.4f}"
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

    Args:
        signal: Trading signal
        actual_change: Actual price change (as decimal, e.g., 0.0123 for +1.23%)

    Returns:
        True if prediction was correct, False otherwise.
    """
    actual_up = actual_change > 0
    normalized = _normalize_signal(signal)

    if normalized in ("强烈买入", "买入"):
        return actual_up
    elif normalized in ("强烈卖出", "卖出"):
        return not actual_up
    else:  # Hold
        # Hold signals always count as miss per specification
        return False


def calculate_accuracy() -> dict:
    """
    Calculate prediction accuracy statistics.

    Returns:
        Dictionary containing:
        - total: Total number of predictions
        - verified: Number of predictions with actual results
        - overall: Overall accuracy percentage (None if no verified predictions)
        - by_signal: Accuracy breakdown by signal type
    """
    predictions = read_predictions()

    if not predictions:
        return {
            "total": 0,
            "verified": 0,
            "overall": None,
            "by_signal": {}
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
            "by_signal": {}
        }

    # Calculate overall accuracy
    hits = sum(1 for p in verified if p["hit"] == "1")
    overall = (hits / verified_count) * 100

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
                "accuracy": (signal_hits / signal_total) * 100
            }

    return {
        "total": total,
        "verified": verified_count,
        "overall": overall,
        "by_signal": by_signal
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
    lines.append("")

    # Format each signal
    for signal, signal_stats in stats['by_signal'].items():
        signal_hits = signal_stats['hits']
        signal_total = signal_stats['total']
        accuracy = signal_stats['accuracy']
        lines.append(f"  {signal} 准确率: {signal_hits}/{signal_total} ({accuracy:.1f}%)")

    return "\n".join(lines)
