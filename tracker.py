import csv
import os
from datetime import datetime

_PREDICTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.csv")
_FIELDS = ["date", "symbol", "price", "signal", "score", "pred_up_prob", "actual_change", "hit"]


def _ensure_file() -> None:
    """Create directory and file with header if not exists."""
    os.makedirs(os.path.dirname(_PREDICTIONS_FILE), exist_ok=True)

    if not os.path.exists(_PREDICTIONS_FILE):
        with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()


def record_prediction(symbol: str, price: float, signal: str, score: int, prob: int) -> None:
    """
    Write a new row to predictions.csv.

    Args:
        symbol: Stock symbol (e.g., '002602')
        price: Prediction price
        signal: Trading signal (Strong Buy, Buy, Hold, Sell, Strong Sell)
        score: Analysis score (0-100)
        prob: Predicted upward probability (0-100)
    """
    _ensure_file()

    today = datetime.now().strftime("%Y-%m-%d")

    with open(_PREDICTIONS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writerow({
            "date": today,
            "symbol": symbol,
            "price": f"{price:.2f}",
            "signal": signal,
            "score": score,
            "pred_up_prob": prob,
            "actual_change": "",
            "hit": ""
        })


def read_predictions() -> list[dict]:
    """
    Read all rows from predictions.csv as list of dicts.

    Returns:
        List of prediction dictionaries. Empty list if file is empty.
    """
    _ensure_file()

    if not os.path.exists(_PREDICTIONS_FILE):
        return []

    with open(_PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


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
        with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(predictions)

    return backfill_count


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

    if signal in ("Strong Buy", "Buy"):
        return actual_up
    elif signal in ("Strong Sell", "Sell"):
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
    signals = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    by_signal = {}

    for signal in signals:
        signal_preds = [p for p in verified if p["signal"] == signal]
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
    lines = ["--- Prediction Self-Reflection ---"]

    lines.append(f"Total predictions: {stats['total']}")
    lines.append(f"Verified samples: {stats['verified']}")

    if stats['verified'] == 0:
        lines.append("No verified predictions yet.")
        return "\n".join(lines)

    verified = stats['verified']
    overall = stats['overall']
    hits = sum(s['hits'] for s in stats['by_signal'].values())

    lines.append(f"Overall accuracy: {hits}/{verified} ({overall:.1f}%)")
    lines.append("")

    # Format each signal
    for signal, signal_stats in stats['by_signal'].items():
        signal_hits = signal_stats['hits']
        signal_total = signal_stats['total']
        accuracy = signal_stats['accuracy']
        lines.append(f"  {signal} accuracy: {signal_hits}/{signal_total} ({accuracy:.1f}%)")

    return "\n".join(lines)
