import csv
import logging
import os
from datetime import datetime

# A股交易成本: 印花税0.05%(卖出) + 佣金约0.025%(买卖各一次) ≈ 0.1% 单边
# 来回成本约 0.15%, 涨跌超过此阈值才算有效 hit
_ROUND_TRIP_COST = 0.0015  # 0.15%

_MAX_TRACK_DAYS = 14
_HIT_DAY_COLUMNS = [f"hit{d}" for d in range(2, _MAX_TRACK_DAYS + 1)]
_SUMMARY_MARKER = "===命中率==="
_DATA_VERSION = 3  # Bump to force re-fill all hit columns

_PREDICTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.csv")
_FIELDS = ["date", "symbol", "name", "price", "signal", "score", "hit"] + _HIT_DAY_COLUMNS

logger = logging.getLogger(__name__)


def _ensure_file() -> None:
    """Create directory and file with header if not exists."""
    os.makedirs(os.path.dirname(_PREDICTIONS_FILE), exist_ok=True)

    if not os.path.exists(_PREDICTIONS_FILE):
        with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()


def _write_predictions(predictions: list[dict]) -> None:
    """Write all predictions to CSV, sorted by date descending, with accuracy summary row."""
    _ensure_file()
    # Project each row onto _FIELDS, dropping legacy fields and filling missing ones
    cleaned = [{f: p.get(f, "") for f in _FIELDS} for p in predictions]
    sorted_preds = sorted(cleaned, key=lambda p: p["date"], reverse=True)

    # Build summary row: date=marker with version, hit columns show accuracy
    summary_row = {field: "" for field in _FIELDS}
    summary_row["date"] = f"{_SUMMARY_MARKER} v{_DATA_VERSION}"
    all_hit_cols = ["hit"] + _HIT_DAY_COLUMNS
    for col in all_hit_cols:
        verified = [p for p in sorted_preds if _parse_hit_value(p.get(col, ""))]
        if verified:
            hits = sum(1 for p in verified if _parse_hit_value(p.get(col, "")) == "1")
            summary_row[col] = f"{hits}/{len(verified)} ({hits / len(verified) * 100:.1f}%)"

    with open(_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_preds)
        writer.writerow(summary_row)


def record_prediction(symbol: str, name: str, price: float, signal: str, score: int) -> None:
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
        "hit": "",
    }
    for col in _HIT_DAY_COLUMNS:
        new_row[col] = ""

    predictions = read_predictions()

    # Remove existing rows for same day + same stock
    predictions = [p for p in predictions if not (p["date"] == today and p["symbol"] == symbol)]

    predictions.append(new_row)
    _write_predictions(predictions)

    logger.info("Prediction recorded: %s(%s) signal=%s score=%d price=%.2f",
                name, symbol, signal, score, price)


def _migrate_predictions(predictions: list[dict], fetch_name_fn) -> list[dict]:
    """Backfill missing 'name' field for old predictions and re-save."""
    from data_source import get_stock_name
    changed = False
    for pred in predictions:
        if not pred.get("name") or not pred["name"].strip():
            pred["name"] = get_stock_name(pred["symbol"])
            changed = True
    if changed:
        logger.info("Migration: backfilled 'name' field for predictions")
        _write_predictions(predictions)
    return predictions


def _migrate_hit_columns(predictions: list[dict]) -> list[dict]:
    """Backfill missing hit2-hit14 columns for old CSV records."""
    changed = False
    for pred in predictions:
        for col in _HIT_DAY_COLUMNS:
            if col not in pred:
                pred[col] = ""
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

    # Extract and filter out summary row; check data version
    summary_version = None
    other_rows = []
    for p in predictions:
        date_val = p.get("date", "")
        if date_val.startswith(_SUMMARY_MARKER):
            # Extract version: "===命中率=== v2"
            parts = date_val.split(" v")
            if len(parts) == 2:
                try:
                    summary_version = int(parts[1])
                except ValueError:
                    pass
        else:
            other_rows.append(p)
    predictions = other_rows

    predictions = _migrate_predictions(predictions, None)
    predictions = _migrate_hit_columns(predictions)

    # If data version changed (or no version found), clear all hit columns to force re-fill
    if summary_version is None or summary_version < _DATA_VERSION:
        all_hit_cols = ["hit"] + _HIT_DAY_COLUMNS
        for p in predictions:
            for col in all_hit_cols:
                p[col] = ""
        _write_predictions(predictions)

    # Normalize date format (2026/3/24 → 2026-03-24) for consistent dedup
    for p in predictions:
        if "/" in p["date"]:
            parts = p["date"].split("/")
            p["date"] = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"

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
    Backfill hit columns for predictions, filling hit through hit14.

    Args:
        fetch_actual_fn: Function that takes (symbol, pred_date, max_days) and returns
                        (base_close, closes_list) where base_close is the close on
                        pred_date from the data source, and closes_list has length
                        max_days with subsequent close prices (None if unavailable).

    Returns:
        Count of records that were backfilled (at least day-1).
    """
    predictions = read_predictions()

    if not predictions:
        return 0

    backfill_count = 0

    for pred in predictions:
        symbol = pred["symbol"]
        pred_date = pred["date"]
        signal = pred["signal"]

        # Determine which days still need filling
        # Old format ("0"/"1") or empty is considered unfilled
        all_hit_cols = ["hit"] + _HIT_DAY_COLUMNS
        last_filled = 0
        for d, col in enumerate(all_hit_cols, start=1):
            raw = pred.get(col, "").strip()
            if raw and raw not in ("0", "1"):
                last_filled = d
            else:
                break
        start_day = last_filled + 1

        if start_day > _MAX_TRACK_DAYS:
            continue  # All 14 days already filled

        # Fetch base close and subsequent closes from data source
        max_days_needed = _MAX_TRACK_DAYS
        base_close, closes = fetch_actual_fn(symbol, pred_date, max_days_needed)

        if base_close is None or not closes:
            continue

        # Fill each day using base_close from data source as reference price
        for d in range(start_day, _MAX_TRACK_DAYS + 1):
            day_idx = d - 1
            if day_idx >= len(closes) or closes[day_idx] is None:
                break

            close_price = closes[day_idx]
            actual_change = (close_price - base_close) / base_close
            hit = _calculate_hit(signal, actual_change)
            change_pct = f"{actual_change * 100:+.2f}%"
            hit_val = "1" if hit else "0"
            col = "hit" if d == 1 else f"hit{d}"
            pred[col] = f"({change_pct} {hit_val})"

            if d == 1:
                backfill_count += 1

    # Always rewrite to keep summary row up-to-date
    _write_predictions(predictions)

    logger.info("Backfill complete: %d records backfilled out of %d total", backfill_count, len(predictions))

    return backfill_count


def _parse_hit_value(raw: str) -> str:
    """Extract hit status ('0' or '1') from a hit column value.

    Supports both new format '(+0.03% 1)' and legacy format '1'.
    """
    raw = raw.strip()
    if not raw:
        return ""
    if raw in ("0", "1"):
        return raw
    if raw.startswith("(") and raw.endswith(")"):
        inner = raw[1:-1].strip()
        parts = inner.rsplit(" ", 1)
        if len(parts) == 2 and parts[1] in ("0", "1"):
            return parts[1]
    return raw


def _parse_hit_change(raw: str) -> float | None:
    """Extract change percentage from a hit column value.

    Returns the change as a float (e.g., -3.40 for -3.40%), or None if unavailable.
    Supports new format '(+0.03% 1)' only.
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("(") and raw.endswith(")"):
        inner = raw[1:-1].strip()
        parts = inner.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                return float(parts[0].replace("%", ""))
            except ValueError:
                return None
    return None


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
            "by_day": {},
        }

    # Filter predictions with actual results
    verified = [p for p in predictions if _parse_hit_value(p.get("hit", ""))]

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
            "by_day": {},
        }

    # Calculate overall accuracy
    hits = sum(1 for p in verified if _parse_hit_value(p["hit"]) == "1")
    overall = (hits / verified_count) * 100

    # Calculate profit/loss ratio from directional signals (exclude 观望)
    directional = [p for p in verified if _normalize_signal(p["signal"]) != "观望"]

    win_changes = []
    loss_changes = []
    for p in directional:
        change = _parse_hit_change(p.get("hit", ""))
        if change is None:
            continue
        is_hit = _parse_hit_value(p["hit"]) == "1"
        signal = _normalize_signal(p["signal"])
        is_sell = signal in ("卖出", "强烈卖出")
        if is_hit:
            win_changes.append(abs(change) if is_sell else change)
        else:
            loss_changes.append(-abs(change) if is_sell else change)

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
        dir_hits = sum(1 for p in directional if _parse_hit_value(p["hit"]) == "1")
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
            signal_hits = sum(1 for p in signal_preds if _parse_hit_value(p["hit"]) == "1")
            by_signal[signal] = {
                "total": signal_total,
                "hits": signal_hits,
                "accuracy": (signal_hits / signal_total) * 100,
            }

    # Calculate by day (day 1 through day 14)
    by_day = {}
    for d in range(1, _MAX_TRACK_DAYS + 1):
        col = "hit" if d == 1 else f"hit{d}"
        day_verified = [p for p in verified if _parse_hit_value(p.get(col, ""))]
        if day_verified:
            day_hits = sum(1 for p in day_verified if _parse_hit_value(p[col]) == "1")
            day_total = len(day_verified)
            by_day[d] = {
                "verified": day_total,
                "hits": day_hits,
                "accuracy": (day_hits / day_total) * 100,
            }

    logger.info("Accuracy calculated: total=%d verified=%d overall=%.1f%% profit=%.2f%% loss=%.2f%% pl_ratio=%s",
                 total, verified_count, overall if verified_count > 0 else 0,
                 avg_profit if avg_profit is not None else 0,
                 avg_loss if avg_loss is not None else 0,
                 f"{profit_loss_ratio:.2f}" if profit_loss_ratio is not None and profit_loss_ratio != float("inf") else "inf")

    return {
        "total": total,
        "verified": verified_count,
        "overall": overall,
        "by_signal": by_signal,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_loss_ratio": profit_loss_ratio,
        "expectancy": expectancy,
        "by_day": by_day,
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
        profit_str = f"{avg_profit:+.2f}%" if avg_profit is not None else "N/A"
        loss_str = f"{avg_loss:+.2f}%" if avg_loss is not None else "N/A"
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

    # Multi-day accuracy (only show days 1, 2, 5, 14)
    by_day = stats.get("by_day", {})
    if by_day:
        lines.append("")
        lines.append("多日命中率:")
        display_days = [1, 2, 5, 14]
        for d in display_days:
            if d in by_day:
                day = by_day[d]
                lines.append(f"  第{d}日: {day['hits']}/{day['verified']} ({day['accuracy']:.1f}%)")

    return "\n".join(lines)
