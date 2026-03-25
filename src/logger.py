"""File-only logging module for stock quantitative analyzer.

Configures Python's ``logging`` to write exclusively to daily log files
under ``data/logs/``. No console handler is attached, so log output
never interferes with the ANSI-formatted reports printed to stdout.

Usage::

    from src.logger import setup_logging
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Application started")
"""

import logging
import os
from datetime import datetime, timedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "data", "logs")
_KEEP_DAYS = 3

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _cleanup_old_logs(log_dir: str, keep_days: int = _KEEP_DAYS) -> None:
    """Delete log files older than *keep_days*."""
    if not os.path.isdir(log_dir):
        return

    cutoff = datetime.now() - timedelta(days=keep_days)
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        # Parse date from filename: 2026-03-24.log
        date_str = filename[:-4]
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if file_date < cutoff:
            try:
                os.remove(os.path.join(log_dir, filename))
            except OSError:
                pass  # File may be locked on Windows; safe to skip


def setup_logging() -> None:
    """Configure the root logger to write to a daily log file.

    - Creates ``data/logs/`` if it does not exist.
    - Cleans up log files older than 3 days.
    - Adds a single ``FileHandler`` to the root logger (no console output).
    - Log level: DEBUG (individual modules control their own verbosity).
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    _cleanup_old_logs(_LOG_DIR)

    log_file = os.path.join(_LOG_DIR, datetime.now().strftime("%Y-%m-%d") + ".log")

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger()
    # Avoid duplicate handlers when called more than once
    if not root.handlers:
        root.setLevel(logging.WARNING)
        root.addHandler(handler)
        # App modules use their own loggers — set them to DEBUG
        for name in ("src.forecast", "src.data_source", "src.indicators",
                      "src.analyzer", "src.tracker", "src.wecom",
                      "src.market", "src.scoring", "src.report"):
            logging.getLogger(name).setLevel(logging.DEBUG)
