"""
Data source module for A-Share stock quantitative analyzer.

Provides functions to fetch historical OHLCV data and real-time quotes
with local CSV caching and multi-source fallback (akshare -> Sina Finance -> Tencent).
Includes retry, circuit breaker, and request throttling for stability.
"""

import csv
import logging
import os
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SINA_HEADERS = {
    "Referer": "http://finance.sina.com.cn",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

_TENCENT_HEADERS = {
    "Referer": "http://finance.qq.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

_SINA_KLINE_URL = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData?symbol={full_code}&scale=240&ma=no&datalen=1000"
)

_SINA_REALTIME_URL = "http://hq.sinajs.cn/list={full_code}"

_TENCENT_REALTIME_URL = "http://qt.gtimg.cn/q={full_code}"

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_HISTORY_DIR = os.path.join(_PROJECT_ROOT, "data", "history")

_CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN = 300  # seconds

_THROTTLE_MIN = 0.3
_THROTTLE_MAX = 0.8
_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_circuit_breaker: dict[str, dict] = {}
_last_request_time: float = 0.0

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry / Throttle / Circuit Breaker helpers
# ---------------------------------------------------------------------------


def _request_with_retry(url: str, headers: dict, max_retries: int = _MAX_RETRIES) -> requests.Response:
    """HTTP GET with exponential back-off retry.

    Retries on ConnectionError, TimeoutError, and requests.ConnectionError.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _throttle()
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp
        except (
            ConnectionError,
            TimeoutError,
            requests.ConnectionError,
            requests.Timeout,
        ) as exc:
            last_exc = exc
            _logger.warning(
                "Request failed (attempt %d/%d): %s — %s",
                attempt, max_retries, url, exc,
            )
            if attempt < max_retries:
                wait = 2 ** attempt
                _logger.info("Retrying in %ds...", wait)
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _throttle() -> None:
    """Add a random delay between consecutive requests to avoid rate-limiting."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _THROTTLE_MIN:
        delay = random.uniform(_THROTTLE_MIN, _THROTTLE_MAX)
        time.sleep(delay)
    _last_request_time = time.time()


def _is_circuit_open(source_name: str) -> bool:
    """Check whether a data source is in its cool-down period."""
    state = _circuit_breaker.get(source_name)
    if state is None:
        return False
    if state["failures"] < _CIRCUIT_BREAKER_THRESHOLD:
        return False
    if time.time() >= state["cooldown_until"]:
        # Cool-down expired — give it another chance
        _logger.info("Circuit breaker for '%s' cool-down expired, re-enabling", source_name)
        state["failures"] = 0
        state["cooldown_until"] = 0.0
        return False
    return True


def _record_success(source_name: str) -> None:
    """Reset failure counter for a data source after a successful request."""
    state = _circuit_breaker.get(source_name)
    if state is not None and state["failures"] > 0:
        _logger.info("Source '%s' recovered, resetting circuit breaker", source_name)
        state["failures"] = 0
        state["cooldown_until"] = 0.0


def _record_failure(source_name: str) -> None:
    """Record a failure and possibly trip the circuit breaker."""
    state = _circuit_breaker.setdefault(
        source_name, {"failures": 0, "cooldown_until": 0.0}
    )
    state["failures"] += 1
    if state["failures"] >= _CIRCUIT_BREAKER_THRESHOLD:
        state["cooldown_until"] = time.time() + _CIRCUIT_BREAKER_COOLDOWN
        _logger.warning(
            "Circuit breaker tripped for '%s' (%d consecutive failures), "
            "cool-down for %ds",
            source_name, state["failures"], _CIRCUIT_BREAKER_COOLDOWN,
        )


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------


def _is_hk_stock(symbol: str) -> bool:
    """Return True if the symbol is a Hong Kong stock (e.g. 'hk2400')."""
    return symbol.lower().startswith("hk")


def _hk_code(symbol: str) -> str:
    """Extract the numeric HK stock code, zero-padded to 5 digits.

    'hk2400' -> '02400', 'hk00700' -> '00700'
    """
    digits = symbol[2:] if symbol.lower().startswith("hk") else symbol
    return digits.zfill(5)


def _exchange_prefix(code: str) -> str:
    """Return the exchange prefix for a stock/index code.

    'hk' prefix -> 'hk' (Hong Kong).
    Codes starting with 6, 9, 5 belong to Shanghai (sh).
    All others belong to Shenzhen (sz).
    """
    if _is_hk_stock(code):
        return "hk"
    first_char = code[0] if code else ""
    if first_char in ("6", "9", "5"):
        return "sh"
    return "sz"


def _cache_path(filename: str) -> str:
    """Return the full path for a cache CSV file, ensuring the directory exists."""
    path = os.path.join(_HISTORY_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _read_cache(filepath: str) -> pd.DataFrame | None:
    """Read a cached CSV file into a DataFrame, or return None if not found.

    Returns data sorted ascending by date (for indicator calculations).
    If the on-disk file is in ascending order, re-saves it as descending.
    """
    if not os.path.isfile(filepath):
        return None
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        if not df.empty and df["date"].iloc[0] < df["date"].iloc[-1]:
            _save_cache(df, filepath)
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to a CSV cache file, sorted by date descending."""
    df = df.copy()
    # Ensure date is string in YYYY-MM-DD format for CSV
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    df.to_csv(filepath, index=False, columns=_CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Historical data fetchers
# ---------------------------------------------------------------------------


def _fetch_via_akshare(full_code: str) -> pd.DataFrame | None:
    """Attempt to fetch daily OHLCV data via akshare. Returns None on failure."""
    try:
        import akshare as ak  # noqa: F811 — lazy import
    except ImportError:
        return None

    try:
        df = ak.stock_zh_a_daily(symbol=full_code, adjust="qfq")
        if df is None or df.empty:
            return None
        # akshare returns a DataFrame; ensure expected columns
        df = df.rename(columns={"day": "date"})
        df = df[_CSV_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return None


def _fetch_via_sina(full_code: str) -> pd.DataFrame | None:
    """Fetch daily K-line data from Sina Finance API. Returns None on failure."""
    url = _SINA_KLINE_URL.format(full_code=full_code)
    try:
        resp = _request_with_retry(url, _SINA_HEADERS)
        records = resp.json()
        if not records:
            return None
        df = pd.DataFrame(records)
        df = df.rename(columns={"day": "date"})
        df = df[_CSV_COLUMNS].copy()
        # Coerce numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        _record_success("sina_kline")
        return df
    except Exception as exc:
        _record_failure("sina_kline")
        _logger.error("Sina K-line fetch failed for %s: %s", full_code, exc)
        return None


def _fetch_via_akshare_hk(symbol: str) -> pd.DataFrame | None:
    """Fetch HK stock daily OHLCV data via akshare (Sina source). Returns None on failure."""
    try:
        import akshare as ak
    except ImportError:
        return None

    try:
        code = _hk_code(symbol)
        df = ak.stock_hk_daily(symbol=code, adjust="qfq")
        if df is None or df.empty:
            return None
        df = df[_CSV_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return None


def _fetch_history(full_code: str) -> pd.DataFrame:
    """Fetch historical data trying akshare first, then Sina as fallback.

    Raises RuntimeError if both sources fail.
    """
    if _is_hk_stock(full_code):
        df = _fetch_via_akshare_hk(full_code)
        if df is not None and not df.empty:
            return df
        raise RuntimeError(f"Failed to fetch HK stock data for {full_code}")

    df = _fetch_via_akshare(full_code)
    if df is not None and not df.empty:
        return df

    df = _fetch_via_sina(full_code)
    if df is not None and not df.empty:
        return df

    raise RuntimeError(f"Failed to fetch data for {full_code} from all sources")


def _fetch_incremental(full_code: str, after_date: str) -> pd.DataFrame:
    """Fetch data after a given date. Falls back to full fetch on failure."""
    if _is_hk_stock(full_code):
        # HK stocks: full fetch via akshare then filter (no incremental Sina K-line API)
        try:
            df = _fetch_via_akshare_hk(full_code)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                after_dt = pd.to_datetime(after_date)
                df = df[df["date"] > after_dt].copy()
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                return df
        except Exception:
            pass
        return _fetch_history(full_code)

    try:
        df = _fetch_via_sina(full_code)
        if df is not None and not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            after_dt = pd.to_datetime(after_date)
            df = df[df["date"] > after_dt].copy()
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass
    # Fallback: full fetch
    return _fetch_history(full_code)


def _dedup_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate dates and sort ascending by date."""
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Tencent real-time quote
# ---------------------------------------------------------------------------


def _fetch_realtime_via_tencent(symbol: str) -> dict | None:
    """Fetch A-share real-time quote from Tencent Finance (qt.gtimg.cn).

    Returns dict with keys: date, open, high, low, close, prev_close, volume, name.
    Returns None on failure.
    """
    if _is_hk_stock(symbol):
        return None

    prefix = _exchange_prefix(symbol)
    full_code = f"{prefix}{symbol}"
    url = _TENCENT_REALTIME_URL.format(full_code=full_code)

    try:
        resp = _request_with_retry(url, _TENCENT_HEADERS)
        resp.encoding = "gbk"
        content = resp.text

        # Response format: v_sh600519="1~贵州茅台~600519~1750.00~1735.00~..."
        # Fields (0-indexed after split by '~'):
        #   0: unknown, 1: name, 2: code, 3: price, 4: prev_close,
        #   5: open, 6: volume(lots), ..., 30: timestamp,
        #   31: change_amount, 32: change_pct,
        #   33: high, 34: low, ..., 36: volume(lots), 37: amount
        parts = content.split('"')
        if len(parts) < 2:
            return None

        fields = parts[1].split("~")
        if len(fields) < 35:
            return None

        price = float(fields[3]) if fields[3] else None
        if price is None or price <= 0:
            return None

        # Tencent volume is in lots (手), convert to shares
        volume_lots = float(fields[6]) if fields[6] else 0
        volume = volume_lots * 100

        quote = {
            "name": fields[1],
            "open": float(fields[5]) if fields[5] else None,
            "high": float(fields[33]) if fields[33] else None,
            "low": float(fields[34]) if fields[34] else None,
            "close": price,
            "prev_close": float(fields[4]) if fields[4] else None,
            "volume": volume,
            "date": fields[30] if len(fields) > 30 else "",
        }
        _record_success("tencent_realtime")
        _logger.debug("Tencent quote for %s: %.2f", full_code, price)
        return quote
    except Exception as exc:
        _record_failure("tencent_realtime")
        _logger.error("Tencent realtime fetch failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Sina real-time quote helpers (refactored)
# ---------------------------------------------------------------------------


def _fetch_realtime_via_sina(symbol: str) -> dict | None:
    """Fetch A-share real-time quote from Sina Finance.

    Returns dict with keys: date, open, high, low, close, prev_close, volume, name.
    Returns None on failure.
    """
    if _is_hk_stock(symbol):
        return None

    prefix = _exchange_prefix(symbol)
    full_code = f"{prefix}{symbol}"
    url = _SINA_REALTIME_URL.format(full_code=full_code)

    try:
        resp = _request_with_retry(url, _SINA_HEADERS)
        content = resp.text
        # Parse: var hq_str_sz002602="股票名,open,prev_close,now,high,low,..."
        parts = content.split('"')
        if len(parts) < 2:
            return None
        fields = parts[1].split(",")
        if len(fields) < 32:
            return None
        quote = {
            "date": fields[30],
            "open": float(fields[1]),
            "high": float(fields[4]),
            "low": float(fields[5]),
            "close": float(fields[3]),
            "prev_close": float(fields[2]),
            "volume": float(fields[8]),
            "name": fields[0],
        }
        _record_success("sina_realtime")
        return quote
    except Exception as exc:
        _record_failure("sina_realtime")
        _logger.error("Sina realtime fetch failed for %s: %s", symbol, exc)
        return None


def _fetch_hk_realtime_via_sina(symbol: str) -> dict | None:
    """Fetch HK stock real-time quote from Sina Finance with retry."""
    code = _hk_code(symbol)
    url = f"http://hq.sinajs.cn/list=rt_hk{code}"
    try:
        resp = _request_with_retry(url, _SINA_HEADERS)
        resp.raise_for_status()
        content = resp.text
        parts = content.split('"')
        if len(parts) < 2:
            return None
        fields = parts[1].split(",")
        if len(fields) < 19:
            return None
        return {
            "date": fields[17],
            "open": float(fields[2]),
            "high": float(fields[4]),
            "low": float(fields[5]),
            "close": float(fields[6]),
            "prev_close": float(fields[3]),
            "volume": float(fields[12]),  # Field 12 = shares; field 11 = amount (HKD)
            "name": fields[1],
        }
    except Exception as exc:
        _logger.error("Sina HK realtime fetch failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _cache_is_stale(latest_date: pd.Timestamp) -> bool:
    """Check if cached data is stale and needs refresh.

    Returns True when the market has closed for a new trading day but the
    cache doesn't contain that day's data yet.
    """
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if latest_date >= pd.Timestamp(today):
        return False  # Cache already has today's data

    # Cache is from a previous day — check if a new trading day has closed
    if now.weekday() < 5 and now.hour >= 15:
        return True  # Weekday past 15:00, market closed
    if now.weekday() < 5 and now.hour < 9:
        return True  # Weekday before open, yesterday's close is available
    if now.weekday() == 5:
        # Saturday: Friday's close should be available
        if latest_date < pd.Timestamp(today - timedelta(days=1)):
            return True
    return False


def _stock_cache_path(symbol: str) -> str:
    """Return the cache CSV path for a stock, including Chinese name in filename.

    Falls back to old naming (code-only) if the name cannot be retrieved,
    and migrates old files to the new naming convention.
    """
    if _is_hk_stock(symbol):
        code = _hk_code(symbol)
        name = get_stock_name(symbol)
        cache_file = f"hk{code}_{name}.csv"
        return _cache_path(cache_file)

    prefix = _exchange_prefix(symbol)
    full_code = f"{prefix}{symbol}"
    name = get_stock_name(symbol)

    new_cache_file = f"{full_code}_{name}.csv"
    new_filepath = _cache_path(new_cache_file)

    old_cache_file = f"{full_code}.csv"
    old_filepath = _cache_path(old_cache_file)

    # Migrate old file to new name if it exists
    if os.path.isfile(old_filepath) and not os.path.isfile(new_filepath):
        try:
            os.rename(old_filepath, new_filepath)
        except OSError:
            pass

    return new_filepath


def get_stock_history(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """Get historical daily OHLCV data for a stock.

    Args:
        symbol: Numeric stock code (e.g. '002602', '600519').
        refresh: If True, ignore cache and fetch fresh data.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
    """
    if _is_hk_stock(symbol):
        full_code = symbol  # e.g. 'hk2400' — already includes prefix
    else:
        prefix = _exchange_prefix(symbol)
        full_code = f"{prefix}{symbol}"
    cache_filepath = _stock_cache_path(symbol)

    if not refresh:
        cached = _read_cache(cache_filepath)
        if cached is not None and not cached.empty:
            latest_date = pd.to_datetime(cached["date"]).max()
            if not _cache_is_stale(latest_date):
                three_days_ago = datetime.now() - timedelta(days=3)
                if latest_date >= pd.Timestamp(three_days_ago):
                    return cached.copy()
            # Incremental update
            try:
                incremental = _fetch_incremental(full_code, latest_date.strftime("%Y-%m-%d"))
                if not incremental.empty:
                    merged = pd.concat([cached, incremental], ignore_index=True)
                    merged = _dedup_and_sort(merged)
                    _save_cache(merged, cache_filepath)
                    return merged.copy()
            except Exception:
                # If incremental fails, fall through to full fetch
                pass

    # Full fetch
    df = _fetch_history(full_code)
    df = _dedup_and_sort(df)
    _save_cache(df, cache_filepath)
    return df.copy()


def get_realtime_quote(symbol: str) -> dict | None:
    """Get real-time quote for a stock during trading hours.

    Supports both A-share and HK stock symbols.
    A-share uses Tencent (priority) -> Sina fallback with circuit breaker.

    Args:
        symbol: Stock code (e.g. '002602' for A-share, 'hk2400' for HK).

    Returns:
        Dict with keys: date, open, high, low, close, volume, name.
        Returns None if the quote cannot be retrieved.
    """
    if _is_hk_stock(symbol):
        return _fetch_hk_realtime_via_sina(symbol)

    # A-share: try Tencent first, then Sina, respecting circuit breakers
    if not _is_circuit_open("tencent_realtime"):
        _logger.debug("Trying Tencent realtime for %s", symbol)
        quote = _fetch_realtime_via_tencent(symbol)
        if quote is not None:
            return quote
        _logger.info("Tencent failed for %s, falling back to Sina", symbol)

    if not _is_circuit_open("sina_realtime"):
        _logger.debug("Trying Sina realtime for %s", symbol)
        quote = _fetch_realtime_via_sina(symbol)
        if quote is not None:
            return quote
        _logger.info("Sina failed for %s", symbol)

    return None


def get_stock_name(symbol: str) -> str:
    """Get the Chinese name of a stock.

    A-share uses Tencent (priority) -> Sina fallback with circuit breaker.

    Args:
        symbol: Stock code (e.g. '002602' for A-share, 'hk2400' for HK).

    Returns:
        Stock name string, or the symbol code itself on failure.
    """
    if _is_hk_stock(symbol):
        code = _hk_code(symbol)
        url = f"http://hq.sinajs.cn/list=rt_hk{code}"
        try:
            resp = _request_with_retry(url, _SINA_HEADERS)
            resp.raise_for_status()
            content = resp.text
            parts = content.split('"')
            if len(parts) < 2:
                return symbol
            fields = parts[1].split(",")
            name = fields[1] if len(fields) > 1 else ""
            return name if name else symbol
        except Exception:
            return symbol

    # A-share: try Tencent first, then Sina
    if not _is_circuit_open("tencent_realtime"):
        quote = _fetch_realtime_via_tencent(symbol)
        if quote is not None and quote.get("name"):
            return quote["name"]

    if not _is_circuit_open("sina_realtime"):
        prefix = _exchange_prefix(symbol)
        full_code = f"{prefix}{symbol}"
        url = _SINA_REALTIME_URL.format(full_code=full_code)
        try:
            resp = _request_with_retry(url, _SINA_HEADERS)
            resp.raise_for_status()
            content = resp.text
            parts = content.split('"')
            if len(parts) < 2:
                return symbol
            name = parts[1].split(",")[0]
            return name if name else symbol
        except Exception:
            return symbol

    return symbol


def get_index_realtime_quote(index_code: str) -> dict | None:
    """Get real-time quote for a market index using its full code.

    Unlike get_realtime_quote (which expects bare stock codes and adds an
    exchange prefix), this function accepts already-prefixed codes such as
    'sz399006' or 'sh000001'.

    Args:
        index_code: Full index code with exchange prefix (e.g. 'sz399006').

    Returns:
        Dict with keys: open, high, low, close, volume.
        Returns None if the quote cannot be retrieved.
    """
    url = _SINA_REALTIME_URL.format(full_code=index_code)
    try:
        resp = _request_with_retry(url, _SINA_HEADERS)
        parts = resp.text.split('"')
        if len(parts) < 2:
            return None
        fields = parts[1].split(",")
        if len(fields) < 6 or float(fields[3]) <= 0:
            return None
        return {
            "open": float(fields[1]),
            "high": float(fields[4]),
            "low": float(fields[5]),
            "close": float(fields[3]),
            "volume": float(fields[8]) if len(fields) > 8 else 0,
        }
    except Exception as exc:
        _logger.error("Index realtime fetch failed for %s: %s", index_code, exc)
        return None


def get_index_history(index_code: str, refresh: bool = False) -> pd.DataFrame:
    """Get historical daily OHLCV data for a market index.

    Args:
        index_code: Full index code with exchange prefix (e.g. 'sh000001').
        refresh: If True, ignore cache and fetch fresh data.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
    """
    cache_file = f"{index_code}.csv"
    cache_filepath = _cache_path(cache_file)

    if not refresh:
        cached = _read_cache(cache_filepath)
        if cached is not None and not cached.empty:
            latest_date = pd.to_datetime(cached["date"]).max()
            if not _cache_is_stale(latest_date):
                three_days_ago = datetime.now() - timedelta(days=3)
                if latest_date >= pd.Timestamp(three_days_ago):
                    return cached.copy()
            # Incremental update
            try:
                incremental = _fetch_incremental(index_code, latest_date.strftime("%Y-%m-%d"))
                if not incremental.empty:
                    merged = pd.concat([cached, incremental], ignore_index=True)
                    merged = _dedup_and_sort(merged)
                    _save_cache(merged, cache_filepath)
                    return merged.copy()
            except Exception:
                pass

    # Full fetch
    df = _fetch_history(index_code)
    df = _dedup_and_sort(df)
    _save_cache(df, cache_filepath)
    return df.copy()
