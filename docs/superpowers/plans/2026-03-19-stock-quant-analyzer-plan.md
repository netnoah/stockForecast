# Stock Quantitative Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a command-line A-share stock analysis tool with multi-source data fetching, technical indicators, weighted scoring, broad market modifier, intraday/after-hours modes, and prediction self-reflection.

**Architecture:** Modular CLI tool with 5 source files — `data_source.py` (fetching + caching), `indicators.py` (pure numpy/pandas calculations), `analyzer.py` (scoring + report formatting), `tracker.py` (CSV-based prediction logging + accuracy stats), `forecast.py` (CLI entry + orchestration). Config-driven via `config.json`. No external TA libraries.

**Tech Stack:** Python 3.x, pandas, numpy, requests, akshare (optional fallback)

---

## File Map

| File | Responsibility | Lines (est.) |
|------|---------------|-------------|
| `config.json` | Stock list, indicator weights, market modifier settings | ~25 |
| `data_source.py` | Multi-source fetch (akshare → Sina), CSV cache, incremental update, real-time quote | ~180 |
| `indicators.py` | MA, MACD, RSI, Bollinger, KDJ, Volume — all pure pandas/numpy | ~150 |
| `analyzer.py` | Weighted scoring, signal rating, market modifier, risk alerts, key levels, position advice, report formatting | ~250 |
| `tracker.py` | Prediction CSV logging, backfill actuals, accuracy statistics | ~120 |
| `forecast.py` | CLI arg parsing, trading-hours detection, workflow orchestration, multi-stock summary | ~150 |

---

### Task 1: Project Scaffolding + Config

**Files:**
- Create: `config.json`
- Create: `data/` directory structure
- Modify: `forecast.py` (delete old content, replace with placeholder main)

- [ ] **Step 1: Create data directories**

```bash
mkdir -p data/history
```

- [ ] **Step 2: Create config.json**

```json
{
  "stocks": ["002602", "600519", "000001"],
  "indicators": {
    "ma": { "enabled": true, "weight": 15 },
    "macd": { "enabled": true, "weight": 20 },
    "rsi": { "enabled": true, "weight": 15 },
    "bollinger": { "enabled": true, "weight": 15 },
    "volume": { "enabled": true, "weight": 15 },
    "kdj": { "enabled": true, "weight": 10 }
  },
  "market_modifier": {
    "enabled": true,
    "max_impact": 15,
    "indices": ["sh000001", "sz399001", "sz399006", "sh000905"]
  }
}
```

- [ ] **Step 3: Replace forecast.py with minimal CLI skeleton**

```python
import argparse
import json
import os
import sys

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="A-Share Stock Quantitative Analyzer")
    parser.add_argument("symbols", nargs="*", help="Stock codes to analyze (e.g. 002602)")
    parser.add_argument("--review", action="store_true", help="Show self-reflection report only")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cached data")
    args = parser.parse_args()

    config = load_config()

    if args.review:
        print("Self-reflection report (not yet implemented)")
        return

    symbols = args.symbols or config.get("stocks", [])
    if not symbols:
        print("No stocks specified. Use positional args or configure stocks in config.json")
        sys.exit(1)

    for symbol in symbols:
        print(f"Analyzing {symbol}... (not yet implemented)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify CLI works**

```bash
python forecast.py --help
python forecast.py
python forecast.py 002602
python forecast.py --review
```

Expected: help text, then "Analyzing 002602... (not yet implemented)"

- [ ] **Step 5: Commit**

```bash
git add config.json forecast.py
git commit -m "feat: project scaffolding with config.json and CLI skeleton"
```

---

### Task 2: Data Source Module (data_source.py)

**Files:**
- Create: `data_source.py`

This module handles all data fetching. It has no tests because it depends on external APIs. Instead, test by running it manually.

- [ ] **Step 1: Create data_source.py with cache management**

The module needs these public functions:

```python
def get_stock_history(symbol: str, refresh: bool = False) -> pd.DataFrame
    """Get historical daily OHLCV data. Uses local CSV cache with incremental updates."""

def get_realtime_quote(symbol: str) -> dict | None
    """Get real-time quote during trading hours. Returns dict with date/open/high/low/close/volume or None."""

def get_index_history(index_code: str, refresh: bool = False) -> pd.DataFrame
    """Get index historical data (same mechanism as stock history)."""
```

Implementation details:

**Stock code to exchange prefix:**
```python
def _exchange_prefix(code: str) -> str:
    return "sh" if code.startswith(("6", "9", "5")) else "sz"
```

**CSV cache path:**
```python
def _cache_path(code: str) -> str:
    prefix = _exchange_prefix(code)
    return os.path.join(os.path.dirname(__file__), "data", "history", f"{prefix}{code}.csv")
```

**Incremental update logic:**
```python
def get_stock_history(symbol: str, refresh: bool = False) -> pd.DataFrame:
    cache_file = _cache_path(symbol)

    if os.path.exists(cache_file) and not refresh:
        cached = pd.read_csv(cache_file, parse_dates=["date"])
        latest_date = cached["date"].max()
        # If cache is up to date (latest is today or last trading day), return cached
        today = pd.Timestamp.now().normalize()
        if latest_date >= today - pd.Timedelta(days=3):
            return cached
        # Fetch only new data after latest_date
        new_data = _fetch_history_remote(symbol, start_date=latest_date)
        if new_data is not None and len(new_data) > 0:
            merged = pd.concat([cached, new_data], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            merged.to_csv(cache_file, index=False)
            return merged
        return cached
    else:
        # First fetch: 3 years
        data = _fetch_history_remote(symbol)
        if data is not None:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            data.to_csv(cache_file, index=False)
            return data
        raise RuntimeError(f"Failed to fetch history data for {symbol}")
```

**Multi-source fetch with fallback:**
```python
def _fetch_history_remote(symbol: str, start_date=None) -> pd.DataFrame | None:
    """Try akshare first, then Sina API. Returns DataFrame or None."""
    df = _fetch_akshare(symbol, start_date)
    if df is not None and len(df) > 0:
        return df
    df = _fetch_sina_history(symbol, start_date)
    if df is not None and len(df) > 0:
        return df
    return None
```

**akshare fetch:**
```python
def _fetch_akshare(symbol: str, start_date=None) -> pd.DataFrame | None:
    try:
        import akshare as ak
        full_code = f"{_exchange_prefix(symbol)}{symbol}"
        df = ak.stock_zh_a_daily(symbol=full_code, adjust="qfq")
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        if start_date:
            df = df[df["date"] > start_date]
        return df
    except Exception:
        return None
```

**Sina history fetch (fallback):**
```python
_HEADERS = {
    "Referer": "http://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def _fetch_sina_history(symbol: str, start_date=None) -> pd.DataFrame | None:
    try:
        prefix = _exchange_prefix(symbol)
        url = (
            f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            f"CN_MarketData.getKLineData?symbol={prefix}{symbol}&scale=240&ma=no&datalen=1000"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["day"])
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["close"])
        if start_date:
            df = df[df["date"] > start_date]
        return df
    except Exception:
        return None
```

**Real-time quote (Sina only — akshare doesn't have real-time):**
```python
def get_realtime_quote(symbol: str) -> dict | None:
    try:
        prefix = _exchange_prefix(symbol)
        url = f"http://hq.sinajs.cn/list={prefix}{symbol}"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        content = resp.text
        data = content.split('"')[1].split(",")
        if len(data) < 32:
            return None
        return {
            "date": data[30],
            "open": float(data[1]),
            "high": float(data[4]),
            "low": float(data[5]),
            "close": float(data[3]),
            "volume": float(data[8]),
        }
    except Exception:
        return None
```

**Index history** uses the same mechanism — `get_index_history` calls `get_stock_history` since index codes (e.g. `sh000001`) already include the exchange prefix.

- [ ] **Step 2: Test data_source.py manually**

```bash
python -c "from data_source import get_stock_history; df = get_stock_history('002602'); print(df.tail()); print(f'Rows: {len(df)}')"
```

Expected: DataFrame with ~750 rows, columns: date, open, high, low, close, volume. Check `data/history/sz002602.csv` was created.

```bash
python -c "from data_source import get_stock_history; df = get_stock_history('002602'); print(f'Rows: {len(df)}')"
```

Expected: Same row count (cache hit, no re-fetch).

- [ ] **Step 3: Commit**

```bash
git add data_source.py
git commit -m "feat: data source module with akshare/Sina fallback and CSV caching"
```

---

### Task 3: Technical Indicators Module (indicators.py)

**Files:**
- Create: `indicators.py`

Pure calculation module. All functions take a DataFrame and return a new DataFrame with added columns. No side effects. No mutation of input — return a copy.

- [ ] **Step 1: Write indicator calculation functions**

```python
import pandas as pd
import numpy as np


def calc_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5, MA10, MA20, MA60 columns."""
    result = df.copy()
    for period in (5, 10, 20, 60):
        result[f"ma{period}"] = result["close"].rolling(period).mean()
    return result


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add DIF, DEA, MACD histogram columns."""
    result = df.copy()
    ema_fast = result["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow, adjust=False).mean()
    result["dif"] = ema_fast - ema_slow
    result["dea"] = result["dif"].ewm(span=signal, adjust=False).mean()
    result["macd_hist"] = 2 * (result["dif"] - result["dea"])
    return result


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column."""
    result = df.copy()
    delta = result["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    result["rsi"] = 100 - (100 / (1 + rs))
    return result


def calc_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add bollinger middle, upper, lower, bandwidth columns."""
    result = df.copy()
    result["boll_mid"] = result["close"].rolling(period).mean()
    rolling_std = result["close"].rolling(period).std()
    result["boll_upper"] = result["boll_mid"] + num_std * rolling_std
    result["boll_lower"] = result["boll_mid"] - num_std * rolling_std
    result["boll_width"] = (result["boll_upper"] - result["boll_lower"]) / result["boll_mid"]
    return result


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """Add K, D, J columns."""
    result = df.copy()
    low_min = result["low"].rolling(n).min()
    high_max = result["high"].rolling(n).max()
    rsv = (result["close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    result["k"] = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    result["d"] = result["k"].ewm(alpha=1 / m2, adjust=False).mean()
    result["j"] = 3 * result["k"] - 2 * result["d"]
    return result


def calc_volume_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume ratio (current vol vs 5-day avg) and volume change columns."""
    result = df.copy()
    result["vol_5d_avg"] = result["volume"].rolling(5).mean()
    result["vol_ratio"] = result["volume"] / result["vol_5d_avg"].replace(0, np.nan)
    return result


def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators and return new DataFrame."""
    result = df.copy()
    result = calc_ma(result)
    result = calc_macd(result)
    result = calc_rsi(result)
    result = calc_bollinger(result)
    result = calc_kdj(result)
    result = calc_volume_analysis(result)
    return result
```

- [ ] **Step 2: Verify indicators calculate correctly**

```bash
python -c "
from data_source import get_stock_history
from indicators import calc_all_indicators
df = get_stock_history('002602')
df = calc_all_indicators(df)
print(df[['date','close','ma5','ma20','dif','dea','rsi','boll_upper','boll_lower','k','d','j','vol_ratio']].tail(10))
"
```

Expected: Last 10 rows with all indicator columns populated, no NaN in recent rows.

- [ ] **Step 3: Commit**

```bash
git add indicators.py
git commit -m "feat: technical indicators module (MA, MACD, RSI, Bollinger, KDJ, Volume)"
```

---

### Task 4: Analyzer Module — Scoring Engine (analyzer.py)

**Files:**
- Create: `analyzer.py`

This is the core analysis engine. It takes indicator data and produces a score, signal, and detailed analysis items.

- [ ] **Step 1: Write the scoring functions**

```python
import pandas as pd
import json
import os


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_ma(df: pd.DataFrame) -> tuple[int, str]:
    """Score moving average alignment. Returns (score, reason)."""
    last = df.iloc[-1]
    score = 0
    reasons = []

    # MA alignment check
    if last["ma5"] > last["ma10"] > last["ma20"]:
        score += 10
        reasons.append(f"MA5>MA10>MA20 bullish alignment")
    elif last["ma5"] < last["ma10"] < last["ma20"]:
        score -= 10
        reasons.append(f"MA5<MA10<MA20 bearish alignment")

    # Price vs MA20
    if last["close"] > last["ma20"]:
        score += 5
        reasons.append(f"Price above MA20 ({last['ma20']:.2f})")
    else:
        score -= 5
        reasons.append(f"Price below MA20 ({last['ma20']:.2f})")

    return score, "; ".join(reasons) if reasons else "No clear MA signal"


def score_macd(df: pd.DataFrame) -> tuple[int, str]:
    """Score MACD signal. Returns (score, reason)."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    # Golden cross / death cross
    if prev["dif"] <= prev["dea"] and last["dif"] > last["dea"]:
        score += 15
        reasons.append("MACD golden cross (DIF crosses above DEA)")
    elif prev["dif"] >= prev["dea"] and last["dif"] < last["dea"]:
        score -= 15
        reasons.append("MACD death cross (DIF crosses below DEA)")

    # MACD histogram direction
    if last["macd_hist"] > 0 and prev["macd_hist"] <= 0:
        score += 8
        reasons.append("MACD histogram turns positive")
    elif last["macd_hist"] < 0 and prev["macd_hist"] >= 0:
        score -= 8
        reasons.append("MACD histogram turns negative")
    elif last["macd_hist"] > prev["macd_hist"] > 0:
        score += 5
        reasons.append("MACD momentum increasing")
    elif last["macd_hist"] < prev["macd_hist"] < 0:
        score -= 5
        reasons.append("MACD momentum decreasing (bearish)")

    return score, "; ".join(reasons) if reasons else "MACD neutral"


def score_rsi(df: pd.DataFrame) -> tuple[int, str]:
    """Score RSI. Returns (score, reason)."""
    last = df.iloc[-1]
    rsi = last["rsi"]
    score = 0
    reasons = []

    if rsi < 30:
        score += 12
        reasons.append(f"RSI={rsi:.1f} oversold zone (rebound expected)")
    elif rsi < 40:
        score += 5
        reasons.append(f"RSI={rsi:.1f} approaching oversold")
    elif rsi > 70:
        score -= 12
        reasons.append(f"RSI={rsi:.1f} overbought zone (pullback risk)")
    elif rsi > 60:
        score -= 3
        reasons.append(f"RSI={rsi:.1f} approaching overbought")
    else:
        reasons.append(f"RSI={rsi:.1f} neutral zone")

    return score, "; ".join(reasons)


def score_bollinger(df: pd.DataFrame) -> tuple[int, str]:
    """Score Bollinger Bands. Returns (score, reason)."""
    last = df.iloc[-1]
    score = 0
    reasons = []

    # Price position within bands
    band_range = last["boll_upper"] - last["boll_lower"]
    if band_range > 0:
        position = (last["close"] - last["boll_lower"]) / band_range
        if position < 0.1:
            score += 12
            reasons.append(f"Price near lower band ({last['boll_lower']:.2f}), support zone")
        elif position < 0.3:
            score += 5
            reasons.append("Price in lower half of Bollinger range")
        elif position > 0.9:
            score -= 10
            reasons.append(f"Price near upper band ({last['boll_upper']:.2f}), resistance zone")
        elif position > 0.7:
            score -= 3
            reasons.append("Price in upper half of Bollinger range")
        else:
            reasons.append("Price in middle of Bollinger range")

    return score, "; ".join(reasons)


def score_kdj(df: pd.DataFrame) -> tuple[int, str]:
    """Score KDJ. Returns (score, reason)."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    j = last["j"]
    prev_j = prev["j"]

    # J value zones
    if j < 20 and prev_j <= 20:
        score += 10
        reasons.append(f"J={j:.1f} in oversold zone")
    elif j > 80 and prev_j >= 80:
        score -= 10
        reasons.append(f"J={j:.1f} in overbought zone")
    elif prev["k"] < prev["d"] and last["k"] > last["d"]:
        score += 12
        reasons.append("KDJ golden cross")
    elif prev["k"] > prev["d"] and last["k"] < last["d"]:
        score -= 12
        reasons.append("KDJ death cross")
    else:
        reasons.append(f"J={j:.1f} neutral")

    return score, "; ".join(reasons)


def score_volume(df: pd.DataFrame) -> tuple[int, str]:
    """Score volume analysis. Returns (score, reason)."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    vol_ratio = last["vol_ratio"]

    # Volume + price direction
    price_up = last["close"] > prev["close"]
    if vol_ratio > 1.5 and price_up:
        score += 12
        reasons.append(f"Volume {vol_ratio:.1f}x avg, price up — strong buying")
    elif vol_ratio > 1.5 and not price_up:
        score -= 10
        reasons.append(f"Volume {vol_ratio:.1f}x avg, price down — distribution")
    elif vol_ratio < 0.5 and price_up:
        score -= 5
        reasons.append(f"Volume shrinking on price rise — divergence warning")
    elif vol_ratio < 0.5 and not price_up:
        score += 5
        reasons.append(f"Volume shrinking on price drop — selling exhaustion")
    elif vol_ratio > 1.0 and price_up:
        score += 5
        reasons.append(f"Volume above average, price up")
    else:
        reasons.append(f"Volume {vol_ratio:.1f}x average, no clear signal")

    return score, "; ".join(reasons)


SIGNAL_RATINGS = [
    (75, 90, "Strong Buy"),
    (60, 74, "Buy"),
    (45, 59, "Hold"),
    (30, 44, "Sell"),
    (10, 29, "Strong Sell"),
]


def score_to_signal(score: int) -> str:
    for low, high, label in SIGNAL_RATINGS:
        if low <= score <= high:
            return label
    return "Hold"


def calculate_stock_score(df: pd.DataFrame, config: dict) -> tuple[int, list[dict]]:
    """
    Calculate weighted score for a stock.
    Returns (raw_score, list_of_indicator_results).
    Each indicator result: {"name": str, "score": int, "reason": str}
    """
    indicator_config = config.get("indicators", {})
    score_funcs = {
        "ma": score_ma,
        "macd": score_macd,
        "rsi": score_rsi,
        "bollinger": score_bollinger,
        "kdj": score_kdj,
        "volume": score_volume,
    }

    total_weight = 0
    weighted_sum = 0
    results = []

    for name, func in score_funcs.items():
        cfg = indicator_config.get(name, {})
        if not cfg.get("enabled", True):
            continue
        weight = cfg.get("weight", 10)
        ind_score, reason = func(df)
        # Clamp individual score to [-20, 20]
        ind_score = max(-20, min(20, ind_score))
        weighted_sum += ind_score * weight
        total_weight += weight
        results.append({"name": name.upper(), "score": ind_score, "reason": reason})

    if total_weight == 0:
        raw_score = 50
    else:
        # Normalize: weighted average mapped to [-20, 20] range, added to base 50
        avg_contribution = weighted_sum / total_weight
        raw_score = 50 + avg_contribution

    return raw_score, results
```

- [ ] **Step 2: Verify scoring works**

```bash
python -c "
from data_source import get_stock_history
from indicators import calc_all_indicators
from analyzer import calculate_stock_score, score_to_signal, load_config
df = get_stock_history('002602')
df = calc_all_indicators(df)
config = load_config()
score, results = calculate_stock_score(df, config)
print(f'Raw score: {score:.1f}')
print(f'Signal: {score_to_signal(int(score))}')
for r in results:
    print(f'  {r[\"name\"]}: {r[\"score\"]:+d} — {r[\"reason\"]}')
"
```

Expected: Score between 10-90, signal label, and 6 indicator results with scores and reasons.

- [ ] **Step 3: Commit**

```bash
git add analyzer.py
git commit -m "feat: scoring engine with weighted indicator evaluation"
```

---

### Task 5: Analyzer Module — Market Modifier + Report Formatting

**Files:**
- Modify: `analyzer.py` (add market modifier, key levels, position advice, report formatter)

- [ ] **Step 1: Add market modifier function to analyzer.py**

```python
from data_source import get_index_history
from indicators import calc_ma, calc_macd

# Map of index codes to display names
INDEX_NAMES = {
    "sh000001": "Shanghai Composite",
    "sz399001": "Shenzhen Component",
    "sz399006": "ChiNext",
    "sh000905": "CSI 500",
}


def classify_index_trend(df: pd.DataFrame) -> str:
    """Classify an index as 'bullish', 'bearish', or 'neutral'."""
    last = df.iloc[-1]
    if pd.isna(last.get("ma20")) or pd.isna(last.get("dif")):
        return "neutral"
    price_above_ma = last["close"] > last["ma20"]
    macd_bullish = last["dif"] > last["dea"]
    if price_above_ma and macd_bullish:
        return "bullish"
    elif not price_above_ma and not macd_bullish:
        return "bearish"
    return "neutral"


def calculate_market_modifier(config: dict) -> tuple[int, list[dict]]:
    """
    Calculate broad market modifier.
    Returns (modifier_score, list_of_index_results).
    """
    market_cfg = config.get("market_modifier", {})
    if not market_cfg.get("enabled", True):
        return 0, []

    max_impact = market_cfg.get("max_impact", 15)
    index_codes = market_cfg.get("indices", [])
    results = []
    bullish_count = 0
    bearish_count = 0

    for code in index_codes:
        try:
            df = get_index_history(code)
            df = calc_ma(df)
            df = calc_macd(df)
            trend = classify_index_trend(df)
            name = INDEX_NAMES.get(code, code)
            results.append({"code": code, "name": name, "trend": trend})
            if trend == "bullish":
                bullish_count += 1
            elif trend == "bearish":
                bearish_count += 1
        except Exception:
            results.append({"code": code, "name": INDEX_NAMES.get(code, code), "trend": "error"})

    total = len(index_codes)

    # Calculate modifier
    if bullish_count >= 3:
        modifier = int(max_impact * (0.67 + 0.33 * bullish_count / total))
    elif bullish_count >= 2:
        modifier = int(max_impact * 0.35)
    elif bearish_count >= 3:
        modifier = -int(max_impact * (0.67 + 0.33 * bearish_count / total))
    elif bearish_count >= 2:
        modifier = -int(max_impact * 0.4)
    else:
        modifier = 0

    modifier = max(-max_impact, min(max_impact, modifier))
    return modifier, results
```

- [ ] **Step 2: Add key levels, position advice, and risk alerts to analyzer.py**

```python
def calculate_key_levels(df: pd.DataFrame) -> dict:
    """Calculate support and resistance levels."""
    last = df.iloc[-1]
    levels = {
        "support": [],
        "resistance": [],
    }
    if not pd.isna(last.get("ma20")):
        levels["support"].append(("MA20", last["ma20"]))
    if not pd.isna(last.get("boll_lower")):
        levels["support"].append(("Boll Lower", last["boll_lower"]))
    if not pd.isna(last.get("boll_upper")):
        levels["resistance"].append(("Boll Upper", last["boll_upper"]))
    # Recent high as resistance
    recent_high = df["high"].tail(20).max()
    levels["resistance"].append(("20D High", recent_high))
    return levels


def calculate_position_advice(score: int, key_levels: dict) -> str:
    """Generate position advice based on score and proximity to key levels."""
    if score >= 75:
        base = "60%"
        note = "strong buy signal"
    elif score >= 60:
        base = "40%"
        note = "buy signal"
    elif score >= 45:
        base = "20%"
        note = "neutral signal"
    elif score >= 30:
        base = "10%"
        note = "sell signal"
    else:
        base = "0%"
        note = "strong sell signal"

    return f"Suggested: {base} ({note})"


def generate_risk_alerts(df: pd.DataFrame, score: int) -> list[str]:
    """Generate risk alert messages."""
    alerts = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if not pd.isna(last.get("rsi")):
        if last["rsi"] > 65:
            alerts.append(f"RSI={last['rsi']:.1f} approaching overbought, watch for pullback")
        if last["rsi"] < 35:
            alerts.append(f"RSI={last['rsi']:.1f} approaching oversold, may rebound")

    if not pd.isna(last.get("boll_upper")) and not pd.isna(last.get("boll_lower")):
        band_range = last["boll_upper"] - last["boll_lower"]
        if band_range > 0:
            position = (last["close"] - last["boll_lower"]) / band_range
            if position > 0.85:
                alerts.append("Close to Bollinger upper band, limited upside room")
            elif position < 0.15:
                alerts.append("Near Bollinger lower band, watch for support breakdown")

    change = (last["close"] - prev["close"]) / prev["close"]
    if abs(change) > 0.05:
        alerts.append(f"Large daily swing ({abs(change)*100:.1f}%), expect volatility")

    if score >= 70:
        alerts.append("High bullish score — consider taking partial profits on existing positions")
    elif score <= 30:
        alerts.append("Low score — avoid adding positions, wait for reversal signal")

    return alerts
```

- [ ] **Step 3: Add report formatter to analyzer.py**

```python
from datetime import datetime


def format_report(
    symbol: str,
    df: pd.DataFrame,
    score: int,
    indicator_results: list[dict],
    market_modifier: int,
    market_results: list[dict],
    key_levels: dict,
    risk_alerts: list[str],
    position_advice: str,
    is_intraday: bool = False,
    realtime_data: dict | None = None,
) -> str:
    """Format the full analysis report as a string."""
    signal = score_to_signal(score)
    lines = []

    # Header
    now = datetime.now()
    if is_intraday:
        date_str = now.strftime("%Y-%m-%d %H:%M")
    else:
        date_str = now.strftime("%Y-%m-%d")

    lines.append("=" * 50)
    lines.append(f"  Stock Analysis Report | {symbol} | {date_str}")
    lines.append("=" * 50)

    # Price info
    last = df.iloc[-1]
    if is_intraday and realtime_data:
        lines.append(f"Current Price: {realtime_data['close']:.2f}  Today's Change: {((realtime_data['close']/realtime_data['open'])-1)*100:+.2f}%")
        lines.append("[Trading Session] Intraday real-time analysis")
    else:
        prev = df.iloc[-2]
        change = (last["close"] - prev["close"]) / prev["close"] * 100
        lines.append(f"Current Price: {last['close']:.2f}  Change: {change:+.2f}%")

    lines.append("")
    lines.append(f"[Signal Rating] {signal}")
    prob = max(10, min(90, score))
    if is_intraday:
        direction = "close higher" if score >= 50 else "close lower"
        lines.append(f"[Today Close Prediction] Leaning towards {direction} ({prob}%)")
    else:
        lines.append(f"[Next-Day Up Probability] {prob}%")

    # Intraday real-time status
    if is_intraday and realtime_data:
        lines.append("")
        lines.append("--- Real-Time Status ---")
        lines.append(f"Current: {realtime_data['close']:.2f} | Open: {realtime_data['open']:.2f} | High: {realtime_data['high']:.2f} | Low: {realtime_data['low']:.2f}")
        daily_range = realtime_data["high"] - realtime_data["low"]
        if daily_range > 0:
            pct = (realtime_data["close"] - realtime_data["low"]) / daily_range * 100
            region = "lower" if pct < 30 else "upper" if pct > 70 else "middle"
            lines.append(f"Intraday position: {region} region ({pct:.0f}th percentile of daily range)")

    # Broad market
    if market_results:
        lines.append("")
        lines.append("--- Broad Market Environment ---")
        for idx in market_results:
            icon = "+" if idx["trend"] == "bullish" else "-" if idx["trend"] == "bearish" else "~"
            label = idx["trend"]
            lines.append(f"  [{icon}] {idx['name']}: {label}")
        bullish_n = sum(1 for i in market_results if i["trend"] == "bullish")
        total_n = len(market_results)
        lines.append(f"Market modifier: {market_modifier:+d} ({bullish_n}/{total_n} indices bullish)")

    # Technical indicators
    lines.append("")
    lines.append("--- Technical Indicators ---")
    for r in indicator_results:
        icon = "+" if r["score"] > 0 else "-" if r["score"] < 0 else "~"
        lines.append(f"  [{icon}] {r['name']}: {r['reason']} ({r['score']:+d})")

    if is_intraday:
        lines.append("  [!] Note: Latest candle is intraday (incomplete), indicator values are approximate")

    # Risk alerts
    if risk_alerts:
        lines.append("")
        lines.append("--- Risk Alerts ---")
        for alert in risk_alerts:
            lines.append(f"  [!] {alert}")

    # Key levels
    lines.append("")
    lines.append("--- Key Levels ---")
    support_str = " / ".join(f"{v:.2f} ({n})" for n, v in key_levels["support"])
    resist_str = " / ".join(f"{v:.2f} ({n})" for n, v in key_levels["resistance"])
    lines.append(f"Support: {support_str}")
    lines.append(f"Resistance: {resist_str}")

    # Position advice
    lines.append("")
    lines.append("--- Position Advice ---")
    lines.append(position_advice)

    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Test the full report**

```bash
python -c "
from data_source import get_stock_history
from indicators import calc_all_indicators
from analyzer import *
df = get_stock_history('002602')
df = calc_all_indicators(df)
config = load_config()
score, ind_results = calculate_stock_score(df, config)
modifier, market_results = calculate_market_modifier(config)
final = max(10, min(90, int(score + modifier)))
key_levels = calculate_key_levels(df)
risk_alerts = generate_risk_alerts(df, final)
position = calculate_position_advice(final, key_levels)
report = format_report('002602', df, final, ind_results, modifier, market_results, key_levels, risk_alerts, position)
print(report)
"
```

Expected: Full formatted report with all sections.

- [ ] **Step 5: Commit**

```bash
git add analyzer.py
git commit -m "feat: market modifier, key levels, risk alerts, and report formatter"
```

---

### Task 6: Tracker Module (tracker.py)

**Files:**
- Create: `tracker.py`

Handles prediction recording, backfilling, and accuracy statistics.

- [ ] **Step 1: Write tracker module**

```python
import csv
import os
from datetime import datetime


_PREDICTIONS_FILE = os.path.join(os.path.dirname(__file__), "data", "predictions.csv")
_FIELDS = ["date", "symbol", "price", "signal", "score", "pred_up_prob", "actual_change", "hit"]


def _ensure_file():
    os.makedirs(os.path.dirname(_PREDICTIONS_FILE), exist_ok=True)
    if not os.path.exists(_PREDICTIONS_FILE):
        with open(_PREDICTIONS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()


def record_prediction(symbol: str, price: float, signal: str, score: int, prob: int) -> None:
    """Record a prediction to the CSV file."""
    _ensure_file()
    today = datetime.now().strftime("%Y-%m-%d")
    with open(_PREDICTIONS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writerow({
            "date": today,
            "symbol": symbol,
            "price": f"{price:.2f}",
            "signal": signal,
            "score": str(score),
            "pred_up_prob": f"{prob}%",
            "actual_change": "",
            "hit": "",
        })


def read_predictions() -> list[dict]:
    """Read all predictions from CSV."""
    _ensure_file()
    with open(_PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def backfill_predictions(fetch_actual_fn) -> int:
    """
    Backfill actual results for unfilled predictions.
    fetch_actual_fn(symbol, date) -> float or None: returns next-day close price.
    Returns number of records backfilled.
    """
    predictions = read_predictions()
    backfilled = 0
    updated_rows = []

    for row in predictions:
        if row["actual_change"] and row["actual_change"].strip():
            updated_rows.append(row)
            continue

        symbol = row["symbol"]
        pred_date = row["date"]
        pred_price = float(row["price"])
        signal = row["signal"]

        actual_close = fetch_actual_fn(symbol, pred_date)
        if actual_close is None:
            updated_rows.append(row)
            continue

        actual_change = (actual_close - pred_price) / pred_price
        # Determine hit: buy signals should see price go up, sell signals should see price go down
        is_buy_signal = signal in ("Strong Buy", "Buy")
        hit = (is_buy_signal and actual_change > 0) or (not is_buy_signal and actual_change < 0)

        row["actual_change"] = f"{actual_change:+.4f}"
        row["hit"] = "1" if hit else "0"
        updated_rows.append(row)
        backfilled += 1

    if backfilled > 0:
        with open(_PREDICTIONS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(updated_rows)

    return backfilled


def calculate_accuracy() -> dict:
    """Calculate accuracy statistics from predictions."""
    predictions = read_predictions()
    verified = [p for p in predictions if p["hit"] and p["hit"].strip()]

    if not verified:
        return {"total": len(predictions), "verified": 0, "overall": None, "by_signal": {}}

    total_hits = sum(1 for p in verified if p["hit"] == "1")
    overall = total_hits / len(verified) * 100

    by_signal = {}
    for signal_name in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"):
        signal_preds = [p for p in verified if p["signal"] == signal_name]
        if signal_preds:
            hits = sum(1 for p in signal_preds if p["hit"] == "1")
            by_signal[signal_name] = {
                "total": len(signal_preds),
                "hits": hits,
                "accuracy": hits / len(signal_preds) * 100,
            }

    return {
        "total": len(predictions),
        "verified": len(verified),
        "overall": overall,
        "by_signal": by_signal,
    }


def format_accuracy_report(stats: dict) -> str:
    """Format accuracy statistics as a readable string."""
    lines = ["--- Prediction Self-Reflection ---"]
    lines.append(f"Total predictions: {stats['total']}")
    lines.append(f"Verified samples: {stats['verified']}")

    if stats["overall"] is not None:
        hits = sum(s["hits"] for s in stats["by_signal"].values())
        lines.append(f"Overall accuracy: {hits}/{stats['verified']} ({stats['overall']:.1f}%)")
        lines.append("")
        for signal_name, s in stats["by_signal"].items():
            lines.append(f"  {signal_name} accuracy: {s['hits']}/{s['total']} ({s['accuracy']:.1f}%)")
    else:
        lines.append("No verified predictions yet.")

    return "\n".join(lines)
```

- [ ] **Step 2: Test tracker manually**

```bash
python -c "
from tracker import record_prediction, calculate_accuracy, format_accuracy_report
record_prediction('002602', 12.35, 'Buy', 68, 68)
stats = calculate_accuracy()
print(format_accuracy_report(stats))
"
```

Expected: "Total predictions: 1, Verified samples: 0, No verified predictions yet."

- [ ] **Step 3: Commit**

```bash
git add tracker.py
git commit -m "feat: tracker module for prediction logging and accuracy statistics"
```

---

### Task 7: Main Orchestration (forecast.py)

**Files:**
- Modify: `forecast.py` (full implementation)

Wire everything together: CLI parsing, trading-hours detection, data fetching, analysis, reporting, tracking, and multi-stock summary.

- [ ] **Step 1: Implement full forecast.py**

```python
import argparse
import json
import os
import sys
from datetime import datetime

from data_source import get_stock_history, get_realtime_quote
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
    """Fetch actual closing price for a given date (used by tracker backfill)."""
    try:
        df = get_stock_history(symbol)
        target = df[df["date"]astype(str).str.contains(pred_date)]
        if len(target) == 0:
            return None
        idx = target.index[0]
        # Get the next trading day's close
        if idx + 1 < len(df):
            return float(df.iloc[idx + 1]["close"])
        return None
    except Exception:
        return None


def analyze_stock(symbol: str, config: dict, refresh: bool = False) -> str | None:
    """Analyze a single stock and return the formatted report."""
    try:
        df = get_stock_history(symbol, refresh=refresh)
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return None

    if df is None or len(df) < 30:
        print(f"  [ERROR] Insufficient data for {symbol} ({len(df) if df is not None else 0} rows)")
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
                df.loc[len(df)] = new_row

    df = calc_all_indicators(df)
    raw_score, ind_results = calculate_stock_score(df, config)
    modifier, market_results = calculate_market_modifier(config)
    final_score = max(10, min(90, int(raw_score + modifier)))
    signal = score_to_signal(final_score)
    key_levels = calculate_key_levels(df)
    risk_alerts = generate_risk_alerts(df, final_score)
    position_advice = calculate_position_advice(final_score, key_levels)

    report = format_report(
        symbol=symbol,
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

    # Record prediction (avoid duplicates for same day)
    record_prediction(symbol, df.iloc[-1]["close"], signal, final_score, final_score)

    return report, signal, final_score


def main():
    parser = argparse.ArgumentParser(description="A-Share Stock Quantitative Analyzer")
    parser.add_argument("symbols", nargs="*", help="Stock codes to analyze (e.g. 002602)")
    parser.add_argument("--review", action="store_true", help="Show self-reflection report only")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cached data")
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
        print("No stocks specified. Use positional args or configure stocks in config.json")
        sys.exit(1)

    multiple = len(symbols) > 1
    reports = []

    if multiple:
        print(f"Analyzing {len(symbols)} stocks...\n")

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
        buy_stocks = [r for r in reports if r["signal"] in ("Strong Buy", "Buy")]
        sell_stocks = [r for r in reports if r["signal"] in ("Strong Sell", "Sell")]
        neutral_stocks = [r for r in reports if r["signal"] == "Hold"]

        print("--- Daily Summary ---")
        if buy_stocks:
            items = ", ".join(f"{r['symbol']} ({r['signal']}, {r['score']}%)" for r in buy_stocks)
            print(f"Watch: {items}")
        if sell_stocks:
            items = ", ".join(f"{r['symbol']} ({r['signal']}, {r['score']}%)" for r in sell_stocks)
            print(f"Avoid: {items}")
        if neutral_stocks:
            items = ", ".join(f"{r['symbol']} ({r['score']}%)" for r in neutral_stocks)
            print(f"Neutral: {items}")

    # Show accuracy stats after all analyses
    stats = calculate_accuracy()
    if stats["verified"] > 0:
        print("")
        print(format_accuracy_report(stats))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test full pipeline**

```bash
python forecast.py 002602
```

Expected: Full analysis report with all sections.

```bash
python forecast.py --review
```

Expected: Self-reflection report.

- [ ] **Step 3: Commit**

```bash
git add forecast.py
git commit -m "feat: full CLI orchestration with multi-stock support and intraday mode"
```

---

### Task 8: Integration Test + Cleanup

**Files:**
- Delete: old prototype code (already replaced in Task 1)
- Verify: end-to-end pipeline works

- [ ] **Step 1: Run end-to-end test**

```bash
python forecast.py 002602 600519 000001
```

Expected: 3 stock reports sorted by score, followed by daily summary and accuracy stats.

- [ ] **Step 2: Test refresh mode**

```bash
python forecast.py --refresh 002602
```

Expected: Report after forced data refresh.

- [ ] **Step 3: Test edge case — invalid stock code**

```bash
python forecast.py 999999
```

Expected: Clear error message, not a crash.

- [ ] **Step 4: Verify data directory structure**

```bash
ls -la data/
ls -la data/history/
```

Expected: `predictions.csv` and at least one stock history CSV.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: integration test and cleanup"
```
