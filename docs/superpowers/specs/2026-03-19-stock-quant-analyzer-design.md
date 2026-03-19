# Stock Quantitative Analyzer Design

## Overview

A command-line tool for A-share stock analysis that predicts next-day up/down probability using technical indicators, provides a comprehensive analysis report, and tracks historical prediction accuracy through a self-reflection mechanism.

## Requirements

- Command-line interface, support multiple stocks per run
- Multi-source data fetching with automatic fallback (akshare → Sina)
- Local CSV cache with incremental updates (3 years initial fetch)
- Extended technical indicators: MA, MACD, RSI, Bollinger Bands, KDJ, volume analysis
- Weighted scoring system mapping to 5-tier signal rating
- Full analysis report: probability + reasons + signal + risk alerts + key levels + position advice
- Self-reflection: record predictions, backfill actual results, calculate accuracy rates

## Project Structure

```
stockForecast/
├── forecast.py          # Main entry: CLI parsing, workflow orchestration
├── data_source.py       # Data fetching + cache management
├── indicators.py        # Technical indicator calculations (numpy/pandas only)
├── analyzer.py          # Analysis engine: scoring, signal rating, report generation
├── tracker.py           # Self-reflection: prediction logging, backtesting, accuracy stats
├── config.json          # Stock list, indicator weights
├── data/
│   ├── predictions.csv  # Prediction records
│   └── history/         # Historical K-line cache (one CSV per stock)
│       ├── sz002602.csv
│       ├── sh600519.csv
│       └── ...
└── docs/
    └── superpowers/specs/
```

## Data Source Strategy

### Multi-source Fallback

1. Attempt akshare (`ak.stock_zh_a_daily()`)
2. If failed (rate limit, connection error), fallback to Sina Finance API
3. If both fail, report error and exit

### Local Cache + Incremental Update

- **First run**: Fetch 3 years of daily data (~750 trading days), save to `data/history/{exchange}{code}.csv`
- **Subsequent runs**: Read the latest date from local file, fetch only new data from that date to today, append and deduplicate
- **Trading day check**: If local latest date is today or the last trading day, skip network request entirely
- **Data cleaning**: Deduplicate by date, handle null values on each append

### History CSV Format

```csv
date,open,high,low,close,volume
2023-03-20,12.50,12.80,12.40,12.70,500000
2023-03-21,12.70,12.90,12.60,12.85,550000
```

## Technical Indicators

All indicators calculated using pandas + numpy. No external TA libraries.

| Indicator | Parameters | Scoring Logic |
|-----------|-----------|---------------|
| **Moving Averages** | MA5, MA10, MA20, MA60 | Bull alignment (MA5>MA10>MA20>MA60) = strong bullish. Price above MA20 = mid-term uptrend |
| **MACD** | EMA12, EMA26, Signal=9 | DIF crosses above DEA (golden cross) = bullish. Top divergence = bearish warning. Bottom divergence = bullish signal |
| **RSI** | Period=14 | RSI<30 oversold = rebound expected (bullish). RSI>70 overbought = pullback risk (bearish) |
| **Bollinger Bands** | MA20, 2σ | Price at lower band = support (bullish). Price at upper band = resistance (bearish). Bandwidth = volatility context |
| **KDJ** | K=9, D=3, J=3 | J value oversold golden cross = bullish. J value overbought death cross = bearish |
| **Volume Analysis** | 5-day avg volume | Volume increases with price rise = bullish. Volume-price divergence = warning. Volume spike on decline = bearish |

## Broad Market Index Factor

### Rationale

A-share stocks are highly correlated with broad market trends. Individual stock technical analysis is insufficient without market context — even strong individual stocks tend to decline in a bearish broad market.

### Index Coverage

| Index | Code | Represents |
|-------|------|------------|
| Shanghai Composite | sh000001 | Overall Shanghai market |
| Shenzhen Component | sz399001 | Overall Shenzhen market |
| ChiNext | sz399006 | Growth / tech stocks |
| CSI 500 | sh000905 | Mid-cap stocks |

### Data Strategy

Index data uses the same incremental cache mechanism as individual stocks. Fetched via `data_source.py` and cached in `data/history/`.

### Market Modifier Logic

The broad market acts as a **modifier** applied to the individual stock score, not as a weighted indicator:

```
final_score = stock_indicator_score + market_modifier
```

Market modifier scoring:

| Condition | Modifier | Description |
|-----------|----------|-------------|
| 3+ indices bullish (price > MA20, MACD bullish) | +10 to +15 | Broad market strong uptrend |
| 2 indices bullish, rest neutral | +3 to +8 | Mildly positive market |
| Indices mixed, no clear direction | -3 to +3 | Neutral market |
| 2+ indices bearish | -8 to -3 | Mildly negative market |
| 3+ indices bearish (price < MA20, MACD bearish) | -15 to -8 | Broad market strong downtrend |

The modifier is calculated using the same MA/MACD indicators applied to each index, producing a simple bullish/bearish/neutral classification per index, then aggregated.

### Report Section

A new "Broad Market" section appears in the report:

```
--- Broad Market Environment ---
✅ Shanghai Composite: above MA20, MACD bullish
✅ Shenzhen Component: above MA20, MACD bullish
❌ ChiNext: below MA20, MACD bearish
✅ CSI 500: above MA20, MACD bullish
Market modifier: +10 (3/4 indices bullish)
```

## Scoring System

### Mechanism

- Base score: 50 (neutral)
- Each indicator contributes a score in range [-20, +20]
- Weighted sum using weights from config.json
- Apply broad market modifier (range [-15, +15])
- Final score clamped to [10, 90] as up-probability percentage

### Signal Rating

| Score Range | Signal | Description |
|-------------|--------|-------------|
| 75-90 | Strong Buy | Multi-indicator bullish consensus |
| 60-74 | Buy | Leaning bullish with some disagreement |
| 45-59 | Hold | Balanced bullish/bearish |
| 30-44 | Sell | Leaning bearish |
| 10-29 | Strong Sell | Multi-indicator bearish consensus |

### Default Weights (config.json)

```json
{
  "stocks": ["002602", "600519", "000001"],
  "indicators": {
    "ma": { "enabled": true, "weight": 15 },
    "macd": { "enabled": true, "weight": 20 },
    "rsi": { "enabled": true, "weight": 15 },
    "bollinger": { "enabled": true, "weight": 15 },
    "volume": { "enabled": true, "weight": 15 },
    "kdj": { "enabled": true, "weight": 10 },
    "pattern": { "enabled": true, "weight": 10 }
  },
  "market_modifier": {
    "enabled": true,
    "max_impact": 15,
    "indices": ["sh000001", "sz399001", "sz399006", "sh000905"]
  }
}
```

## Analysis Report Format

```
========================================
  Stock Analysis Report | 002602 | 2026-03-19
========================================
Current Price: 12.35  Change: +1.23%

[Signal Rating] Buy
[Up Probability] 68%

--- Broad Market Environment ---
✅ Shanghai Composite: above MA20, MACD bullish
✅ Shenzhen Component: above MA20, MACD bullish
❌ ChiNext: below MA20, MACD bearish
✅ CSI 500: above MA20, MACD bullish
Market modifier: +10 (3/4 indices bullish)

--- Technical Indicators ---
✅ MA: MA5>MA10>MA20 bullish alignment (+15)
✅ MACD: DIF crosses above DEA, golden cross formed (+12)
⚠️ RSI: 65.3, approaching overbought zone (+5)
✅ Bollinger: Price running above middle band (+10)
✅ KDJ: J value 42, neutral-slightly bullish zone (+6)
✅ Volume: Volume 23% above 5-day average (+11)

--- Risk Alerts ---
⚠️ RSI approaching 70 overbought line, watch for short-term pullback
⚠️ Close to Bollinger upper band, limited upside room

--- Key Levels ---
Support: 11.80 (MA20) / 11.50 (Bollinger lower band)
Resistance: 12.80 (Bollinger upper band) / 13.10 (Previous high)

--- Position Advice ---
Suggested: 30% (bullish signal but near resistance, avoid heavy position)
```

## Self-Reflection Module

### Prediction Record (predictions.csv)

```csv
date,symbol,price,signal,score,pred_up_prob,actual_change,hit
2026-03-19,002602,12.35,Buy,68,68%,,
```

- `actual_change`: Backfilled on subsequent runs by fetching the next trading day's actual price change
- `hit`: Determined after backfill — buy signal + actual up = hit; sell signal + actual down = hit

### Backfill Logic

On each run, scan `predictions.csv` for rows where `actual_change` is empty:
1. For each unfilled row, fetch the actual closing price of the next trading day
2. Calculate `actual_change = (next_close - prediction_price) / prediction_price`
3. Set `hit = true` if the signal direction matches the actual direction

### Accuracy Report

Output after each analysis run:

```
--- Prediction Self-Reflection ---
Total predictions: 156
Verified samples: 142
Overall accuracy: 89/142 (62.7%)
Strong Buy accuracy: 23/28 (82.1%)
Buy accuracy: 45/72 (62.5%)
Hold accuracy: 15/25 (60.0%)
Sell accuracy: 6/17 (35.3%)
```

## Command-Line Interface

```
python forecast.py                  # Analyze all stocks in config.json
python forecast.py 002602           # Analyze a single stock
python forecast.py 002602 600519    # Analyze multiple stocks
python forecast.py --review         # Show self-reflection report only
python forecast.py --refresh 002602 # Force refresh data (ignore cache)
```

### Multi-Stock Output

When analyzing multiple stocks, sort by signal strength and show summary:

```
Analyzing 3 stocks...

========================================
  1/3 | 600519
========================================
[Signal Rating] Hold
[Up Probability] 52%
...

========================================
  2/3 | 002602
========================================
[Signal Rating] Buy
[Up Probability] 68%
...

========================================
  3/3 | 000001
========================================
[Signal Rating] Sell
[Up Probability] 35%
...

--- Daily Summary ---
Watch: 002602 (Buy, 68%)
Avoid: 000001 (Sell, 35%)
Neutral: 600519 (52%)
```

## Dependencies

- `requests` — HTTP client for Sina API
- `pandas` — Data manipulation and indicator calculation
- `numpy` — Numerical computation for indicators
- `akshare` (optional) — Primary data source, graceful fallback to Sina if unavailable

## Error Handling

- Network errors: Retry once, then fallback to next data source
- Invalid stock code: Validate format (6 digits), report clear error
- Insufficient data: If less than 60 trading days, warn that some indicators may be inaccurate
- File I/O errors: Handle missing data directory gracefully, create on first use
