# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A股量化分析工具 — predicts stock trends using technical indicators (MA, MACD, RSI, KDJ, Bollinger, Volume) with composite scoring and market modifier. Supports A-shares, HK stocks, and market indices. Outputs Chinese-language reports to console and optionally pushes to WeChat Work (企业微信) via webhook.

## Commands

```bash
# Analyze stocks
python forecast.py 002602              # single stock
python forecast.py 002602 600519        # multiple stocks
python forecast.py -l                  # use config.json stock_list

# Review prediction accuracy
python forecast.py --review

# Force refresh cached data
python forecast.py 002602 --refresh

# Install dependencies
pip install -r requirements.txt
```

## Architecture

```
forecast.py          # CLI entry point, orchestration
├── data_source.py   # Data fetching: akshare → Sina API fallback, CSV cache in data/history/
├── indicators.py    # Technical indicator calculations (MA, MACD, RSI, KDJ, Bollinger, Volume)
├── analyzer.py      # Scoring engine, signal generation, report formatting
├── tracker.py       # Prediction logging to CSV, backfilling actuals, accuracy stats
└── wecom.py         # WeChat Work webhook report pushing
```

**Data flow:** `data_source` → `indicators` → `analyzer` (scoring + market modifier) → `tracker` + `wecom`

## Key Concepts

- **Scoring:** Each indicator scores [-20, +20], weighted by config → composite `raw_score` [-100, +100]
- **Market modifier:** Broad index trends (上证, 深证, 创业板, 中证500) adjust score by [-15, +15]
- **Signals:** ≥50 强烈买入, 15-49 买入, -14~14 观望, -49~-15 卖出, ≤-50 强烈卖出
- **Trading hours mode:** Weekdays 9:30-15:00 triggers intraday real-time data merge
- **Cache:** Historical data cached as CSV in `data/history/{code}.csv`, refreshed when stale (>3 days)
- **Stock format:** exchange prefix + code (`sz002602`, `sh600000`, `hk2400`, `sh000001` for indices)
- **A-share color convention:** Red = bullish (涨), Green = bearish (跌) — opposite of Western markets

## Configuration

All config in `config.json`: stock list, WeChat webhook URL, indicator weights, market modifier settings.

## Notes

- No tests, no CI, no virtual environment configured
- Report text is in Chinese with ANSI color codes
- WeChat webhook messages have a 4096-byte limit with auto-truncation
- HK stocks use `ak.stock_hk_daily()` instead of the A-share data path
