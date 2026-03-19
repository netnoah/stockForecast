# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock data fetching tool using the [akshare](https://github.com/akfamily/akshare) library to retrieve A-share daily OHLCV data from Sina Finance API.

## Commands

```bash
# Run the main script
python test.py

# Install dependencies
pip install akshare
```

## Tech Stack

- **Python 3.x** — no build system, no virtual environment configured
- **akshare** (v1.18.40) — Chinese financial data API wrapper
- Data source: `ak.stock_zh_a_daily()` via Sina Finance interface

## Notes

- The project is a single-file prototype (`test.py`). No module structure, tests, or CI configured.
- akshare uses Sina Finance by default for `stock_zh_a_daily` — the comment notes Sina has more lenient firewall policies than East Money (东财).
- Stock symbol format: exchange prefix + code (e.g., `sz002602` for Shenzhen, `sh600000` for Shanghai).
