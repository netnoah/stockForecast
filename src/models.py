"""Data models for the stock forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Encapsulates all outputs of a single stock analysis."""

    symbol: str
    stock_name: str
    score: int
    signal: str
    trend_status: str
    indicator_results: list[dict]
    market_modifier: int
    market_results: list[dict]
    key_levels: dict
    risk_alerts: list[str]
    position_advice: str
    is_intraday: bool
    realtime_data: dict | None
    session_label: str
    report: str = ""
