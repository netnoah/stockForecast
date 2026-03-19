"""
Technical indicators module for A-Share stock quantitative analyzer.

Provides pure calculation functions for common technical indicators.
All functions take a DataFrame and return a NEW DataFrame with added columns.
No side effects. No mutation of input — always return a copy.
"""

import numpy as np
import pandas as pd


def calc_ma(df: pd.DataFrame, periods: tuple[int, ...] = (5, 10, 20, 60)) -> pd.DataFrame:
    """Add Moving Average (MA) columns to the DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
        periods: Tuple of MA periods to calculate. Default: (5, 10, 20, 60).

    Returns:
        New DataFrame with added MA5, MA10, MA20, MA60 columns (named according to periods).
    """
    result = df.copy()
    close = pd.to_numeric(result["close"], errors="coerce")

    for period in periods:
        col_name = f"ma{period}"
        result[col_name] = close.rolling(window=period, min_periods=period).mean()

    return result


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence) columns to the DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
        fast: Fast EMA period. Default: 12.
        slow: Slow EMA period. Default: 26.
        signal: Signal line EMA period. Default: 9.

    Returns:
        New DataFrame with added DIF, DEA, MACD columns.
        - DIF: Fast EMA - Slow EMA
        - DEA: Signal line (EMA of DIF)
        - MACD: Histogram = 2 * (DIF - DEA)
    """
    result = df.copy()
    close = pd.to_numeric(result["close"], errors="coerce")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)

    result["dif"] = dif
    result["dea"] = dea
    result["macd"] = macd_hist

    return result


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI) column to the DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
        period: RSI calculation period. Default: 14.

    Returns:
        New DataFrame with added RSI column (0-100 scale).
    """
    result = df.copy()
    close = pd.to_numeric(result["close"], errors="coerce")

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    result["rsi"] = rsi

    return result


def calc_bollinger(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Add Bollinger Bands columns to the DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
        period: Moving average period. Default: 20.
        num_std: Number of standard deviations for bands. Default: 2.0.

    Returns:
        New DataFrame with added boll_mid, boll_upper, boll_lower, boll_width columns.
        - boll_mid: Middle band (SMA)
        - boll_upper: Upper band (middle + num_std * std)
        - boll_lower: Lower band (middle - num_std * std)
        - boll_width: Band width normalized by middle
    """
    result = df.copy()
    close = pd.to_numeric(result["close"], errors="coerce")

    boll_mid = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()

    boll_upper = boll_mid + num_std * rolling_std
    boll_lower = boll_mid - num_std * rolling_std

    # Handle division by zero for width calculation
    boll_width = (boll_upper - boll_lower) / boll_mid.replace(0, np.nan)

    result["boll_mid"] = boll_mid
    result["boll_upper"] = boll_upper
    result["boll_lower"] = boll_lower
    result["boll_width"] = boll_width

    return result


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """Add KDJ (Stochastic Oscillator) columns to the DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
        n: RSV calculation period. Default: 9.
        m1: K value smoothing period. Default: 3.
        m2: D value smoothing period. Default: 3.

    Returns:
        New DataFrame with added K, D, J columns.
        - K: Fast stochastic line
        - D: Slow stochastic line
        - J: Derived line = 3*K - 2*D
    """
    result = df.copy()
    low = pd.to_numeric(result["low"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    close = pd.to_numeric(result["close"], errors="coerce")

    low_min = low.rolling(window=n, min_periods=n).min()
    high_max = high.rolling(window=n, min_periods=n).max()

    # Handle division by zero
    denominator = (high_max - low_min).replace(0, np.nan)
    rsv = (close - low_min) / denominator * 100

    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d

    result["k"] = k
    result["d"] = d
    result["j"] = j

    return result


def calc_volume_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume analysis columns to the DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        New DataFrame with added vol_5d_avg, vol_ratio columns.
        - vol_5d_avg: 5-day moving average of volume
        - vol_ratio: Current volume / 5-day average (volume surge indicator)
    """
    result = df.copy()
    volume = pd.to_numeric(result["volume"], errors="coerce")

    vol_5d_avg = volume.rolling(window=5, min_periods=5).mean()
    vol_ratio = volume / vol_5d_avg.replace(0, np.nan)

    result["vol_5d_avg"] = vol_5d_avg
    result["vol_ratio"] = vol_ratio

    return result


def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all technical indicators to the DataFrame.

    This is a convenience function that applies all indicator calculations
    in sequence. Each calculation adds new columns to the result.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        New DataFrame with all indicator columns added:
        - MA: ma5, ma10, ma20, ma60
        - MACD: dif, dea, macd
        - RSI: rsi
        - Bollinger Bands: boll_mid, boll_upper, boll_lower, boll_width
        - KDJ: k, d, j
        - Volume: vol_5d_avg, vol_ratio
    """
    result = df.copy()
    result = calc_ma(result)
    result = calc_macd(result)
    result = calc_rsi(result)
    result = calc_bollinger(result)
    result = calc_kdj(result)
    result = calc_volume_analysis(result)
    return result
