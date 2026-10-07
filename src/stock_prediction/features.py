"""Feature engineering module for financial technical indicators."""

from typing import Tuple

import numpy as np
import pandas as pd


def add_moving_averages(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = (20, 50),
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute rolling simple moving averages (SMA) for given windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target price column.
    windows : Tuple[int, ...], optional
        Tuple of window sizes (in periods/days), default is (20, 50).
    price_col : str, optional
        Column name to calculate moving averages over, default is 'Close'.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns 'MA{window}' for each window size.
    """
    df = df.copy()
    for window in windows:
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        df[f"MA{window}"] = df[price_col].rolling(window=window).mean()
    return df


def add_rsi(
    df: pd.DataFrame,
    window: int = 14,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI) using standard rolling averages.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target price column.
    window : int, optional
        Lookback period for RSI, default is 14.
    price_col : str, optional
        Column name to calculate RSI over, default is 'Close'.

    Returns
    -------
    pd.DataFrame
        DataFrame with new column 'RSI'. Values range between 0 and 100.
    """
    if window <= 0:
        raise ValueError(f"RSI window must be positive, got {window}")

    df = df.copy()
    delta = df[price_col].diff()

    # Separate positive and negative price changes
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate rolling mean gains and losses
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Relative strength: handle loss == 0 safely
    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is 0 and avg_gain > 0, RSI is 100; if both are 0, RSI is 50
    rsi = np.where(avg_loss == 0, np.where(avg_gain > 0, 100.0, 50.0), rsi)

    df["RSI"] = rsi
    return df


def build_features(
    df: pd.DataFrame,
    ma_windows: Tuple[int, ...] = (20, 50),
    rsi_window: int = 14,
    dropna: bool = True,
) -> pd.DataFrame:
    """Orchestrate feature engineering pipeline for moving averages and RSI.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted chronologically.
    ma_windows : Tuple[int, ...], optional
        Moving average windows, default (20, 50).
    rsi_window : int, optional
        RSI lookback window, default 14.
    dropna : bool, optional
        Whether to drop initial NaN rows resulting from rolling windows.
        Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with MA and RSI indicators.
    """
    df_feat = add_moving_averages(df, windows=ma_windows)
    df_feat = add_rsi(df_feat, window=rsi_window)

    if dropna:
        df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat
