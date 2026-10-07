"""Unit tests for the technical indicators feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from stock_prediction.features import add_moving_averages, add_rsi, build_features


@pytest.fixture
def sample_stock_df() -> pd.DataFrame:
    """Generate deterministic synthetic stock data for testing."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = 100.0 + np.sin(np.linspace(0, 10, n)) * 10.0 + np.arange(n) * 0.2
    return pd.DataFrame({"Date": dates, "Close": prices})


def test_add_moving_averages_calculation(sample_stock_df: pd.DataFrame):
    """Verify MA columns are created with mathematically correct rolling means."""
    df_ma = add_moving_averages(sample_stock_df, windows=(20, 50))

    assert "MA20" in df_ma.columns
    assert "MA50" in df_ma.columns

    # First 19 rows should be NaN for MA20
    assert df_ma["MA20"].iloc[:19].isna().all()
    assert not pd.isna(df_ma["MA20"].iloc[19])

    # Check exact average for index 19 (first 20 rows)
    expected_ma20 = sample_stock_df["Close"].iloc[:20].mean()
    assert np.isclose(df_ma["MA20"].iloc[19], expected_ma20)

    # First 49 rows should be NaN for MA50
    assert df_ma["MA50"].iloc[:49].isna().all()
    assert not pd.isna(df_ma["MA50"].iloc[49])


def test_add_moving_averages_invalid_window(sample_stock_df: pd.DataFrame):
    """Verify that non-positive windows raise ValueError."""
    with pytest.raises(ValueError):
        add_moving_averages(sample_stock_df, windows=(0, 20))


def test_add_rsi_bounds_and_structure(sample_stock_df: pd.DataFrame):
    """Verify RSI stays strictly within [0, 100] and has NaN for initial window."""
    df_rsi = add_rsi(sample_stock_df, window=14)

    assert "RSI" in df_rsi.columns
    assert df_rsi["RSI"].iloc[:14].isna().all()

    valid_rsi = df_rsi["RSI"].dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_add_rsi_strictly_increasing_and_decreasing():
    """Verify RSI hits extremes on monotonic trends."""
    up_prices = pd.DataFrame({"Close": [float(i) for i in range(1, 30)]})
    df_up = add_rsi(up_prices, window=14)
    assert np.isclose(df_up["RSI"].iloc[-1], 100.0)

    down_prices = pd.DataFrame({"Close": [float(50 - i) for i in range(30)]})
    df_down = add_rsi(down_prices, window=14)
    assert np.isclose(df_down["RSI"].iloc[-1], 0.0)


def test_build_features_pipeline(sample_stock_df: pd.DataFrame):
    """Verify end-to-end feature pipeline produces clean dataset without NaNs."""
    df_feat = build_features(sample_stock_df, ma_windows=(20, 50), rsi_window=14, dropna=True)

    expected_cols = {"Date", "Close", "MA20", "MA50", "RSI"}
    assert expected_cols.issubset(set(df_feat.columns))
    assert df_feat.isna().sum().sum() == 0
    # Length should be original length minus 49 initial rows dropped by 50-day window
    assert len(df_feat) == len(sample_stock_df) - 49
