"""Unit tests for the leakage-safe preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from stock_prediction.preprocessing import (
    create_sequences,
    fit_scaler,
    inverse_transform,
    prepare_datasets,
    split_data,
    transform_data,
)


@pytest.fixture
def synthetic_time_series() -> pd.DataFrame:
    """Generate 100 days of synthetic ordered stock prices."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = np.linspace(100.0, 200.0, 100)
    return pd.DataFrame({"Date": dates, "Close": prices})


def test_split_data_chronological_order(synthetic_time_series: pd.DataFrame):
    """Verify that split_data does not shuffle and preserves temporal ordering."""
    train_df, test_df = split_data(synthetic_time_series, train_ratio=0.8)

    assert len(train_df) == 80
    assert len(test_df) == 20

    # Ensure temporal boundary is strictly respected
    assert train_df["Date"].max() < test_df["Date"].min()
    assert train_df["Close"].max() < test_df["Close"].min()


def test_split_data_invalid_ratios(synthetic_time_series: pd.DataFrame):
    """Verify ValueError is raised on invalid split ratios."""
    with pytest.raises(ValueError):
        split_data(synthetic_time_series, train_ratio=1.2)
    with pytest.raises(ValueError):
        split_data(synthetic_time_series, train_ratio=-0.1)


def test_scaler_avoids_data_leakage():
    """Verify scaler min/max comes purely from training set."""
    train_prices = np.array([10.0, 20.0, 30.0]).reshape(-1, 1)
    test_prices = np.array([5.0, 40.0]).reshape(-1, 1)

    scaler = fit_scaler(train_prices)
    assert scaler.data_min_[0] == 10.0
    assert scaler.data_max_[0] == 30.0

    scaled_train = transform_data(train_prices, scaler)
    assert np.allclose(scaled_train, [[0.0], [0.5], [1.0]])

    # Because test data contains values outside [10, 30], scaled test will exceed [0, 1]
    # This mathematically proves that test distribution was NOT leaked into scaler parameters!
    scaled_test = transform_data(test_prices, scaler)
    assert scaled_test[0, 0] < 0.0  # 5.0 scaled is -0.25
    assert scaled_test[1, 0] > 1.0  # 40.0 scaled is 1.5

    # Inverse transformation should reconstruct exact original values
    reconstructed_test = inverse_transform(scaled_test, scaler)
    assert np.allclose(reconstructed_test, test_prices)


def test_create_sequences_shapes_and_alignment():
    """Verify sequence generator shapes, length, and temporal target alignment."""
    data = np.arange(10, dtype=np.float32).reshape(-1, 1)
    seq_length = 3

    X, y = create_sequences(data, sequence_length=seq_length)

    # Expected sequences: 10 - 3 = 7
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)

    # Check first sequence: X[0] should be [0, 1, 2], y[0] should be 3
    assert np.array_equal(X[0].flatten(), np.array([0, 1, 2]))
    assert y[0] == 3.0

    # Check last sequence: X[-1] should be [6, 7, 8], y[-1] should be 9
    assert np.array_equal(X[-1].flatten(), np.array([6, 7, 8]))
    assert y[-1] == 9.0


def test_create_sequences_invalid_length():
    """Verify ValueError is raised if sequence_length >= data length."""
    data = np.array([1, 2, 3]).reshape(-1, 1)
    with pytest.raises(ValueError):
        create_sequences(data, sequence_length=5)


def test_prepare_datasets_pipeline(synthetic_time_series: pd.DataFrame):
    """Verify prepare_datasets integrates train and test cleanly without target leakage."""
    train_df, test_df = split_data(synthetic_time_series, train_ratio=0.8)
    seq_len = 10

    X_train, y_train, X_test, y_test, scaler = prepare_datasets(
        train_df, test_df, feature_col="Close", sequence_length=seq_len
    )

    # Train had 80 rows -> 80 - 10 = 70 sequences
    assert X_train.shape == (70, seq_len, 1)
    assert y_train.shape == (70,)

    # Test had 20 rows -> with 10 lookback buffer, produces exactly 20 test sequences!
    assert X_test.shape == (20, seq_len, 1)
    assert y_test.shape == (20,)

    # The unscaled y_test matches test_df["Close"]
    unscaled_y_test = inverse_transform(y_test, scaler).flatten()
    assert np.allclose(unscaled_y_test, test_df["Close"].to_numpy())
