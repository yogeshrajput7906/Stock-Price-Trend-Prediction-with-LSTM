"""Unit tests for the evaluation metrics and naive baseline module."""

import numpy as np

from stock_prediction.evaluate import (
    calculate_directional_accuracy,
    calculate_metrics,
    generate_naive_baseline,
)


def test_calculate_metrics_exact_values():
    """Verify MAE, RMSE, and MAPE computations on synthetic deterministic data."""
    y_true = np.array([100.0, 102.0, 98.0, 105.0])
    y_pred = np.array([102.0, 100.0, 100.0, 105.0])

    # Absolute errors: [2, 2, 2, 0] -> MAE = 6 / 4 = 1.5
    # Squared errors: [4, 4, 4, 0] -> MSE = 12 / 4 = 3.0 -> RMSE = sqrt(3) ~ 1.73205
    metrics = calculate_metrics(y_true, y_pred)

    assert np.isclose(metrics["MAE"], 1.5)
    assert np.isclose(metrics["RMSE"], np.sqrt(3.0))
    assert metrics["MAPE"] > 0.0


def test_calculate_directional_accuracy():
    """Verify calculation of price movement direction accuracy."""
    y_true = np.array([105.0, 102.0, 106.0])
    prev = np.array([100.0, 105.0, 102.0])  # Actual: [up, down, up]

    # Pred 1: [108 (up), 101 (down), 107 (up)] -> 3/3 = 100%
    y_pred_perfect = np.array([108.0, 101.0, 107.0])
    acc_perfect = calculate_directional_accuracy(y_true, y_pred_perfect, prev)
    assert np.isclose(acc_perfect, 100.0)

    # Pred 2: [99 (down), 101 (down), 101 (down)] -> [wrong, correct, wrong] -> 1/3 = 33.33%
    y_pred_partial = np.array([99.0, 101.0, 101.0])
    acc_partial = calculate_directional_accuracy(y_true, y_pred_partial, prev)
    assert np.isclose(acc_partial, 100.0 / 3.0)


def test_generate_naive_baseline():
    """Verify persistence model extracts the last observation of each sequence."""
    # 2 sequences of length 3 with 1 feature
    X = np.array([
        [[10.0], [11.0], [12.0]],
        [[11.0], [12.0], [13.0]],
    ])
    baseline_pred = generate_naive_baseline(X)
    assert np.array_equal(baseline_pred, np.array([12.0, 13.0]))
