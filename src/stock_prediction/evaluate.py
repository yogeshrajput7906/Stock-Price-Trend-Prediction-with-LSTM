"""Evaluation metrics and baseline benchmark module."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_prices: np.ndarray,
) -> float:
    """Calculate percentage of correct directional price movement predictions.

    Directional Accuracy measures whether the predicted change (up or down)
    matches the actual market direction from the previous day's price.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth prices at timestep t.
    y_pred : np.ndarray
        Predicted prices at timestep t.
    previous_prices : np.ndarray
        Actual prices at timestep t-1.

    Returns
    -------
    float
        Percentage of correct directional movements (0.0 to 100.0).
    """
    actual_direction = np.sign(y_true - previous_prices)
    predicted_direction = np.sign(y_pred - previous_prices)

    correct = np.equal(actual_direction, predicted_direction)
    return float(np.mean(correct) * 100.0)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_prices: np.ndarray = None,
) -> Dict[str, float]:
    """Calculate regression and forecasting performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth prices.
    y_pred : np.ndarray
        Model predictions.
    previous_prices : np.ndarray, optional
        Lag-1 prices for directional accuracy.

    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, RMSE, MAPE (%), and optionally Directional Accuracy (%).
    """
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = root_mean_squared_error(y_true_flat, y_pred_flat)

    # Mean Absolute Percentage Error (avoid division by 0)
    non_zero_mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) / y_true_flat[non_zero_mask])) * 100.0

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }

    if previous_prices is not None:
        prev_flat = np.asarray(previous_prices).flatten()
        da = calculate_directional_accuracy(y_true_flat, y_pred_flat, prev_flat)
        metrics["Directional_Accuracy"] = float(da)

    return metrics


def generate_naive_baseline(
    X_test_unscaled: np.ndarray,
) -> np.ndarray:
    """Generate persistence baseline predictions: y_hat_t = y_{t-1}.

    In financial time-series forecasting, tomorrow's price is often best
    approximated by today's closing price. The last value of the input sequence
    is today's price.

    Parameters
    ----------
    X_test_unscaled : np.ndarray
        3D array of unscaled input sequences with shape (samples, sequence_length, features).

    Returns
    -------
    np.ndarray
        1D array of persistence predictions (today's price predicting tomorrow).
    """
    # The last element in each lookback window is the most recently observed price
    return X_test_unscaled[:, -1, 0]


def compare_models(
    baseline_metrics: Dict[str, float],
    lstm_metrics: Dict[str, float],
) -> str:
    """Format a clean comparison markdown/text table between Baseline and LSTM."""
    header = f"{'Metric':<25} | {'Naive Baseline':<15} | {'LSTM Model':<15}"
    divider = "-" * len(header)
    rows = [divider, header, divider]

    all_keys = ["MAE", "RMSE", "MAPE"]
    if "Directional_Accuracy" in baseline_metrics and "Directional_Accuracy" in lstm_metrics:
        all_keys.append("Directional_Accuracy")

    for k in all_keys:
        b_val = baseline_metrics.get(k, 0.0)
        l_val = lstm_metrics.get(k, 0.0)
        suffix = "%" if "MAPE" in k or "Directional" in k else "$"
        rows.append(f"{k:<25} | {b_val:>13.4f}{suffix} | {l_val:>13.4f}{suffix}")
    rows.append(divider)
    return "\n".join(rows)
