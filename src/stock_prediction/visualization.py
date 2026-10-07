"""Visualization utilities for time-series stock analysis and model evaluation."""

from pathlib import Path
from typing import Optional, Union

import matplotlib

# Use headless backend for portability and CI environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_price_and_moving_averages(
    df: pd.DataFrame,
    date_col: str = "Date",
    price_col: str = "Close",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot stock price alongside 20-day and 50-day moving averages."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[date_col], df[price_col], label="Close Price", color="#1f77b4", linewidth=1.5)

    if "MA20" in df.columns:
        ax.plot(df[date_col], df["MA20"], label="20-Day MA", color="#ff7f0e", linestyle="--", linewidth=1.2)
    if "MA50" in df.columns:
        ax.plot(df[date_col], df["MA50"], label="50-Day MA", color="#2ca02c", linestyle="--", linewidth=1.2)

    ax.set_title("Stock Price with 20-Day and 50-Day Moving Averages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_rsi(
    df: pd.DataFrame,
    date_col: str = "Date",
    rsi_col: str = "RSI",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot Relative Strength Index (RSI) with standard 70/30 overbought/oversold bands."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[date_col], df[rsi_col], label="RSI (14-day)", color="#6f42c1", linewidth=1.5)
    ax.axhline(70, linestyle="--", color="#d62728", alpha=0.8, label="Overbought (70)")
    ax.axhline(30, linestyle="--", color="#2ca02c", alpha=0.8, label="Oversold (30)")
    ax.fill_between(df[date_col], 70, 100, color="#d62728", alpha=0.08)
    ax.fill_between(df[date_col], 0, 30, color="#2ca02c", alpha=0.08)

    ax.set_title("Relative Strength Index (RSI)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("RSI Value", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_history(
    history,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot training and validation loss curves across training epochs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["loss"], label="Training Loss (MSE)", color="#1f77b4", linewidth=1.5)

    if "val_loss" in history.history:
        ax.plot(epochs, history.history["val_loss"], label="Validation Loss (MSE)", color="#ff7f0e", linestyle="--", linewidth=1.5)

    ax.set_title("LSTM Model Training History", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Squared Error", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_actual_vs_predicted(
    actual: pd.Series,
    lstm_predicted: pd.Series,
    naive_predicted: Optional[pd.Series] = None,
    dates: Optional[pd.Series] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot actual test prices vs LSTM predictions and baseline."""
    fig, ax = plt.subplots(figsize=(13, 6))
    x_axis = dates if dates is not None else range(len(actual))

    ax.plot(x_axis, actual, label="Actual Price", color="#111111", linewidth=2.0)
    ax.plot(x_axis, lstm_predicted, label="LSTM Prediction", color="#1f77b4", linewidth=1.6)

    if naive_predicted is not None:
        ax.plot(x_axis, naive_predicted, label="Naive Persistence Baseline (Lag 1)", color="#ff7f0e", linestyle=":", alpha=0.8, linewidth=1.4)

    ax.set_title("Test Set Price Predictions: LSTM vs Actual vs Naive Baseline", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date" if dates is not None else "Trading Days", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
