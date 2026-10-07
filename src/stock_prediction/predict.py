"""Inference module for loading a saved model and generating future price predictions."""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf

from stock_prediction.config import (
    DATA_PATH,
    MODEL_FILE,
    PRICE_COLUMN,
    SEQUENCE_LENGTH,
)
from stock_prediction.data import load_stock_data
from stock_prediction.preprocessing import fit_scaler, inverse_transform, transform_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_disk(model_path: Union[str, Path] = MODEL_FILE) -> tf.keras.Model:
    """Load a compiled Keras model from disk.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the saved `.keras` model file.

    Returns
    -------
    tf.keras.Model
        Loaded model instance.

    Raises
    ------
    FileNotFoundError
        If the saved model file cannot be found.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{path.resolve()}'. "
            "Please run 'python -m stock_prediction.train' first."
        )
    return tf.keras.models.load_model(path)


def predict_next_close(
    recent_prices: np.ndarray,
    model: tf.keras.Model,
    scaler,
) -> float:
    """Predict the next trading day's closing price from recent price history.

    Parameters
    ----------
    recent_prices : np.ndarray
        Array containing at least `sequence_length` recent closing prices.
    model : tf.keras.Model
        Trained LSTM model.
    scaler : MinMaxScaler
        Pre-fitted scaler.

    Returns
    -------
    float
        Predicted next closing price in original currency/dollars.
    """
    if len(recent_prices) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} recent prices, got {len(recent_prices)}"
        )

    # Take the latest sequence
    input_slice = recent_prices[-SEQUENCE_LENGTH:].reshape(-1, 1)
    scaled_input = transform_data(input_slice, scaler)
    # Shape into (1, SEQUENCE_LENGTH, 1)
    model_input = scaled_input.reshape(1, SEQUENCE_LENGTH, 1)

    scaled_prediction = model.predict(model_input, verbose=0)
    unscaled_prediction = inverse_transform(scaled_prediction, scaler)
    return float(unscaled_prediction[0, 0])


def main():
    """Run sample inference using the latest historical sequence."""
    logger.info("Loading recent data for inference...")
    df = load_stock_data(DATA_PATH)
    prices = df[PRICE_COLUMN].to_numpy()

    # Fit scaler on full history for latest single-step forward demonstration
    scaler = fit_scaler(prices)

    logger.info(f"Loading trained model from: {MODEL_FILE}")
    model = load_model_from_disk(MODEL_FILE)

    last_price = prices[-1]
    predicted_next = predict_next_close(prices, model, scaler)

    print("\n" + "=" * 45)
    print(f"Latest Recorded Close ({df['Date'].iloc[-1].date()}): ${last_price:.2f}")
    print(f"LSTM Predicted Next Close:               ${predicted_next:.2f}")
    print(f"Predicted Change:                        ${predicted_next - last_price:+.2f}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
