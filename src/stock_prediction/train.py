"""End-to-end training pipeline for stock price prediction."""

import logging
import os
import random

import numpy as np
import tensorflow as tf

from stock_prediction.config import (
    BATCH_SIZE,
    DATA_PATH,
    DATE_COLUMN,
    DROPOUT_RATE,
    EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE,
    LSTM_UNITS,
    MA_WINDOWS,
    MODEL_FILE,
    MODELS_DIR,
    PATIENCE,
    PRICE_COLUMN,
    RANDOM_SEED,
    RSI_WINDOW,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    VALIDATION_SPLIT,
)
from stock_prediction.data import load_stock_data
from stock_prediction.evaluate import calculate_metrics, compare_models
from stock_prediction.features import build_features
from stock_prediction.model import build_lstm_model
from stock_prediction.preprocessing import (
    inverse_transform,
    prepare_datasets,
    split_data,
)
from stock_prediction.visualization import (
    plot_actual_vs_predicted,
    plot_price_and_moving_averages,
    plot_rsi,
    plot_training_history,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_training_pipeline() -> dict:
    """Execute the complete training, evaluation, and artifact generation pipeline."""
    set_seed(RANDOM_SEED)

    # 1. Load Data
    logger.info(f"Loading stock data from: {DATA_PATH}")
    df_raw = load_stock_data(DATA_PATH, date_col=DATE_COLUMN, price_col=PRICE_COLUMN)
    logger.info(f"Loaded {len(df_raw)} records spanning {df_raw[DATE_COLUMN].min().date()} to {df_raw[DATE_COLUMN].max().date()}")

    # 2. Feature Engineering
    logger.info("Computing technical indicators (MA20, MA50, RSI)...")
    df_feat = build_features(
        df_raw,
        ma_windows=MA_WINDOWS,
        rsi_window=RSI_WINDOW,
        dropna=True,
    )
    logger.info(f"Dataset after feature engineering and dropping initial NaNs: {len(df_feat)} rows")

    # Save indicator figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_price_and_moving_averages(df_feat, save_path=FIGURES_DIR / "indicators_ma.png")
    plot_rsi(df_feat, save_path=FIGURES_DIR / "indicators_rsi.png")
    logger.info(f"Exploratory figures saved to: {FIGURES_DIR}")

    # 3. Train / Test Split (Chronological)
    logger.info(f"Splitting data chronologically (train_ratio={TRAIN_RATIO})...")
    train_df, test_df = split_data(df_feat, train_ratio=TRAIN_RATIO)
    logger.info(f"Train subset: {len(train_df)} days | Test subset: {len(test_df)} days")

    # 4. Leakage-Safe Scaling & Sequence Creation
    logger.info(f"Fitting scaler strictly on training set and generating {SEQUENCE_LENGTH}-day sequences...")
    X_train, y_train, X_test, y_test, scaler = prepare_datasets(
        train_df=train_df,
        test_df=test_df,
        feature_col=PRICE_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
    )
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 5. Extract Ground Truth and Naive Persistence Baseline
    y_test_dollars = inverse_transform(y_test, scaler).flatten()
    # The last known price in each test input window is today's closing price
    last_known_scaled = X_test[:, -1, :]
    naive_pred_dollars = inverse_transform(last_known_scaled, scaler).flatten()

    # 6. Build and Train Model
    logger.info("Constructing LSTM neural network...")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    logger.info(f"Training LSTM for up to {EPOCHS} epochs (batch_size={BATCH_SIZE}, non-shuffled validation)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=False,  # Critical for time series validation integrity
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Save Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    logger.info(f"Trained model saved to: {MODEL_FILE}")

    # Plot training loss curve
    plot_training_history(history, save_path=FIGURES_DIR / "training_validation_loss.png")

    # 8. Inference and Rescaling
    logger.info("Generating predictions on test set...")
    lstm_pred_scaled = model.predict(X_test, verbose=0)
    lstm_pred_dollars = inverse_transform(lstm_pred_scaled, scaler).flatten()

    # 9. Model Evaluation & Benchmark Comparison
    baseline_metrics = calculate_metrics(
        y_true=y_test_dollars,
        y_pred=naive_pred_dollars,
        previous_prices=naive_pred_dollars,
    )
    lstm_metrics = calculate_metrics(
        y_true=y_test_dollars,
        y_pred=lstm_pred_dollars,
        previous_prices=naive_pred_dollars,
    )

    comparison_report = compare_models(baseline_metrics, lstm_metrics)
    print("\n" + comparison_report + "\n")

    # Plot predictions
    test_dates = test_df[DATE_COLUMN].reset_index(drop=True)
    plot_actual_vs_predicted(
        actual=y_test_dollars,
        lstm_predicted=lstm_pred_dollars,
        naive_predicted=naive_pred_dollars,
        dates=test_dates,
        save_path=FIGURES_DIR / "actual_vs_predicted.png",
    )
    logger.info(f"Evaluation plot saved to: {FIGURES_DIR / 'actual_vs_predicted.png'}")

    return {
        "baseline_metrics": baseline_metrics,
        "lstm_metrics": lstm_metrics,
        "comparison_report": comparison_report,
    }


if __name__ == "__main__":
    run_training_pipeline()
