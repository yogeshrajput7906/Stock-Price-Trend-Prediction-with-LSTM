"""Configuration parameters for the stock price prediction pipeline."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "AAPL.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "lstm_stock_model.keras"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility
RANDOM_SEED = 42

# Time Series & Preprocessing
SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.8
PRICE_COLUMN = "Close"
DATE_COLUMN = "Date"

# Technical Indicators
MA_WINDOWS = (20, 50)
RSI_WINDOW = 14

# LSTM Architecture & Training Hyperparameters
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
VALIDATION_SPLIT = 0.1
