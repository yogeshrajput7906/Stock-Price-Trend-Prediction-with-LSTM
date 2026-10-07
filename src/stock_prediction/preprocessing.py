"""Time-series preprocessing module ensuring zero data leakage."""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically into train and test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted chronologically.
    train_ratio : float, optional
        Fraction of data to allocate to training, default is 0.8 (80%).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) partitioned chronologically without shuffling.

    Raises
    ------
    ValueError
        If train_ratio is not strictly between 0 and 1, or df is too small.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if len(df) < 2:
        raise ValueError(f"Dataset too small to split: {len(df)} rows")

    split_index = int(len(df) * train_ratio)
    if split_index == 0 or split_index == len(df):
        raise ValueError("Dataset cannot be split with the given train_ratio.")

    train_df = df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    return train_df, test_df


def fit_scaler(
    train_data: Union[np.ndarray, pd.Series, pd.DataFrame],
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> MinMaxScaler:
    """Fit a MinMaxScaler strictly on training data to avoid data leakage.

    Parameters
    ----------
    train_data : Union[np.ndarray, pd.Series, pd.DataFrame]
        Training feature values.
    feature_range : Tuple[float, float], optional
        Target normalization range, default is (0.0, 1.0).

    Returns
    -------
    MinMaxScaler
        Fitted scaler instance.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    if isinstance(train_data, (pd.Series, pd.DataFrame)):
        values = train_data.to_numpy()
    else:
        values = np.asarray(train_data)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    scaler.fit(values)
    return scaler


def transform_data(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Transform values using a pre-fitted scaler.

    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, pd.DataFrame]
        Data to normalize.
    scaler : MinMaxScaler
        Pre-fitted scaler.

    Returns
    -------
    np.ndarray
        2D normalized numpy array.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        values = data.to_numpy()
    else:
        values = np.asarray(data)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    return scaler.transform(values)


def inverse_transform(
    data: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Inverse transform normalized values back to original price scale.

    Parameters
    ----------
    data : np.ndarray
        Normalized values.
    scaler : MinMaxScaler
        Scaler used during original transformation.

    Returns
    -------
    np.ndarray
        Rescaled values in original currency/price scale.
    """
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return scaler.inverse_transform(arr)


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sliding window input sequences and next-step targets.

    Parameters
    ----------
    data : np.ndarray
        Normalized sequential time-series data (1D or 2D).
    sequence_length : int, optional
        Lookback window size (number of past timesteps), default is 60.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: shape (num_sequences, sequence_length, num_features)
        y: shape (num_sequences,) containing target next-step values.

    Raises
    ------
    ValueError
        If sequence_length is non-positive or exceeds data length.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if len(arr) <= sequence_length:
        raise ValueError(
            f"Data length ({len(arr)}) must be greater than sequence_length ({sequence_length})"
        )

    X_list, y_list = [], []
    for i in range(sequence_length, len(arr)):
        X_list.append(arr[i - sequence_length : i])
        y_list.append(arr[i, 0])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def prepare_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_col: str = "Close",
    sequence_length: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """End-to-end leak-free preparation of train/test sequences and scaler.

    The scaler is fitted strictly on the training set. To generate test sequences
    for every day in the test period without discarding the initial `sequence_length`
    days, a lookback buffer from the end of the train set is prepended to the test set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Chronological training subset.
    test_df : pd.DataFrame
        Chronological testing subset.
    feature_col : str, optional
        Target price feature name, default is 'Close'.
    sequence_length : int, optional
        Lookback window size, default is 60.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]
        (X_train, y_train, X_test, y_test, scaler)
    """
    scaler = fit_scaler(train_df[[feature_col]])

    # Scale training data and generate train sequences
    scaled_train = transform_data(train_df[[feature_col]], scaler)
    X_train, y_train = create_sequences(scaled_train, sequence_length=sequence_length)

    # Prepend the last `sequence_length` rows of train to test for seamless evaluation
    lookback_buffer = train_df.iloc[-sequence_length:][[feature_col]]
    test_series = pd.concat([lookback_buffer, test_df[[feature_col]]], ignore_index=True)
    scaled_test = transform_data(test_series, scaler)
    X_test, y_test = create_sequences(scaled_test, sequence_length=sequence_length)

    return X_train, y_train, X_test, y_test, scaler
