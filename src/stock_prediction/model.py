"""LSTM neural network model definition."""

from typing import Tuple

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """Construct and compile a stacked LSTM regression model.

    Architecture
    ------------
    Input (sequence_length, features)
      ↓
    LSTM (50 units, return_sequences=True)
      ↓
    Dropout (0.2)
      ↓
    LSTM (50 units, return_sequences=False)
      ↓
    Dropout (0.2)
      ↓
    Dense (25 units, relu)
      ↓
    Dense (1 unit, linear)

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape tuple (sequence_length, num_features).
    lstm_units : int, optional
        Number of recurrent hidden units per LSTM layer, default 50.
    dropout_rate : float, optional
        Dropout regularization fraction, default 0.2.
    learning_rate : float, optional
        Adam optimizer learning rate, default 0.001.

    Returns
    -------
    Sequential
        Compiled Keras sequential model with Mean Squared Error loss.
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units=lstm_units, return_sequences=True),
            Dropout(rate=dropout_rate),
            LSTM(units=lstm_units, return_sequences=False),
            Dropout(rate=dropout_rate),
            Dense(units=25, activation="relu"),
            Dense(units=1),
        ]
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model
