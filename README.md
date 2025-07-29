# Stock Price Trend Prediction with LSTM

This project predicts stock price trends using a Long Short-Term Memory (LSTM) neural network model. It uses historical data, moving averages, and the Relative Strength Index (RSI) to make predictions on future price movements.

---

## Tools & Technologies Used
- Python
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## Dataset
- **File:** `AAPL.csv`
- **Description:** Daily historical closing prices of Apple Inc. (AAPL).
- You can use any stock dataset with a `Date` and `Close` column.

---

## Features
- 20-Day and 50-Day Moving Averages
- Relative Strength Index (RSI)
- Data Normalization
- Sequence preparation for LSTM
- Train/Test split
- LSTM-based time series prediction
- Visual comparison between actual vs predicted prices

---

## How It Works

1. **Data Loading:**
   - Reads data from `AAPL.csv` or uses dummy data if the file is missing.

2. **Feature Engineering:**
   - Adds 20-day and 50-day moving averages.
   - Computes RSI using 14-day average gains/losses.

3. **Preprocessing:**
   - Normalizes the `Close` prices using MinMaxScaler.
   - Prepares sequences for LSTM.

4. **Model Building:**
   - Builds an LSTM model using TensorFlow/Keras.
   - Trains using 80% of data and tests on the remaining 20%.

5. **Visualization:**
   - Actual vs predicted stock prices.
   - Close prices with Moving Averages.
   - RSI graph with thresholds.

---

## Results

- LSTM successfully captures trends in stock price movements.
- Added technical indicators (MA20, MA50, RSI) for deeper insight.

---

## Folder Structure
**Stock-Price-Trend-Prediction**
- ├── AAPL.csv
- ├── Stock_LSTM_Prediction.py
- ├── README.md
- └── requirements.txt (optional)

 
---

## ✅ Requirements

To install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow.



## Author
YOGESH RAJPUT
- Project: Stock Price Trend Prediction using LSTM


