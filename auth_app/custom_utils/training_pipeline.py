import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path


# Create sequences for LSTM
def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


# Train + Save per stock
def train_and_save_lstm_pipeline(stock_df, symbol, sequence_length=60, epochs=50, batch_size=32):
    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_df[['Close']].values)

    # Sequences
    X, y = create_sequences(scaled_data, sequence_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    # Train
    print(f"🚀 Training LSTM for {symbol} ...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    # Create output dir
    models_dir = Path("../../models/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save with stock name
    model_path = models_dir / f"{symbol}_lstm_model.h5"
    scaler_path = models_dir / f"{symbol}_scaler.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Saved model: {model_path}")
    print(f"✅ Saved scaler: {scaler_path}\n")


if __name__ == "__main__":
    data_dir = Path("data")
    stock_files = [
        "TCS_BSE_full.csv",
        "RELIANCE_BSE_full.csv",
        "HDFCBANK_BSE_full.csv",
        "BHARTIARTL_BSE_full.csv",
        "ICICIBANK_BSE_full.csv"
    ]

    for file in stock_files:
        symbol = file.replace("_full.csv", "")  # e.g., TCS_BSE
        stock_df = pd.read_csv(data_dir / file)
        train_and_save_lstm_pipeline(stock_df, symbol)

# This script trains and saves LSTM models for stock price prediction.
# It creates sequences from the stock data, builds an LSTM model, trains it, and saves both the model and scaler.
# It assumes stock data is in CSV files located in a "data" directory.
