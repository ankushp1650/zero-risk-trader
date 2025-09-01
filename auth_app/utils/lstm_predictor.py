import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
from pathlib import Path


def predict_next_day(symbol, csv_path, sequence_length=60):
    """
    Load trained LSTM model + scaler and predict next day's closing price.
    Args:
        symbol (str): Stock symbol (e.g., "TCS_BSE")
        csv_path (str): Path to CSV file with historical stock data
        sequence_length (int): Number of past days to use for prediction
    Returns:
        float: Predicted next-day closing price
    """
    models_dir = Path("auth_app/models")

    # Load model + scaler
    model_path = models_dir / f"{symbol}_lstm_model.h5"
    scaler_path = models_dir / f"{symbol}_scaler.pkl"

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Load recent stock data
    df = pd.read_csv(csv_path)
    if "Close" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a 'Close' column")

    # Scale data
    scaled_data = scaler.transform(df[['Close']].values)

    # Get last `sequence_length` days
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))

    # Predict
    next_day_scaled = model.predict(last_sequence, verbose=0)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

    return next_day_price
