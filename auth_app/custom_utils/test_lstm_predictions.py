import os
import warnings
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

# -------------------------------
# ENVIRONMENT SETTINGS
# -------------------------------
# Suppress TensorFlow info and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show errors
warnings.filterwarnings("ignore")  # Suppress Python warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN floating-point differences


# -------------------------------
# LSTM PREDICTION FUNCTION
# -------------------------------
def predict_next_day(model_path, input_sequence):
    """
    Loads a pre-trained LSTM model and predicts the next day value.
    """
    # Load model without compiling (prevents compile warnings)
    model = load_model(model_path, compile=False)

    # Optional: compile if you want to use model.evaluate()
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    # Ensure input is the correct shape (1, timesteps, features)
    input_sequence = np.array(input_sequence).reshape(1, -1, 1)

    # Predict
    prediction = model.predict(input_sequence, verbose=0)
    return float(prediction[0][0])


# -------------------------------
# EXAMPLE USAGE
# -------------------------------
if __name__ == "__main__":
    # Example: paths to your saved LSTM models
    model_paths = {
        "TCS_BSE": r"C:\Users\Ankush\PycharmProjects\zero-risk-trader\models\TCS_BSE_lstm_model.h5",
        "RELIANCE_BSE": r"C:\Users\Ankush\PycharmProjects\zero-risk-trader\models\RELIANCE_BSE_lstm_model.h5",
        "HDFCBANK_BSE": r"C:\Users\Ankush\PycharmProjects\zero-risk-trader\models\HDFCBANK_BSE_lstm_model.h5",
        "BHARTIARTL_BSE": r"C:\Users\Ankush\PycharmProjects\zero-risk-trader\models\BHARTIARTL_BSE_lstm_model.h5",
        "ICICIBANK_BSE": r"C:\Users\Ankush\PycharmProjects\zero-risk-trader\models\ICICIBANK_BSE_lstm_model.h5"
    }

    # Example dummy input sequences (replace with your real sequences)
    dummy_input = [np.random.rand(60) for _ in range(len(model_paths))]

    print("🔍 Evaluating LSTM models...\n")
    for i, (stock, path) in enumerate(model_paths.items()):
        next_day = predict_next_day(path, dummy_input[i])
        print(f"📊 {stock}:")
        print(f"   📈 Next-Day Prediction = {next_day:.2f}\n")
