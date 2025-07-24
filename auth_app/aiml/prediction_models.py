import numpy as np
import pandas as pd
import io
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from django.shortcuts import render
from auth_app.models import LSTMModelStorage


def train_linear_model(df):
    df1 = df[['Low', 'Open', 'High', 'Close']].iloc[0:1]
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_model_predictions = linear_model.predict(X_test)

    lm_mse = mean_squared_error(y_test, linear_model_predictions)
    # Add predictions to the corresponding rows in the original dataframe
    df.loc[y_test.index, 'Linear_Model'] = linear_model_predictions

    return linear_model, linear_model_predictions, y_test.values, lm_mse, y_test.index, df1, df


def train_decision_tree_model(stock_df):
    # Assuming stock_df has a 'Date' column and 'Close' price, and is already cleaned and sorted
    stock_df = stock_df.copy()
    stock_df['Target'] = stock_df['Close'].shift(-1)  # Predict next day's Close
    stock_df.dropna(inplace=True)

    X = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    stock_df.loc[y_test.index, 'Decision_Tree_Model'] = predictions

    return model, mse, stock_df


def train_random_forest_model(stock_df):
    stock_df = stock_df.copy()
    stock_df['Target'] = stock_df['Close'].shift(-1)
    stock_df.dropna(inplace=True)
    # Convert all features to numeric (forcefully), and handle issues early
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')

    X = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rf_mse = mean_squared_error(y_test, predictions)
    stock_df.loc[y_test.index, 'Random_Forest_Model'] = predictions

    return model, X_train, X_test, rf_mse, stock_df


def train_svm_model(stock_df):
    stock_df = stock_df.copy()
    stock_df['Target'] = stock_df['Close'].shift(-1)
    stock_df.dropna(inplace=True)

    X = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse_svm = mean_squared_error(y_test, predictions)

    # Inverse transform back to original scale for SHAP
    X_train_df = pd.DataFrame(scaler.inverse_transform(X_train), columns=X.columns)
    X_test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)
    stock_df.loc[y_test.index, 'SVM_Model'] = predictions

    return model, scaler, X_train_df, X_test_df, mse_svm, stock_df


def train_lstm_model(stock_df, sequence_length=60):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    # Step 1: Scale data
    data = stock_df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    x = np.array(x)
    y = np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))

    # Step 2: Train/Test split (80/20)
    split_index = int(len(x) * 0.8)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Step 3: Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Step 4: Predict
    predictions = model.predict(x_test)

    # Step 5: Align predictions with stock_df
    prediction_start_index = sequence_length + split_index
    prediction_indices = stock_df.index[prediction_start_index:prediction_start_index + len(predictions)]

    # Convert predictions back from scaled to original values
    predictions_rescaled = scaler.inverse_transform(predictions)

    # Create new column and fill predictions
    stock_df['LSTM_Model'] = np.nan
    stock_df.loc[prediction_indices, 'LSTM_Model'] = predictions_rescaled.flatten()

    # Calculate MSE in original scale
    actuals_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(actuals_rescaled, predictions_rescaled)

    return model, scaler, sequence_length, mse, stock_df


# STEP 2: Save to DB

# Save LSTM model and scaler to DB for a specific user and stock
def save_lstm_to_db(request, stock_name, model, scaler, sequence_length=60, description=""):
    model_buffer = io.BytesIO()
    scaler_buffer = io.BytesIO()

    joblib.dump(model, model_buffer)
    joblib.dump(scaler, scaler_buffer)

    LSTMModelStorage.objects.update_or_create(
        user=request.user,
        stock_name=stock_name,
        defaults={
            'model_blob': model_buffer.getvalue(),
            'scaler_blob': scaler_buffer.getvalue(),
            'sequence_length': sequence_length,
            'description': description,
        }
    )


# Load LSTM model and scaler from DB for a specific user and stock
def load_lstm_from_db(request, stock_name):
    user = request.user
    try:
        record = LSTMModelStorage.objects.get(user=user, stock_name=stock_name)
        model = joblib.load(io.BytesIO(record.model_blob))
        scaler = joblib.load(io.BytesIO(record.scaler_blob))
        return model, scaler, record.sequence_length
    except LSTMModelStorage.DoesNotExist:
        return None, None, None


# STEP 4: Predict next day
def predict_next_day_lstm(model, scaler, stock_df, sequence_length=60):
    last_data = stock_df['Close'].values[-sequence_length:].reshape(-1, 1)
    scaled_last_data = scaler.transform(last_data)
    X_test = np.reshape(scaled_last_data, (1, sequence_length, 1))
    predicted_price = model.predict(X_test, verbose=0)
    final_prediction = round(scaler.inverse_transform(predicted_price)[0][0], 2)

    return final_prediction


def predict_next_day(model, last_row):
    last_features = last_row[['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1)
    return round(model.predict(last_features)[0], 2)


def predict_next_day_svm(model, scaler, last_row):
    features = last_row[['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    return round(model.predict(features_scaled)[0], 2)


def calculate_avg_error(stock_data, model_col):
    """
    Calculate the average error of the model's predictions.

    :param stock_data: DataFrame containing stock data
    :param model_col: Column name containing the model's predicted 'Close' prices
    :return: Average error as a float
    """
    df = stock_data.copy()

    # Ensure numeric values
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df[model_col] = pd.to_numeric(df[model_col], errors='coerce')

    # Drop rows with NaN values in required columns
    df.dropna(subset=['Close', model_col], inplace=True)

    # Calculate absolute error and average
    errors = abs(df['Close'] - df[model_col])
    avg_error = round(errors.mean(), 2)

    return avg_error


def calculate_success_rate(df, prediction_col, tolerance=0.02):
    # Ensure types are correct
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df[prediction_col] = pd.to_numeric(df[prediction_col], errors='coerce')

    # Drop rows where prediction or actual is NaN after conversion
    df.dropna(subset=['Close', prediction_col], inplace=True)

    actual_close = df['Close']
    predicted_close = df[prediction_col]

    success = abs(actual_close - predicted_close) / actual_close <= tolerance
    success_rate = success.sum() / len(success) * 100

    return round(success_rate, 2)


def calculate_directional_success_rate(stock_data, model_col):
    """
    Calculate the directional success rate by comparing the predicted price change
    to the actual price change.

    :param stock_data: DataFrame containing stock data, including 'Close' price
    :param model_col: Column name containing the model's predicted 'Close' prices
    :return: Directional success rate as a percentage
    """
    df = stock_data.copy()

    # Ensure numeric types
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df[model_col] = pd.to_numeric(df[model_col], errors='coerce')

    # Calculate actual and predicted price change from previous day
    prev_close = df['Close'].shift(1)
    actual_change = df['Close'] - prev_close
    predicted_change = df[model_col] - prev_close

    # Determine correct direction
    correct_direction = ((actual_change > 0) & (predicted_change > 0)) | \
                        ((actual_change < 0) & (predicted_change < 0))

    # Drop NaNs before computing final rate
    correct_direction = correct_direction.dropna()

    # Calculate and return success rate
    directional_success_rate = round(correct_direction.sum() / len(correct_direction) * 100, 2)
    return directional_success_rate


def calculate_model_scores(merged_df):
    import pandas as pd
    # print(merged_df.to_string())
    # merged_df.to_csv("file3333.csv")
    # Weights
    w_success = 0.4
    w_directional = 0.4
    w_error = 0.2

    models = [
        'Linear_Model',
        'Decision_Tree_Model',
        'Random_Forest_Model',
        'SVM_Model',
        'LSTM_Model'
    ]

    results = []

    for idx, row in merged_df.iterrows():
        stock = row['Stock_Name']
        for model in models:
            try:
                s_rate = pd.to_numeric(row[f'{model}_Success_Rate'], errors='coerce')
                d_rate = pd.to_numeric(row[f'{model}_Directional_Success_Rate'], errors='coerce')
                error = pd.to_numeric(row[f'{model}_Avg_Error'], errors='coerce')

                if pd.isna(s_rate) or pd.isna(d_rate) or pd.isna(error):
                    continue

                score = (w_success * (s_rate / 100)) + \
                        (w_directional * (d_rate / 100)) - \
                        (w_error * error)

                results.append({
                    'Stock_Name': stock,
                    'Model': model,
                    'Success_Rate': s_rate,
                    'Directional_Success_Rate': d_rate,
                    'Average_Error': error,
                    'Normalized_Models_Score': score
                })
            except KeyError:
                continue

    df = pd.DataFrame(results)
    df['Best_Model'] = df.groupby('Stock_Name')['Normalized_Models_Score'] \
        .transform(lambda x: x == x.max()) \
        .map({True: 'Yes', False: 'No'})
    df_best_only = df[df['Best_Model'] == 'Yes'].reset_index(drop=True)

    return df, df_best_only
