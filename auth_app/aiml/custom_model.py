
from sklearn.ensemble import RandomForestRegressor
from auth_app.custom_utils.graphs import fig_to_base64
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('Agg')  # must be set before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from keras.optimizers import Adam, SGD, RMSprop
# from keras.callbacks import EarlyStopping
# from keras.losses import MeanSquaredError, MeanAbsoluteError
# from keras import Sequential
# from keras.src.layers import LSTM, Dense




# def train_test_data(df):
#     df['Target'] = df['Close'].shift(-1)
#     df.dropna(inplace=True)
#     X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
#     y = df['Target']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#     return X_train, X_test, y_train, y_test

def train_test_data(df):
    # print(df.columns)
    # print(df.dtypes)
    # print("Original df tail:\n", df.head(3)[['Date', 'Close']])

    # Convert 'Date' to datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        # print(df['Date'])
        df.set_index('Date', inplace=True)

    else:
        df.index = pd.to_datetime(df.index)

    # print(df.index.strftime('%Y-%m-%d'))

    # Convert all price columns to numeric (handling strings or bad values)
    cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN (after conversion)
    df.dropna(subset=cols_to_convert, inplace=True)

    # Create 'Target' and drop resulting NaN
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    df.sort_index(ascending=True, inplace=True)

    # print("Tail after shift:\n", df[['Date', 'Close', 'Target']].tail(3))

    # Features and target
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Target']
    # print(y.dtypes)
    # print(y.index.strftime('%Y-%m-%d'))
    # Split without shuffling and return (with datetime index preserved)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # test_size = 0.2
    # split_index = int(len(df) * (1 - test_size))
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    # print("X_train")
    # print(X_train.head(3))
    # print("y_train")
    # print(y_train.head(3))
    # print("X_test")
    # print(X_test.head(3))
    # print("y_test")
    # print(y_test.head(3))

    return X_train, X_test, y_train, y_test


def evaluate_model_metrics(y_true, y_pred, rmse):
    # Average Error
    avg_error = np.mean(np.abs(y_true - y_pred))

    # Normalized Score
    normalized_score = 1 / (1 + rmse)

    # Success Rate: within 5% tolerance
    tolerance = 0.05
    success_flags = np.abs(y_true - y_pred) <= (tolerance * y_true)
    success_rate = np.sum(success_flags) / len(y_true)

    # Directional Success
    if len(y_true) > 1:
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(np.diff(y_pred))
        directional_success = np.sum(actual_direction == predicted_direction) / len(actual_direction)
    else:
        directional_success = None

    return avg_error, normalized_score, success_rate, directional_success


def linear_model_hyper_tuning_chart(X_train, X_test, y_train, y_test, fit_intercept, regularization, alpha, stock_name):
    # Initialize the model
    if regularization == "ridge":
        model = Ridge(alpha=alpha)  # L2 regularization (Ridge)
        model_name = f"Ridge Regression (alpha={alpha})"
    elif regularization == "lasso":
        model = Lasso(alpha=alpha)  # L1 regularization (Lasso)
        model_name = f"Lasso Regression (alpha={alpha})"
    else:
        model = LinearRegression(fit_intercept=fit_intercept)
        model_name = "Linear Regression"

    # Fit the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(predictions)
    preds = np.array(predictions, dtype=float)
    # y_test = np.array(y_test, dtype=float)
    # y_index = y_test.index

    # Make y_test a pandas Series (with original index if needed)
    # if not isinstance(y_test, pd.Series):
    #     y_test = pd.Series(y_test)

    # y_index = y_test.index
    # Keep original index for y_test if it exists
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)
    print(y_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Extended metrics
    avg_error, normalized_score, success_rate, directional_success = evaluate_model_metrics(y_test.values, preds, rmse)

    # Summary DataFrame
    summary_lm_df = pd.DataFrame([{
        'Stock': stock_name,
        'Model': model_name,
        # 'Index': y_index[0],
        'Index': y_test.index[0].strftime('%Y-%m-%d'),
        'Actual_Close': y_test.values[0],
        'Predicted_Close': preds[0],
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'Success_Rate': success_rate,
        'Directional_Success': directional_success,
        'Avg_Error': avg_error,
        'Normalized_Score': normalized_score,
        'Best_Model': False  # To be updated after all models
    }])
    # print(summary_lm_df.to_string())
    # Model coefficients (relevant for linear models with regularization)
    model_coeffs = None
    if isinstance(model, LinearRegression) or isinstance(model, Ridge) or isinstance(model, Lasso):
        model_coeffs = model.coef_

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Larger figure size for better visibility

    # Time series plot with gridlines
    axs[0].plot(range(len(y_test)), y_test, label='Actual', color='blue', linewidth=2)
    axs[0].plot(range(len(preds)), preds, label='Predicted', color='red', linestyle='--', linewidth=2)
    axs[0].set_title(f'Actual vs Predicted Prices ({model_name})', fontsize=14)
    axs[0].set_xlabel('Time / Index', fontsize=12)
    axs[0].set_ylabel('Price', fontsize=12)
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Scatter plot with enhanced styling
    scatter = axs[1].scatter(y_test, preds, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    axs[1].set_title(f'Predicted vs Actual Prices ({model_name})', fontsize=14)
    axs[1].set_xlabel('Actual Price', fontsize=12)
    axs[1].set_ylabel('Predicted Price', fontsize=12)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display R² and RMSE on the plot for evaluation (optional)
    axs[0].text(0.05, 0.95, f"R² = {r2:.3f}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')
    axs[0].text(0.05, 0.90, f"MSE = {mse:.3f}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')
    axs[0].text(0.05, 0.85, f"RMSE = {rmse:.3f}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

    if model_coeffs is not None:
        coeff_str = '\n'.join([f"{feature}: {coef:.4f}" for feature, coef in zip(X_train.columns, model.coef_)])
        axs[0].text(0.05, 0.80, f"Model Coefficients:\n{coeff_str}", transform=axs[0].transAxes, fontsize=10,
                    verticalalignment='top')

    plt.tight_layout(pad=3.0)
    return fig_to_base64(fig), summary_lm_df


def decision_tree_hyper_tuning_chart(X_train, X_test, y_train, y_test, max_depth, min_samples_split,
                                     criterion, stock_name):
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    preds = np.array(predictions, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    preds = np.nan_to_num(preds)
    y_test = np.nan_to_num(y_test)

    # Make y_test a pandas Series (with original index if needed)
    # Keep original index for y_test if it exists
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)

    # y_index = y_test.index

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    # # Create a one-row summary DataFrame
    # summary_dt_df = pd.DataFrame([{
    #     'Stock': stock_name,
    #     'Model': "Decision Tree",
    #     'Index': y_index[0],  # or use y_index[-1] for the last point
    #     'Actual_Close': y_test.values[0],
    #     'Predicted_Close': preds[0],
    #     'R2': r2,
    #     'MSE': mse,
    #     'RMSE': rmse
    # }])

    # Extended metrics
    avg_error, normalized_score, success_rate, directional_success = evaluate_model_metrics(y_test.values, preds, rmse)

    # Summary DataFrame
    summary_dt_df = pd.DataFrame([{
        'Stock': stock_name,
        'Model': "Decision Tree",
        'Index': y_test.index[0].strftime('%Y-%m-%d'),
        'Actual_Close': y_test.values[0],
        'Predicted_Close': preds[0],
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'Success_Rate': success_rate,
        'Directional_Success': directional_success,
        'Avg_Error': avg_error,
        'Normalized_Score': normalized_score,
        'Best_Model': False  # To be updated after all models
    }])
    # First Plot: Actual vs Predicted and Scatter
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f'Decision Tree Model\nCriterion: {criterion}, max_depth={max_depth}, min_samples_split={min_samples_split}',
        fontsize=16)

    axs[0].plot(y_test, label='Actual', color='blue', linewidth=2)
    axs[0].plot(preds, label='Predicted', color='orange', linestyle='--', linewidth=2)
    axs[0].set_title('Actual vs Predicted', fontsize=14)
    axs[0].set_xlabel('Time / Index', fontsize=12)
    axs[0].set_ylabel('Price', fontsize=12)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[1].scatter(y_test, preds, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    axs[1].set_title('Predicted vs Actual Scatter', fontsize=14)
    axs[1].set_xlabel('Actual Price', fontsize=12)
    axs[1].set_ylabel('Predicted Price', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Evaluation metrics
    # axs[0].text(0.02, 0.95, f"R² = {r2:.3f}\nMSE = {mse:.3f}\nRMSE = {rmse:.3f}",
    #             transform=axs[0].transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round'))
    # Display R² and RMSE on the plot for evaluation (optional)
    axs[0].text(0.05, 0.95, f"R² = {r2:.3f}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')
    axs[0].text(0.05, 0.90, f"MSE = {mse:.3f}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')
    axs[0].text(0.05, 0.85, f"RMSE = {rmse:.3f}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    main_chart = fig_to_base64(fig)

    # Second Plot: Feature Importances
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in
                                                                         range(X_train.shape[1])]
    importances = model.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    ax_imp.barh(sorted_names, sorted_importances, color='steelblue', edgecolor='black')
    ax_imp.set_title(f'Feature Importances\n(Criterion: {criterion}, max_depth={max_depth})', fontsize=14)
    ax_imp.set_xlabel('Importance', fontsize=12)
    ax_imp.set_ylabel('Features', fontsize=12)
    ax_imp.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    importance_chart = fig_to_base64(fig_imp)

    return main_chart, importance_chart, summary_dt_df


def random_forest_hyper_tuning_chart(X_train, X_test, y_train, y_test,
                                     n_estimators, rf_max_depth,
                                     min_samples_split, criterion, stock_name):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(rf_max_depth) if rf_max_depth != 'None' else None,
        min_samples_split=int(min_samples_split),
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # Convert predictions and y_test to float arrays safely
    predictions = np.array(pred, dtype=np.float64)
    # y_test = np.array(y_test, dtype=np.float64)

    # # Make y_test a pandas Series (with original index if needed)
    # if not isinstance(y_test, pd.Series):
    #     y_test = pd.Series(y_test)

    # y_index = y_test.index
    # Keep original index for y_test if it exists
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Create a one-row summary DataFrame
    # summary_rf_df = pd.DataFrame([{
    #     'Stock': stock_name,
    #     'Model': "Random Forest",
    #     'Index': y_index[0],  # or use y_index[-1] for the last point
    #     'Actual_Close': y_test.values[0],
    #     'Predicted_Close': predictions[0],
    #     'R2': r2,
    #     'MSE': mse,
    #     'RMSE': rmse
    # }])

    # Extended metrics
    avg_error, normalized_score, success_rate, directional_success = evaluate_model_metrics(y_test.values, predictions,
                                                                                            rmse)

    # Summary DataFrame
    summary_rf_df = pd.DataFrame([{
        'Stock': stock_name,
        'Model': "Random Forest",
        'Index': y_test.index[0].strftime('%Y-%m-%d'),
        'Actual_Close': y_test.values[0],
        'Predicted_Close': predictions[0],
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'Success_Rate': success_rate,
        'Directional_Success': directional_success,
        'Avg_Error': avg_error,
        'Normalized_Score': normalized_score,
        'Best_Model': False  # To be updated after all models
    }])
    # Plot 1: Actual vs Predicted and Scatter
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(range(len(y_test)), y_test, label='Actual', color='blue', linewidth=2)
    axs[0].plot(range(len(predictions)), predictions, label='Predicted', color='orange', linestyle='--', linewidth=2)
    axs[0].set_title(f'Random Forest Predictions\nn_estimators={n_estimators}, max_depth={rf_max_depth}, '
                     f'min_samples_split={min_samples_split}, criterion={criterion}', fontsize=11)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].scatter(y_test, predictions, alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axs[1].set_title('Predicted vs Actual', fontsize=13)
    axs[1].grid(True)

    axs[0].text(0.02, 0.95, f"R² = {r2:.3f}", transform=axs[0].transAxes)
    axs[0].text(0.02, 0.90, f"MSE = {mse:.3f}", transform=axs[0].transAxes)
    axs[0].text(0.02, 0.85, f"RMSE = {rmse:.3f}", transform=axs[0].transAxes)

    plt.tight_layout()
    main_chart = fig_to_base64(fig)

    # Plot 2: Feature Importances
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in
                                                                         range(X_train.shape[1])]
    importances = model.feature_importances_

    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    bars = ax_imp.bar(feature_names, importances, color='lightgreen', edgecolor='black')
    ax_imp.set_title('Feature Importances (Random Forest)')
    ax_imp.set_ylabel('Importance')
    ax_imp.set_xticks(range(len(feature_names)))
    ax_imp.set_xticklabels(feature_names, rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax_imp.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    importance_chart = fig_to_base64(fig_imp)

    return main_chart, importance_chart, summary_rf_df


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return image_base64


def svm_hyper_tuning_chart(X_train, X_test, y_train, y_test, kernel, C, epsilon, gamma, degree, coef0, stock_name):
    # Scale the features

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=degree, coef0=coef0)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    predictions = np.array(pred, dtype=np.float64)
    # y_test = np.array(y_test, dtype=np.float64)
    # Keep original index for y_test if it exists
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Make y_test a pandas Series (with original index if needed)
    # if not isinstance(y_test, pd.Series):
    #     y_test = pd.Series(y_test)
    #
    # y_index = y_test.index

    # Create a one-row summary DataFrame
    # summary_svm_df = pd.DataFrame([{
    #     'Stock': stock_name,
    #     'Model': "SVM",
    #     'Index': y_index[0],  # or use y_index[-1] for the last point
    #     'Actual_Close': y_test.values[0],
    #     'Predicted_Close': predictions[0],
    #     'R2': r2,
    #     'MSE': mse,
    #     'RMSE': rmse
    # }])
    # Extended metrics
    avg_error, normalized_score, success_rate, directional_success = evaluate_model_metrics(y_test.values, predictions,
                                                                                            rmse)

    # Summary DataFrame
    summary_svm_df = pd.DataFrame([{
        'Stock': stock_name,
        'Model': "SVM",
        'Index': y_test.index[0].strftime('%Y-%m-%d'),
        'Actual_Close': y_test.values[0],
        'Predicted_Close': predictions[0],
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'Success_Rate': success_rate,
        'Directional_Success': directional_success,
        'Avg_Error': avg_error,
        'Normalized_Score': normalized_score,
        'Best_Model': False  # To be updated after all models
    }])

    # First Plot: Actual vs Predicted
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(range(len(y_test)), y_test, label='Actual', color='blue', linewidth=2)
    axs[0].plot(range(len(predictions)), predictions, label='Predicted', color='red', linestyle='--', linewidth=2)
    axs[0].set_title(f'SVM Prediction\nKernel={kernel}, C={C}, ε={epsilon}, γ={gamma}, deg={degree}, coef0={coef0}',
                     fontsize=11)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].text(0.05, 0.95, f"R² = {r2:.3f}", transform=axs[0].transAxes)
    axs[0].text(0.05, 0.90, f"MSE = {mse:.3f}", transform=axs[0].transAxes)
    axs[0].text(0.05, 0.85, f"RMSE = {rmse:.3f}", transform=axs[0].transAxes)

    # Second Plot: Scatter of Predictions vs Actual
    axs[1].scatter(y_test, predictions, alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axs[1].set_title('Predicted vs Actual Prices (SVM)', fontsize=13)
    axs[1].grid(True)

    plt.tight_layout()
    main_chart_svm = fig_to_base64(fig)

    # Feature importance placeholder (SVR does not have feature_importances_)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 3))
    ax_imp.text(0.5, 0.5, "SVR does not support feature importance natively.",
                ha='center', va='center', fontsize=12, color='gray')
    ax_imp.axis('off')
    importance_chart_svm = fig_to_base64(fig_imp)

    return main_chart_svm, importance_chart_svm, summary_svm_df

#
# def lstm_hyper_tuning_chart(X_train, X_test, y_train, y_test,
#                             lstm_units, epochs, batch_size,
#                             learning_rate, dropout, optimizer, num_layers,
#                             loss_function, activation_function, stock_name):
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from keras.models import Sequential
#     from keras.layers import LSTM, Dense, Dropout
#     from keras.optimizers import Adam, SGD, RMSprop
#     from keras.losses import MeanSquaredError, MeanAbsoluteError
#     from keras.callbacks import EarlyStopping
#     from sklearn.metrics import r2_score, mean_squared_error
#     from keras import backend as K
#
#     # ----------------------------
#     # 1. Convert data to numeric
#     # ----------------------------
#     for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
#         if col in X_train.columns:
#             X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
#         if col in X_test.columns:
#             X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
#
#     y_train = pd.to_numeric(y_train, errors='coerce')
#     y_test = pd.to_numeric(y_test, errors='coerce')
#
#     # Drop NaNs and align
#     X_train = X_train.dropna()
#     y_train = y_train.loc[X_train.index]
#     X_test = X_test.dropna()
#     y_test = y_test.loc[X_test.index]
#
#     # ----------------------------
#     # 2. Convert to numpy and reshape
#     # ----------------------------
#     X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])).astype(np.float32)
#     X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])).astype(np.float32)
#     y_train = np.array(y_train, dtype=np.float32)
#     y_test_values = np.array(y_test, dtype=np.float32)  # keep y_test index separately
#
#     # ----------------------------
#     # 3. Optimizer & Loss
#     # ----------------------------
#     opt = {
#         'adam': Adam(learning_rate=learning_rate),
#         'sgd': SGD(learning_rate=learning_rate)
#     }.get(optimizer.lower(), RMSprop(learning_rate=learning_rate))
#
#     loss = MeanSquaredError() if loss_function.lower() == 'mse' else MeanAbsoluteError()
#
#     # ----------------------------
#     # 4. Build LSTM model
#     # ----------------------------
#     model = Sequential()
#     for i in range(num_layers):
#         return_seq = i < num_layers - 1
#         model.add(LSTM(units=lstm_units,
#                        activation=activation_function,
#                        return_sequences=return_seq,
#                        input_shape=(X_train.shape[1], X_train.shape[2]) if i == 0 else None))
#         model.add(Dropout(dropout))
#     model.add(Dense(1))
#     model.compile(optimizer=opt, loss=loss)
#
#     # ----------------------------
#     # 5. Train model
#     # ----------------------------
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     history = model.fit(X_train, y_train,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(X_test, y_test_values),
#                         callbacks=[early_stop],
#                         verbose=0)
#
#     # ----------------------------
#     # 6. Predictions & Metrics
#     # ----------------------------
#     predictions = model.predict(X_test).flatten()
#
#     r2 = r2_score(y_test_values, predictions)
#     mse = mean_squared_error(y_test_values, predictions)
#     rmse = np.sqrt(mse)
#
#     # Use your evaluate_model_metrics function if exists
#     avg_error, normalized_score, success_rate, directional_success = evaluate_model_metrics(
#         y_test_values, predictions, rmse
#     )
#
#     # ----------------------------
#     # 7. Summary DataFrame
#     # ----------------------------
#     summary_lstm_df = pd.DataFrame([{
#         'Stock': stock_name,
#         'Model': "LSTM",
#         'Index': y_test.index[0].strftime('%Y-%m-%d'),
#         'Actual_Close': y_test_values[0],
#         'Predicted_Close': predictions[0],
#         'R2': r2,
#         'MSE': mse,
#         'RMSE': rmse,
#         'Success_Rate': success_rate,
#         'Directional_Success': directional_success,
#         'Avg_Error': avg_error,
#         'Normalized_Score': normalized_score,
#         'Best_Model': False
#     }])
#
#     # ----------------------------
#     # 8. Plotting
#     # ----------------------------
#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#
#     # Actual vs Predicted
#     axs[0, 0].plot(range(len(y_test_values)), y_test_values, label='Actual', color='blue', linewidth=2)
#     axs[0, 0].plot(range(len(predictions)), predictions, label='Predicted', color='red', linestyle='--', linewidth=2)
#     axs[0, 0].set_title(f'LSTM Prediction\nUnits={lstm_units}, Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Dropout={dropout}, Layers={num_layers}', fontsize=10)
#     axs[0, 0].legend()
#     axs[0, 0].grid(True)
#     axs[0, 0].text(0.05, 0.95, f"R² = {r2:.3f}", transform=axs[0, 0].transAxes)
#     axs[0, 0].text(0.05, 0.90, f"MSE = {mse:.3f}", transform=axs[0, 0].transAxes)
#     axs[0, 0].text(0.05, 0.85, f"RMSE = {rmse:.3f}", transform=axs[0, 0].transAxes)
#
#     # Scatter plot
#     axs[0, 1].scatter(y_test_values, predictions, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
#     min_val = min(min(y_test_values), min(predictions))
#     max_val = max(max(y_test_values), max(predictions))
#     axs[0, 1].plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--')
#     axs[0, 1].set_title('Predicted vs Actual Prices (LSTM)', fontsize=13)
#     axs[0, 1].grid(True)
#
#     # Loss over epochs
#     axs[1, 0].plot(history.history['loss'], label='Train Loss', color='blue', linestyle='-', linewidth=2)
#     axs[1, 0].plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--', linewidth=2)
#     axs[1, 0].set_title('Model Loss over Epochs', fontsize=13)
#     axs[1, 0].legend()
#     axs[1, 0].grid(True)
#
#     # Placeholder for feature importance
#     axs[1, 1].text(0.5, 0.5, "LSTM does not support feature importance directly.\nUse SHAP or similar techniques.",
#                    ha='center', va='center', fontsize=12, color='gray')
#     axs[1, 1].axis('off')
#
#     plt.tight_layout()
#     main_chart_lstm = fig_to_base64(fig)
#
#     # Clear session to prevent memory issues
#     K.clear_session()
#
#     return main_chart_lstm, summary_lstm_df

#
# def lstm_hyper_tuning_chart(X_train, X_test, y_train, y_test,
#                             lstm_units, epochs, batch_size,
#                             learning_rate, dropout, optimizer, num_layers,
#                             loss_function, activation_function, stock_name):
#     # Convert X_train columns to numeric
#     for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
#         if column in X_train.columns:
#             X_train[column] = pd.to_numeric(X_train[column], errors='coerce')
#         if column in X_test.columns:
#             X_test[column] = pd.to_numeric(X_test[column], errors='coerce')
#
#     # Convert y_train and y_test to numeric
#     y_train = pd.to_numeric(y_train, errors='coerce')
#     y_test = pd.to_numeric(y_test, errors='coerce')
#
#     # Drop NaNs and realign y accordingly
#     X_train = X_train.dropna()
#     y_train = y_train[X_train.index]
#     X_test = X_test.dropna()
#     y_test = y_test[X_test.index]
#
#     # Print types after conversion
#     print("X_train dtype after conversion:", X_train.dtypes)
#     print("y_train dtype after conversion:", y_train.dtypes)
#
#     # Convert to NumPy and reshape
#     X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])).astype(np.float32)
#     X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])).astype(np.float32)
#     y_train = np.array(y_train, dtype=np.float32)
#     # y_test = np.array(y_test, dtype=np.float32)
#     # Keep original index for y_test if it exists
#     if not isinstance(y_test, pd.Series):
#         y_test = pd.Series(y_test, index=X_test.index)
#
#     # Optimizer setup
#     if optimizer == 'adam':
#         opt = Adam(learning_rate=learning_rate)
#     elif optimizer == 'sgd':
#         opt = SGD(learning_rate=learning_rate)
#     else:
#         opt = RMSprop(learning_rate=learning_rate)
#
#     # Loss function
#     loss = MeanSquaredError() if loss_function == 'mse' else MeanAbsoluteError()
#
#     # Build LSTM model
#     model = Sequential()
#     for i in range(num_layers):
#         return_seq = i < num_layers - 1
#         model.add(LSTM(units=lstm_units,
#                        activation=activation_function,
#                        return_sequences=return_seq,
#                        input_shape=(X_train.shape[1], X_train.shape[2]) if i == 0 else None))
#         model.add(Dropout(dropout))
#
#     model.add(Dense(1))
#     model.compile(optimizer=opt, loss=loss)
#
#     # Train model
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     history = model.fit(X_train, y_train,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(X_test, y_test),
#                         callbacks=[early_stop],
#                         verbose=0)
#
#     # Predictions
#     predictions = model.predict(X_test).flatten()
#     # y_test = y_test.flatten()
#
#     # Metrics
#     r2 = r2_score(y_test, predictions)
#     mse = mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(mse)
#
#
#
#     # Make y_test a pandas Series (with original index if needed)
#     # if not isinstance(y_test, pd.Series):
#     #     y_test = pd.Series(y_test)
#     #
#     # y_index = y_test.index
#
#     # Create a one-row summary DataFrame
#     # summary_lstm_df = pd.DataFrame([{
#     #     'Stock': stock_name,
#     #     'Model': "LSTM",
#     #     'Index': y_index[0],  # or use y_index[-1] for the last point
#     #     'Actual_Close': y_test.values[0],
#     #     'Predicted_Close': predictions[0],
#     #     'R2': r2,
#     #     'MSE': mse,
#     #     'RMSE': rmse
#     # }])
#     avg_error, normalized_score, success_rate, directional_success = evaluate_model_metrics(y_test.values, predictions,
#                                                                                             rmse)
#
#     # Summary DataFrame
#     summary_lstm_df = pd.DataFrame([{
#         'Stock': stock_name,
#         'Model': "LSTM",
#         'Index': y_test.index[0].strftime('%Y-%m-%d'),
#         'Actual_Close': y_test.values[0],
#         'Predicted_Close': predictions[0],
#         'R2': r2,
#         'MSE': mse,
#         'RMSE': rmse,
#         'Success_Rate': success_rate,
#         'Directional_Success': directional_success,
#         'Avg_Error': avg_error,
#         'Normalized_Score': normalized_score,
#         'Best_Model': False  # To be updated after all models
#     }])
#     # Plotting
#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#
#     # Actual vs Predicted
#     axs[0, 0].plot(range(len(y_test)), y_test, label='Actual', color='blue', linewidth=2)
#     axs[0, 0].plot(range(len(predictions)), predictions, label='Predicted', color='red', linestyle='--', linewidth=2)
#     axs[0, 0].set_title(
#         f'LSTM Prediction\nUnits={lstm_units}, Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Dropout={dropout}, Layers={num_layers}',
#         fontsize=10)
#     axs[0, 0].legend()
#     axs[0, 0].grid(True)
#     axs[0, 0].text(0.05, 0.95, f"R² = {r2:.3f}", transform=axs[0, 0].transAxes)
#     axs[0, 0].text(0.05, 0.90, f"MSE = {mse:.3f}", transform=axs[0, 0].transAxes)
#     axs[0, 0].text(0.05, 0.85, f"RMSE = {rmse:.3f}", transform=axs[0, 0].transAxes)
#
#     # Scatter Plot
#     axs[0, 1].scatter(y_test, predictions, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
#     min_val = min(min(y_test), min(predictions))
#     max_val = max(max(y_test), max(predictions))
#     axs[0, 1].plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--')
#     axs[0, 1].set_title('Predicted vs Actual Prices (LSTM)', fontsize=13)
#     axs[0, 1].grid(True)
#
#     # Loss Over Epochs
#     axs[1, 0].plot(history.history['loss'], label='Train Loss', color='blue', linestyle='-', linewidth=2)
#     axs[1, 0].plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--', linewidth=2)
#     axs[1, 0].set_title('Model Loss over Epochs', fontsize=13)
#     axs[1, 0].legend()
#     axs[1, 0].grid(True)
#
#     # Feature Importance Placeholder
#     axs[1, 1].text(0.5, 0.5, "LSTM does not support feature importance directly.\nUse SHAP or similar techniques.",
#                    ha='center', va='center', fontsize=12, color='gray')
#     axs[1, 1].axis('off')
#
#     plt.tight_layout()
#     main_chart_lstm = fig_to_base64(fig)
#
#     return main_chart_lstm, summary_lstm_df

# def save_predictions_to_db_hyper_tuning(symbol, linear_predictions, decision_tree_predictions, random_forest_predictions, svm_predictions, lstm_predictions, actual_close_values):
#     # Store predictions in the database, assuming StockPrediction model exists
#     HyperModelPrediction.objects.create(
#         symbol=symbol,
#         linear_predictions=linear_predictions.tolist(),
#         decision_tree_predictions=decision_tree_predictions.tolist(),
#         random_forest_predictions=random_forest_predictions.tolist(),
#         svm_predictions=svm_predictions.tolist(),
#         lstm_predictions=lstm_predictions.tolist(),
#         actual_close_values=actual_close_values.tolist()  # Save the actual close values
#     )
