import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import log_print, mean_absolute_percentage_error


def fit_arma_model(train_clean):
    """Fit ARMA model using auto_arima"""
    log_print("\nFitting ARMA model with auto_arima...")
    arma_model_auto = pm.auto_arima(train_clean,
                                    start_p=1, start_q=1,
                                    test='adf',
                                    max_p=3, max_q=3,
                                    m=1,
                                    d=0,  # Force d=0 for ARMA
                                    seasonal=False,
                                    D=0,
                                    trace=True,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)

    log_print(f"Auto ARMA Best Order: {arma_model_auto.order}")
    return arma_model_auto


def fit_arima_model(train_clean):
    """Fit non-seasonal ARIMA model using auto_arima"""
    log_print("\nFitting Non-Seasonal ARIMA model with auto_arima...")
    arima_model_auto = pm.auto_arima(train_clean,
                                     start_p=1, start_q=1,
                                     test='adf',
                                     max_p=3, max_q=3,
                                     m=1,
                                     d=None,  # Let auto_arima determine d
                                     seasonal=False,
                                     D=0,
                                     trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True)

    log_print(f"Auto Non-Seasonal ARIMA Best Order: {arima_model_auto.order}")
    return arima_model_auto


def fit_sarima_model(train_clean):
    """Fit SARIMA model using auto_arima"""
    log_print("\nFitting SARIMA model with auto_arima...")
    log_print("Note: SARIMA m=12 used. If data is daily with yearly seasonality, m approx 252 might be better.")

    sarima_model_auto = pm.auto_arima(train_clean,
                                      start_p=1, start_q=1,
                                      test='adf',
                                      max_p=3, max_q=3,
                                      m=12,  # frequency of series
                                      d=None,  # let model determine 'd'
                                      seasonal=True,
                                      start_P=0,
                                      D=None,  # let model determine 'D'
                                      trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)

    log_print(f"Auto ARIMA Best Order: {sarima_model_auto.order}, Seasonal Order: {sarima_model_auto.seasonal_order}")
    return sarima_model_auto


def evaluate_arima_model(model_fit, train_clean, train_actual_original, test, test_actual_original,
                         scaler, model_name="ARIMA Model"):
    """Evaluate ARIMA-type model performance"""
    # Generate in-sample predictions for training set
    in_sample = model_fit.predict_in_sample()
    in_sample = pd.Series(in_sample, index=train_clean.index).interpolate(method='linear')

    # Inverse transform in-sample predictions
    in_sample_log_scale = pd.Series(
        scaler.inverse_transform(in_sample.values.reshape(-1, 1)).flatten(),
        index=in_sample.index)
    in_sample_original_scale = np.exp(in_sample_log_scale)

    # Align and clean for training evaluation
    common_idx_train = train_actual_original.index.intersection(in_sample_original_scale.index)
    y_true_train = train_actual_original.loc[common_idx_train].interpolate().fillna(method='ffill').fillna(
        method='bfill')
    y_pred_train = in_sample_original_scale.loc[common_idx_train].interpolate().fillna(method='ffill').fillna(
        method='bfill')

    if y_true_train.isnull().any() or y_pred_train.isnull().any():
        log_print(f"Warning: NaNs detected in {model_name} training evaluation series. Filling with 0.")
        y_true_train = y_true_train.fillna(0)
        y_pred_train = y_pred_train.fillna(0)

    # Calculate metrics for training set
    mae_train = mean_absolute_error(y_true_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    r2_train = r2_score(y_true_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_true_train, y_pred_train)

    # Forecast for test period
    forecast_test_array = model_fit.predict(n_periods=len(test))
    forecast_test = pd.Series(forecast_test_array, index=test.index).interpolate(method='linear')

    # Inverse transform forecast
    forecast_log_scale = pd.Series(
        scaler.inverse_transform(forecast_test.values.reshape(-1, 1)).flatten(),
        index=forecast_test.index)
    forecast_original_scale = np.exp(forecast_log_scale)

    # Align and clean for evaluation
    common_idx = test_actual_original.index.intersection(forecast_original_scale.index)
    y_true = test_actual_original.loc[common_idx].interpolate().fillna(method='ffill').fillna(method='bfill')
    y_pred = forecast_original_scale.loc[common_idx].interpolate().fillna(method='ffill').fillna(method='bfill')

    if y_true.isnull().any() or y_pred.isnull().any():
        log_print(f"Warning: NaNs detected in {model_name} evaluation series. Filling with 0.")
        y_true = y_true.fillna(0)
        y_pred = y_pred.fillna(0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Log results
    log_print(f"\n{model_name} Model Evaluation (Original Scale Metrics):")
    log_print("Training Set Metrics:")
    log_print(f"MAE: {mae_train:.2f}")
    log_print(f"RMSE: {rmse_train:.2f}")
    log_print(f"R²: {r2_train:.4f}")
    log_print(f"MAPE: {mape_train:.2f}%")
    log_print("\nTest Set Metrics:")
    log_print(f"MAE: {mae:.2f}")
    log_print(f"RMSE: {rmse:.2f}")
    log_print(f"R²: {r2:.4f}")
    log_print(f"MAPE: {mape:.2f}%")

    metrics = {
        'train': {'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train, 'mape': mape_train},
        'test': {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
    }

    return forecast_test, metrics


def forecast_future(model_fit, horizon, forecast_dates, alpha=0.05):
    """Generate forecasts and confidence intervals for future periods"""
    forecast_array, ci_array = model_fit.predict(n_periods=horizon, return_conf_int=True, alpha=alpha)
    forecast = pd.Series(forecast_array, index=forecast_dates).interpolate(method='linear')
    lower = pd.Series(ci_array[:, 0], index=forecast_dates).interpolate(method='linear')
    upper = pd.Series(ci_array[:, 1], index=forecast_dates).interpolate(method='linear')

    return forecast, lower, upper