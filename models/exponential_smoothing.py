import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import log_print, mean_absolute_percentage_error


def fit_ses_model(train_clean):
    """Fit Simple Exponential Smoothing model"""
    log_print("\nFitting Simple Exponential Smoothing (SES) model...")
    ses_model = SimpleExpSmoothing(train_clean, initialization_method="estimated")
    ses_fit = ses_model.fit(optimized=True)

    log_print("SES Model Parameters:")
    log_print(f"Alpha (smoothing_level): {ses_fit.params.get('smoothing_level', 'N/A')}")

    return ses_fit


def fit_holt_model(train_clean):
    """Fit Holt's Linear Trend model"""
    log_print("\nFitting Holt's Linear Method model...")
    holt_model = Holt(train_clean, exponential=False, initialization_method="estimated")
    holt_fit = holt_model.fit(optimized=True)

    log_print("Holt's Model Parameters:")
    log_print(f"Alpha (smoothing_level): {holt_fit.params.get('smoothing_level', 'N/A')}")
    log_print(f"Beta (smoothing_trend): {holt_fit.params.get('smoothing_trend', 'N/A')}")

    return holt_fit


def fit_holt_winters_model(train_clean):
    """Fit Holt-Winters Triple Exponential Smoothing model"""
    log_print("\nFitting Holt-Winters model...")
    use_boxcox_hw = False
    log_print("Disabling Box-Cox for Holt-Winters as data is log-transformed & standardized.")

    hw_model = ExponentialSmoothing(
        train_clean,
        seasonal_periods=12,
        trend='add',
        seasonal='add',
        use_boxcox=use_boxcox_hw
    )
    hw_fit = hw_model.fit(optimized=True)

    log_print("Holt-Winters Model Parameters:")
    log_print(f"Alpha (level): {hw_fit.params['smoothing_level']}")
    log_print(f"Beta (trend): {hw_fit.params['smoothing_trend']}")
    log_print(f"Gamma (seasonal): {hw_fit.params['smoothing_seasonal']}")

    return hw_fit


def generate_confidence_intervals(model_fit, forecast, forecast_length, alpha=0.05, model_type='hw'):
    """Generate confidence intervals for forecasts"""
    if model_type == 'ses':
        std_err = np.sqrt(model_fit.sse / (len(model_fit.fittedvalues) - 1))
        params = 1
    elif model_type == 'holt':
        std_err = np.sqrt(model_fit.sse / (len(model_fit.fittedvalues) - 2))
        params = 2
    else:  # Holt-Winters
        std_err = np.sqrt(model_fit.sse / (len(model_fit.fittedvalues) - 3))
        params = 3

    forecast_lower = forecast - norm.ppf(1 - alpha / 2) * std_err
    forecast_upper = forecast + norm.ppf(1 - alpha / 2) * std_err

    return forecast_lower.interpolate(method='linear'), forecast_upper.interpolate(method='linear')


def evaluate_model(model_fit, train_clean, train_actual_original, test, test_actual_original,
                   scaler, model_name="Model"):
    """Evaluate model performance on training and test sets"""
    # Generate in-sample predictions for training set
    in_sample = model_fit.fittedvalues
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
    forecast_test = model_fit.forecast(len(test))
    forecast_test = pd.Series(forecast_test, index=test.index).interpolate(method='linear')

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