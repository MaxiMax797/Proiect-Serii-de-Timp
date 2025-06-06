import pandas as pd
import numpy as np
from datetime import datetime
from config import FORECAST_HORIZON, OUTPUT_DIR
from utils import log_print, clean_forecast_dataframe
import os


def inverse_transform_forecast(forecast, scaler):
    """Inverse transform a forecast from standardized log scale to original scale"""
    forecast_log_scale = pd.Series(
        scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten(),
        index=forecast.index
    )
    forecast_original_scale = np.exp(forecast_log_scale)
    return forecast_log_scale, forecast_original_scale


def generate_forecast_dates(horizon=FORECAST_HORIZON):
    """Generate dates for the forecast period"""
    forecast_dates = pd.date_range(start='2019-01-01', periods=horizon, freq='MS')
    return forecast_dates


def compare_models(metrics_dict):
    """Compare models based on metrics"""
    log_print("\nModel Comparison on Test Set:")

    # Create comparison dataframe
    models = list(metrics_dict.keys())
    comparison_data = {
        'Metric': ['MAE', 'RMSE', 'R²', 'MAPE (%)']
    }

    for model in models:
        comparison_data[model] = [
            metrics_dict[model]['test']['mae'],
            metrics_dict[model]['test']['rmse'],
            metrics_dict[model]['test']['r2'],
            metrics_dict[model]['test']['mape']
        ]

    comparison_df = pd.DataFrame(comparison_data)
    log_print(comparison_df)

    # Save comparison table to CSV
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

    # Determine best model based on RMSE
    rmse_values = {model: metrics_dict[model]['test']['rmse'] for model in models}
    best_model = min(rmse_values, key=rmse_values.get)
    log_print(f"\nBest performing model based on RMSE: {best_model} (RMSE: {rmse_values[best_model]:.2f})")

    return rmse_values, best_model


def save_forecasts(forecasts_dict, forecast_dates):
    """Save all forecasts to CSV file"""
    forecast_df = pd.DataFrame({'Date': forecast_dates})
    forecast_df.set_index('Date', inplace=True)

    for model_name, forecast_data in forecasts_dict.items():
        forecast_df[f'{model_name}_Forecast'] = forecast_data['forecast'].values
        forecast_df[f'{model_name}_Lower_CI'] = forecast_data['lower'].values
        forecast_df[f'{model_name}_Upper_CI'] = forecast_data['upper'].values

    # Clean the dataframe
    forecast_df = clean_forecast_dataframe(forecast_df)

    # Save to CSV
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'forecasts_2019_original_scale.csv'))

    return forecast_df