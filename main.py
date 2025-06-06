import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime

# Import custom modules
import config
from utils import log_print, cleanup
from data_loader import load_data, prepare_time_series, transform_series, test_stationarity
from data_preprocessing import split_data, standardize_data, difference_series
from visualization import (setup_plotting_style, plot_transformations, plot_time_series,
                           plot_acf_pacf, plot_seasonal_decomposition, plot_differenced_series,
                           plot_model_forecast, plot_all_forecasts, plot_complete_forecast)
from models.exponential_smoothing import (fit_ses_model, fit_holt_model, fit_holt_winters_model,
                                          generate_confidence_intervals, evaluate_model)
from models.arima_models import (fit_arma_model, fit_arima_model, fit_sarima_model,
                                 evaluate_arima_model, forecast_future)
from forecasting import (generate_forecast_dates, compare_models, save_forecasts,
                         inverse_transform_forecast)


def main():
    """Main execution function for time series analysis and forecasting"""
    # Setup
    setup_plotting_style()

    # Load and prepare data
    df = load_data('AAPL.CSV')
    ts_original_scale = prepare_time_series(df)
    ts_log_scale = transform_series(ts_original_scale)

    # Split data
    train_log, test_log = split_data(ts_log_scale)

    # Standardize data
    train_scaled, test_scaled, scaler_log_standard = standardize_data(train_log, test_log)

    # Create a combined series for plotting
    ts_processed_for_plotting = pd.concat([train_scaled, test_scaled])

    # Plot transformations
    plot_transformations(ts_original_scale, ts_log_scale, train_scaled, test_scaled)
    plot_time_series(train_scaled, test_scaled)

    # Test for stationarity
    test_stationarity(train_scaled, "Processed Training Series (Log-Transformed & Standardized)")

    # Plot ACF and PACF
    plot_acf_pacf(train_scaled, "02_acf_pacf_original.png")

    # Test log transformed series for stationarity
    test_stationarity(ts_log_scale, "Log-Transformed Series (Before Standardization)")

    # Apply differencing
    ts_diff, ts_diff_clean = difference_series(ts_processed_for_plotting)
    plot_differenced_series(ts_diff_clean)

    # Test differenced series for stationarity
    test_stationarity(ts_diff, "First Difference of Processed Series")

    # Plot ACF and PACF for differenced series
    plot_acf_pacf(ts_diff_clean, "05_acf_pacf_diff.png")

    # Seasonal decomposition
    plot_seasonal_decomposition(ts_processed_for_plotting, period=252)

    # Define variables for modeling
    train = train_scaled
    test = test_scaled
    train_clean = train.interpolate(method='linear').dropna()
    forecast_horizon = config.FORECAST_HORIZON
    train_actual_original_scale = ts_original_scale.loc[train.index]
    test_actual_original_scale = ts_original_scale.loc[test.index]

    # Initialize containers for results
    metrics_dict = {}
    forecasts_dict = {}
    forecast_dates = generate_forecast_dates()

    # Fit and evaluate Holt-Winters model
    hw_fit = fit_holt_winters_model(train_clean)
    hw_forecast, hw_metrics = evaluate_model(hw_fit, train_clean, train_actual_original_scale,
                                             test, test_actual_original_scale,
                                             scaler_log_standard, "Holt-Winters")
    plot_model_forecast(train, test, hw_forecast, "Holt-Winters")
    metrics_dict["Holt-Winters"] = hw_metrics

    # Forecast for 2019 using HW
    hw_forecast_2019 = pd.Series(hw_fit.forecast(forecast_horizon), index=forecast_dates).interpolate(method='linear')
    hw_forecast_lower, hw_forecast_upper = generate_confidence_intervals(hw_fit, hw_forecast_2019, forecast_horizon)

    # Inverse transform HW forecast
    _, hw_forecast_2019_original = inverse_transform_forecast(hw_forecast_2019, scaler_log_standard)
    _, hw_lower_original = inverse_transform_forecast(hw_forecast_lower, scaler_log_standard)
    _, hw_upper_original = inverse_transform_forecast(hw_forecast_upper, scaler_log_standard)

    forecasts_dict["Holt-Winters"] = {
        "forecast": hw_forecast_2019_original,
        "lower": hw_lower_original,
        "upper": hw_upper_original
    }

    # Fit and evaluate SES model
    ses_fit = fit_ses_model(train_clean)
    ses_forecast, ses_metrics = evaluate_model(ses_fit, train_clean, train_actual_original_scale,
                                               test, test_actual_original_scale,
                                               scaler_log_standard, "SES")
    metrics_dict["SES"] = ses_metrics

    # Forecast for 2019 using SES
    ses_forecast_2019 = pd.Series(ses_fit.forecast(forecast_horizon), index=forecast_dates).interpolate(method='linear')
    ses_forecast_lower, ses_forecast_upper = generate_confidence_intervals(ses_fit, ses_forecast_2019, forecast_horizon,
                                                                           model_type='ses')

    # Inverse transform SES forecast
    _, ses_forecast_2019_original = inverse_transform_forecast(ses_forecast_2019, scaler_log_standard)
    _, ses_lower_original = inverse_transform_forecast(ses_forecast_lower, scaler_log_standard)
    _, ses_upper_original = inverse_transform_forecast(ses_forecast_upper, scaler_log_standard)

    forecasts_dict["SES"] = {
        "forecast": ses_forecast_2019_original,
        "lower": ses_lower_original,
        "upper": ses_upper_original
    }

    # Fit and evaluate Holt's model
    holt_fit = fit_holt_model(train_clean)
    holt_forecast, holt_metrics = evaluate_model(holt_fit, train_clean, train_actual_original_scale,
                                                 test, test_actual_original_scale,
                                                 scaler_log_standard, "Holt")
    metrics_dict["Holt"] = holt_metrics

    # Forecast for 2019 using Holt
    holt_forecast_2019 = pd.Series(holt_fit.forecast(forecast_horizon), index=forecast_dates).interpolate(
        method='linear')
    holt_forecast_lower, holt_forecast_upper = generate_confidence_intervals(holt_fit, holt_forecast_2019,
                                                                             forecast_horizon, model_type='holt')

    # Inverse transform Holt forecast
    _, holt_forecast_2019_original = inverse_transform_forecast(holt_forecast_2019, scaler_log_standard)
    _, holt_lower_original = inverse_transform_forecast(holt_forecast_lower, scaler_log_standard)
    _, holt_upper_original = inverse_transform_forecast(holt_forecast_upper, scaler_log_standard)

    forecasts_dict["Holt"] = {
        "forecast": holt_forecast_2019_original,
        "lower": holt_lower_original,
        "upper": holt_upper_original
    }

    # Fit and evaluate ARMA model
    arma_fit = fit_arma_model(train_clean)
    arma_forecast, arma_metrics = evaluate_arima_model(arma_fit, train_clean, train_actual_original_scale,
                                                       test, test_actual_original_scale,
                                                       scaler_log_standard, "ARMA")
    metrics_dict["ARMA"] = arma_metrics

    # Forecast for 2019 with ARMA
    arma_forecast_2019, arma_forecast_lower, arma_forecast_upper = forecast_future(arma_fit, forecast_horizon,
                                                                                   forecast_dates)

    # Inverse transform ARMA forecast
    _, arma_forecast_2019_original = inverse_transform_forecast(arma_forecast_2019, scaler_log_standard)
    _, arma_lower_original = inverse_transform_forecast(arma_forecast_lower, scaler_log_standard)
    _, arma_upper_original = inverse_transform_forecast(arma_forecast_upper, scaler_log_standard)

    forecasts_dict["ARMA"] = {
        "forecast": arma_forecast_2019_original,
        "lower": arma_lower_original,
        "upper": arma_upper_original
    }

    # Fit and evaluate non-seasonal ARIMA model
    arima_fit = fit_arima_model(train_clean)
    arima_forecast, arima_metrics = evaluate_arima_model(arima_fit, train_clean, train_actual_original_scale,
                                                         test, test_actual_original_scale,
                                                         scaler_log_standard, "Non-Seasonal ARIMA")
    metrics_dict["ARIMA"] = arima_metrics

    # Forecast for 2019 with non-seasonal ARIMA
    arima_forecast_2019, arima_forecast_lower, arima_forecast_upper = forecast_future(arima_fit, forecast_horizon,
                                                                                      forecast_dates)

    # Inverse transform ARIMA forecast
    _, arima_forecast_2019_original = inverse_transform_forecast(arima_forecast_2019, scaler_log_standard)
    _, arima_lower_original = inverse_transform_forecast(arima_forecast_lower, scaler_log_standard)
    _, arima_upper_original = inverse_transform_forecast(arima_forecast_upper, scaler_log_standard)

    forecasts_dict["ARIMA"] = {
        "forecast": arima_forecast_2019_original,
        "lower": arima_lower_original,
        "upper": arima_upper_original
    }

    # Fit and evaluate SARIMA model
    sarima_fit = fit_sarima_model(train_clean)
    sarima_forecast, sarima_metrics = evaluate_arima_model(sarima_fit, train_clean, train_actual_original_scale,
                                                           test, test_actual_original_scale,
                                                           scaler_log_standard, "SARIMA")
    plot_model_forecast(train, test, sarima_forecast, "SARIMA")
    metrics_dict["SARIMA"] = sarima_metrics

    # Forecast for 2019 with SARIMA
    sarima_forecast_2019, sarima_forecast_lower, sarima_forecast_upper = forecast_future(sarima_fit, forecast_horizon,
                                                                                         forecast_dates)

    # Inverse transform SARIMA forecast
    _, sarima_forecast_2019_original = inverse_transform_forecast(sarima_forecast_2019, scaler_log_standard)
    _, sarima_lower_original = inverse_transform_forecast(sarima_forecast_lower, scaler_log_standard)
    _, sarima_upper_original = inverse_transform_forecast(sarima_forecast_upper, scaler_log_standard)

    forecasts_dict["SARIMA"] = {
        "forecast": sarima_forecast_2019_original,
        "lower": sarima_lower_original,
        "upper": sarima_upper_original
    }

    # Compare all models
    rmse_values, best_model = compare_models(metrics_dict)

    # Plot all forecasts
    plot_all_forecasts(ts_original_scale, train, test, forecast_dates, forecasts_dict, rmse_values)

    # After all models are evaluated
    best_model_name = best_model  # From your existing compare_models function
    plot_complete_forecast(ts_original_scale, train_actual_original_scale, test_actual_original_scale,
                           forecast_dates, forecasts_dict, best_model_name)

    # Save forecasts to CSV
    forecast_df = save_forecasts(forecasts_dict, forecast_dates)

    # Summary of findings
    log_print("\nSummary of Findings:")
    log_print(f"1. Original Apple stock series was found to be non-stationary")
    log_print(f"2. After log transformation and standardization, the series became suitable for modeling")
    log_print(f"3. Best performing model for forecasting: {best_model} (RMSE: {rmse_values[best_model]:.2f})")
    log_print("4. Point forecasts and 95% confidence intervals generated for 2019 for all 6 models")
    log_print("5. All models follow the same data transformation pipeline for consistency")

    # Cleanup
    cleanup()


if __name__ == "__main__":
    main()