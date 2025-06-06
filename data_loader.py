import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import BDay
import os


def log_print(*args, **kwargs):
    """Helper function to log output to both console and file"""
    print(*args, **kwargs)  # Print to console
    # If you have a log file configured elsewhere, you would print there too


def load_data(file_path):
    """Load data from CSV file and prepare it for time series analysis"""
    df = pd.read_csv(file_path)
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Interpolate NaN values in all columns
    for col in df.columns:
        df[col] = df[col].interpolate(method='linear')

    log_print("Data loaded. Shape:", df.shape)
    log_print("First 5 rows:")
    log_print(df.head())

    return df


def prepare_time_series(df):
    """Handle missing business days and ensure all values are positive"""
    # Extract Adj Close column and create time series
    ts_loaded = df['Adj Close'].interpolate(method='linear')

    # Handle missing business days
    bday_idx = pd.date_range(start=ts_loaded.index.min(), end=ts_loaded.index.max(), freq=BDay())
    ts_original_scale = ts_loaded.reindex(bday_idx).fillna(method='ffill').fillna(method='bfill')

    # Ensure all values are positive before log transform
    if (ts_original_scale <= 0).any():
        log_print(f"Warning: {(ts_original_scale <= 0).sum()} non-positive values found. Clamping to 1e-6.")
        ts_original_scale[ts_original_scale <= 0] = 1e-6
    if ts_original_scale.isnull().any():
        log_print("Warning: NaNs still present after fill. Interpolating.")
        ts_original_scale = ts_original_scale.interpolate(method='linear').fillna(1e-6)

    log_print("\nOriginal Scale Data (Business Days, Positive) Description:")
    log_print(ts_original_scale.describe())

    return ts_original_scale


def transform_series(ts_original_scale):
    """Apply log transformation to time series"""
    ts_log_scale = np.log(ts_original_scale)

    # Handle potential -inf from log(very_small_number)
    ts_log_scale = ts_log_scale.replace([np.inf, -np.inf], np.nan).interpolate(method='linear')
    if ts_log_scale.isnull().any():
        log_print("Warning: NaNs present in log-transformed series. Filling with mean.")
        ts_log_scale = ts_log_scale.fillna(ts_log_scale.mean())

    log_print("Log transformation applied. Description:")
    log_print(ts_log_scale.describe())

    return ts_log_scale


def test_stationarity(series, title):
    """Test for stationarity using ADF test"""
    # Ensure no NaNs for ADF test
    series_clean = series.interpolate(method='linear').dropna()
    result = adfuller(series_clean)
    labels = ['ADF Test Statistic', 'p-value', '# of Lags Used', '# of Observations Used']

    log_print(f"\nADF Test for {title}")
    for value, label in zip(result, labels):
        log_print(f'{label} : {value}')

    if result[1] <= 0.05:
        log_print("Strong evidence against the null hypothesis")
        log_print("Reject the null hypothesis")
        log_print("Data has no unit root and is stationary")
    else:
        log_print("Weak evidence against the null hypothesis")
        log_print("Fail to reject the null hypothesis")
        log_print("Data has a unit root and is non-stationary")

    return result[1] <= 0.05  # Return True if stationary