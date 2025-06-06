import numpy as np
import os
from config import OUTPUT_DIR

# File for log output
log_file_path = os.path.join(OUTPUT_DIR, 'analysis_log.txt')
log_file = open(log_file_path, 'w', encoding='utf-8')


def log_print(*args, **kwargs):
    """Print to both console and log file"""
    print(*args, **kwargs)  # Print to console
    print(*args, **kwargs, file=log_file)  # Print to log file


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE with handling for NaN values and zeros"""
    # Convert to numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    # Handle any zero values in y_true to prevent division by zero
    nonzero_mask = y_true != 0
    y_true, y_pred = y_true[nonzero_mask], y_pred[nonzero_mask]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def clean_forecast_dataframe(df):
    """Clean dataframe by handling NaN values aggressively"""
    df = df.interpolate(method='linear')
    df = df.fillna(method='ffill').fillna(method='bfill')

    # If any NaN values remain, replace with column means
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # Final check - replace any remaining NaNs with 0
    return df.fillna(0)


def cleanup():
    """Close log file and perform any cleanup"""
    log_print(f"\nAll outputs have been saved to: {OUTPUT_DIR}")
    log_file.close()