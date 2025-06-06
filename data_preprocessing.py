import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
from utils import log_print


def split_data(ts_log_scale):
    """Split data into training and test sets"""
    train_log = ts_log_scale[:'2016-12-31']
    test_log = ts_log_scale['2017-01-01':'2018-11-30']

    log_print(
        f"\nLog-scale Training set: {train_log.index.min()} to {train_log.index.max()}, {len(train_log)} observations")
    log_print(f"Log-scale Test set: {test_log.index.min()} to {test_log.index.max()}, {len(test_log)} observations")

    return train_log, test_log


def standardize_data(train_log, test_log):
    """Standardize log-transformed data"""
    log_print("\n## Step 3: Standardization of Log-Transformed Data ##")
    scaler_log_standard = StandardScaler()

    # Fit the scaler ONLY on the training data
    scaler_log_standard.fit(train_log.values.reshape(-1, 1))

    # Transform training and test data using the FITTED scaler
    train_scaled = pd.Series(
        scaler_log_standard.transform(train_log.values.reshape(-1, 1)).flatten(),
        index=train_log.index
    )
    test_scaled = pd.Series(
        scaler_log_standard.transform(test_log.values.reshape(-1, 1)).flatten(),
        index=test_log.index
    )

    log_print("Standardization applied (fitted on train, transformed train/test).")
    log_print("\nScaled Training Data Description:")
    log_print(train_scaled.describe())
    log_print("\nScaled Test Data Description:")
    log_print(test_scaled.describe())

    return train_scaled, test_scaled, scaler_log_standard


def difference_series(ts):
    """Apply differencing to the series"""
    ts_diff = ts.diff().dropna()
    # Interpolate NaN values after differencing (first value will be NaN)
    ts_diff = ts_diff.interpolate(method='linear')
    # Additional dropna to ensure clean data for plots and tests
    ts_diff_clean = ts_diff.dropna()

    return ts_diff, ts_diff_clean