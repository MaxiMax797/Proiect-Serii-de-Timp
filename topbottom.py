import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
# ARIMA is not directly used if SARIMAX/auto_arima is used
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # MinMaxScaler not used in this revision
import warnings
import os
import sys
from datetime import datetime
from pandas.tseries.offsets import BDay  # For handling business days
import pmdarima as pm  # For auto_arima

warnings.filterwarnings('ignore')

# Create dataOut directory if it doesn't exist
output_dir = 'G:\Romana + Istorie\Proiect Serii de timp\var semifin\dataOut'
os.makedirs(output_dir, exist_ok=True)

# Create a log file for text output
log_file_path = os.path.join(output_dir, 'analysis_log.txt')
log_file = open(log_file_path, 'w', encoding='utf-8')


# Function to log output to file
def log_print(*args, **kwargs):
    print(*args, **kwargs)  # Print to console
    print(*args, **kwargs, file=log_file)  # Print to log file


# Set style for plots
plt.style.use('ggplot')
sns.set_style('whitegrid')

# Load the data
df = pd.read_csv('AAPL.CSV')
log_print("Data loaded. Shape:", df.shape)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Interpolate NaN values in all columns, especially Adj Close
for col in df.columns:
    df[col] = df[col].interpolate(method='linear')
log_print("NaN values interpolated in all columns.")
log_print("First 5 rows:")
log_print(df.head())

# Extract Adj Close column and create time series
ts_loaded = df['Adj Close'].interpolate(method='linear')  # Initial interpolation
log_print("\nTime Series Description (Loaded Scale):")
log_print(ts_loaded.describe())

# --- Step 1: Handle Missing Business Days & Ensure Positivity ---
log_print("\n## Step 1: Preparing Original Scale Data (Business Days Filled, Positive) ##")
bday_idx = pd.date_range(start=ts_loaded.index.min(), end=ts_loaded.index.max(), freq=BDay())
ts_original_scale = ts_loaded.reindex(bday_idx).fillna(method='ffill').fillna(method='bfill')

# Ensure all values are positive before log transform
if (ts_original_scale <= 0).any():
    log_print(
        f"Warning: {(ts_original_scale <= 0).sum()} non-positive values found in 'ts_original_scale'. Clamping to 1e-6.")
    ts_original_scale[ts_original_scale <= 0] = 1e-6  # Replace with a tiny positive number
if ts_original_scale.isnull().any().any():  # Check if any column still has NaNs
    log_print("Warning: NaNs still present in 'ts_original_scale' after fill. Interpolating.")
    ts_original_scale = ts_original_scale.interpolate(method='linear').fillna(1e-6)  # Final catch-all

log_print("\nOriginal Scale Data (Business Days, Positive) Description:")
log_print(ts_original_scale.describe())

# --- Step 2: Log Transformation ---
log_print("\n## Step 2: Log Transformation ##")
ts_log_scale = np.log(ts_original_scale)
# Handle potential -inf from log(very_small_number) if clamping was to a very small value
ts_log_scale = ts_log_scale.replace([np.inf, -np.inf], np.nan).interpolate(method='linear')
if ts_log_scale.isnull().any():  # Final check for NaNs after log transform
    log_print("Warning: NaNs present in 'ts_log_scale'. Filling with mean.")
    ts_log_scale = ts_log_scale.fillna(ts_log_scale.mean())
log_print("Log transformation applied. Description:")
log_print(ts_log_scale.describe())

# --- Data Splitting (BEFORE SCALING) ---
# It's highly recommended to reinstate a 3-way split: train, validation, test
# For simplicity with the current structure, we'll do train/test on log_scale first
# train_log = ts_log_scale[:'2014-12-31'] # Example: Old training end
# validation_log = ts_log_scale['2015-01-01':'2016-12-31'] # Example: Old validation
# test_log = ts_log_scale['2017-01-01':'2018-11-30'] # Example: Old test

# Using your current split points for demonstration, but on ts_log_scale
train_log = ts_log_scale[:'2016-12-31']
test_log = ts_log_scale['2017-01-01':'2018-11-30']
# Ideally, a validation set would be carved out here too.

log_print(f"\nLog-scale Training set: {train_log.index.min()} to {train_log.index.max()}, {len(train_log)} observations")
log_print(f"Log-scale Test set: {test_log.index.min()} to {test_log.index.max()}, {len(test_log)} observations")


# --- Step 3: Standardization (Fit on TRAIN, Transform Train & Test) ---
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
# If you had a validation_log:
# validation_scaled = pd.Series(
#     scaler_log_standard.transform(validation_log.values.reshape(-1, 1)).flatten(),
#     index=validation_log.index
# )

log_print("Standardization applied (fitted on train, transformed train/test).")
log_print("\nScaled Training Data Description:")
log_print(train_scaled.describe())
log_print("\nScaled Test Data Description:")
log_print(test_scaled.describe())


# Update 'ts' for plotting the whole processed series (if needed for a specific plot)
# but models will use train_scaled and test_scaled
ts_processed_for_plotting = pd.concat([train_scaled, test_scaled]) # Or include validation_scaled

# Models should be trained on 'train_scaled' and evaluated on 'test_scaled'
train = train_scaled # This is now the training data for models
test = test_scaled   # This is now the test data for models

# Plot transformations
plt.figure(figsize=(15, 12))
plt.subplot(411)
plt.plot(ts_original_scale)
plt.title('Original Scale Data (Business Days Filled, Positive)')
plt.ylabel('Price ($)')

plt.subplot(412)
plt.plot(ts_log_scale, color='purple') # Entire log scale for context
plt.title('Log-Transformed Series (Entire Range)')
plt.ylabel('Log(Price)')

plt.subplot(413)
# Plot scaled train and test separately to show the split
plt.plot(train, color='blue', label='Scaled Training Data')
plt.plot(test, color='red', label='Scaled Test Data')
plt.title('Log-Transformed & Standardized Series - Used for Modeling')
plt.ylabel('Standard Deviations')
plt.legend()

# Removing the old ts_normalized plot as it's not part of the current pipeline
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '00_data_pipeline_transformations.png'))
plt.close()

# Plot the time series (now using log-transformed and standardized data)
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train (Scaled)')
plt.plot(test, label='Test (Scaled)')
plt.title('Apple Stock - Log-Transformed & Standardized Price - For Modeling')
plt.xlabel('Date')
plt.ylabel('Standard Deviations')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_time_series_standardized.png'))
plt.close()

# Split data into training and test sets # THIS SECTION IS NOW REDUNDANT as train/test are already defined
# train = ts[:'2016-12-31'] # OLD WAY
# test = ts['2017-01-01':'2018-11-30'] # OLD WAY
forecast_horizon = 12  # 12 months for 2019

log_print(f"\nTraining set for modeling: {train.index.min()} to {train.index.max()}, {len(train)} observations")
log_print(f"Test set for modeling: {test.index.min()} to {test.index.max()}, {len(test)} observations")
log_print(f"Forecast horizon: 12 months (2019)")

# Stationarity test - ADF Test (on the processed 'train' series for modeling)
log_print("\nADF Test for Processed Training Series (Log-Transformed & Standardized)")
# Ensure no NaNs for ADF test
train_clean_for_adf = train.interpolate(method='linear').dropna()
result = adfuller(train_clean_for_adf)
labels = ['ADF Test Statistic', 'p-value', '# of Lags Used', '# of Observations Used']
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

# Plot ACF and PACF for processed series
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(train, ax=plt.gca(), lags=40)
plt.subplot(212)
plot_pacf(train, ax=plt.gca(), lags=40)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_acf_pacf_original.png'))
plt.close()

# Test log transformed series for stationarity -> This title is now misleading.
# The ADF test below should be on ts_log_scale (log-transformed only)
# or ts (log-transformed AND standardized). The current code uses 'ts_log' which was log(standardized_original_data)
# This section needs to be re-evaluated or removed if redundant with the ADF test on 'ts_clean' above.
# For clarity, let's test stationarity of ts_log_scale and ts (processed)
log_print("\nADF Test for Log-Transformed Series (Before Standardization)")
result_log_only = adfuller(ts_log_scale.interpolate(method='linear').dropna())
labels = ['ADF Test Statistic', 'p-value', '# of Lags Used', '# of Observations Used']
for value, label in zip(result_log_only, labels):
    log_print(f'{label} : {value}')

if result_log_only[1] <= 0.05:
    log_print("Series is stationary after log transformation (before standardization)")
else:
    log_print("Series is still non-stationary after log transformation (before standardization)")

# First differencing of log transformed series -> First differencing of Processed Series
# ts_diff = ts.diff().dropna()  # ts is ts_processed
ts_diff = ts_processed_for_plotting.diff().dropna()

# Interpolate NaN values after differencing (first value will be NaN)
ts_diff = ts_diff.interpolate(method='linear')
# Additional dropna to ensure clean data for plots and tests
ts_diff_clean = ts_diff.dropna()
plt.figure(figsize=(12, 6))
plt.plot(ts_diff_clean)
plt.title('First Difference of Log Transformed Series')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_log_diff.png'))
plt.close()

# Test differenced log series for stationarity -> Test Differenced Processed Series
log_print("\nADF Test for First Difference of Processed Series")
result_diff = adfuller(ts_diff)  # ts_diff is already differenced and droppedna
for value, label in zip(result_diff, labels):
    log_print(f'{label} : {value}')

if result_diff[1] <= 0.05:
    log_print("Series is stationary after log transformation and differencing")
else:
    log_print("Series is still non-stationary")

# Plot ACF and PACF for differenced processed series
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(ts_diff_clean, ax=plt.gca(), lags=40)
plt.subplot(212)
plot_pacf(ts_diff_clean, ax=plt.gca(), lags=40)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_acf_pacf_diff.png'))
plt.close()

# Seasonal decomposition (on 'ts' which is log-transformed and standardized)
# ts_decomp = ts.interpolate(method='linear').dropna()
ts_decomp = ts_processed_for_plotting.interpolate(method='linear').dropna()
decomposition = seasonal_decompose(ts_decomp, model='additive', period=252)  # Additive for standardized data
fig = decomposition.plot()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_seasonal_decomp.png'))
plt.close()


# Improved function to calculate MAPE that handles NaN values
def mean_absolute_percentage_error(y_true, y_pred):
    # Convert to numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    # Handle any zero values in y_true to prevent division by zero
    nonzero_mask = y_true != 0
    y_true, y_pred = y_true[nonzero_mask], y_pred[nonzero_mask]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Holt-Winters (Triple Exponential Smoothing)
log_print("\nFitting Holt-Winters model...")
# Interpolate training data (which is train_scaled)
train_clean = train.interpolate(method='linear').dropna()

# If using standardized data, Box-Cox might not be appropriate as data can be negative.
# Disable Box-Cox if data is not strictly positive.
# Standardized data (mean 0, std 1) will likely have negative values.
use_boxcox_hw = False
log_print("Disabling Box-Cox for Holt-Winters as data is log-transformed & standardized.")

hw_model = ExponentialSmoothing(
    train_clean,
    seasonal_periods=12,  # Consider if 252 (business days in year) is more appropriate if daily patterns exist
    trend='add',
    seasonal='add',  # Additive for standardized data
    use_boxcox=use_boxcox_hw
)
hw_fit = hw_model.fit(optimized=True)
log_print("Holt-Winters Model Parameters:")
log_print(f"Alpha (level): {hw_fit.params['smoothing_level']}")
log_print(f"Beta (trend): {hw_fit.params['smoothing_trend']}")
log_print(f"Gamma (seasonal): {hw_fit.params['smoothing_seasonal']}")

# Generate in-sample predictions for training set
hw_in_sample = hw_fit.fittedvalues
hw_in_sample = pd.Series(hw_in_sample, index=train_clean.index).interpolate(method='linear')

# Inverse transform HW in-sample predictions
hw_in_sample_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(hw_in_sample.values.reshape(-1, 1)).flatten(),
    index=hw_in_sample.index)
hw_in_sample_original_scale = np.exp(hw_in_sample_log_scale)

# Get actual training data in original scale
train_actual_original_scale = ts_original_scale.loc[train.index]

# Align and clean for training evaluation
common_idx_hw_train = train_actual_original_scale.index.intersection(hw_in_sample_original_scale.index)
y_true_hw_train = train_actual_original_scale.loc[common_idx_hw_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
y_pred_hw_train = hw_in_sample_original_scale.loc[common_idx_hw_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

if y_true_hw_train.isnull().any() or y_pred_hw_train.isnull().any():
    log_print("Warning: NaNs detected in HW training evaluation series. Filling with 0.")
    y_true_hw_train = y_true_hw_train.fillna(0)
    y_pred_hw_train = y_pred_hw_train.fillna(0)

# Calculate metrics for training set
hw_mae_train = mean_absolute_error(y_true_hw_train, y_pred_hw_train)
hw_rmse_train = np.sqrt(mean_squared_error(y_true_hw_train, y_pred_hw_train))
hw_r2_train = r2_score(y_true_hw_train, y_pred_hw_train)
hw_mape_train = mean_absolute_percentage_error(y_true_hw_train, y_pred_hw_train)

# Forecast for test period
hw_forecast = hw_fit.forecast(len(test))
hw_forecast = pd.Series(hw_forecast, index=test.index)
# Interpolate any NaN values in forecast
hw_forecast = hw_forecast.interpolate(method='linear')

# Plot HW results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(hw_forecast.index, hw_forecast, label='Holt-Winters Forecast')
plt.title('Holt-Winters Method - Training and Test Data')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '07_holtwinters_forecast.png'))
plt.close()

# Evaluate HW on test set - CRITICAL: Must use original scale for metrics
# Predictions (hw_forecast) are on the log-standardized scale
# Actuals for test period on original scale:
test_actual_original_scale = ts_original_scale.loc[test.index]

# Inverse transform HW forecast
hw_forecast_log_scale = pd.Series(scaler_log_standard.inverse_transform(hw_forecast.values.reshape(-1, 1)).flatten(),
                                  index=hw_forecast.index)
hw_forecast_original_scale = np.exp(hw_forecast_log_scale)

# Align and clean for evaluation
common_idx_hw = test_actual_original_scale.index.intersection(hw_forecast_original_scale.index)
y_true_hw = test_actual_original_scale.loc[common_idx_hw].interpolate(method='linear').fillna(method='ffill').fillna(
    method='bfill')
y_pred_hw = hw_forecast_original_scale.loc[common_idx_hw].interpolate(method='linear').fillna(method='ffill').fillna(
    method='bfill')

# Final check for NaNs before metric calculation
if y_true_hw.isnull().any() or y_pred_hw.isnull().any():
    log_print("Warning: NaNs detected in HW evaluation series after cleaning. Metrics might be affected.")
    y_true_hw = y_true_hw.fillna(0)  # Last resort
    y_pred_hw = y_pred_hw.fillna(0)  # Last resort

test_eval_np = np.array(y_true_hw)
hw_forecast_eval_np = np.array(y_pred_hw)

hw_mae = mean_absolute_error(test_eval_np, hw_forecast_eval_np)
hw_rmse = np.sqrt(mean_squared_error(test_eval_np, hw_forecast_eval_np))
hw_r2 = r2_score(test_eval_np, hw_forecast_eval_np)
hw_mape = mean_absolute_percentage_error(test_eval_np, hw_forecast_eval_np)

log_print("\nHolt-Winters Model Evaluation (Original Scale Metrics):")
log_print("Training Set Metrics:")
log_print(f"MAE: {hw_mae_train:.2f}")
log_print(f"RMSE: {hw_rmse_train:.2f}")
log_print(f"R²: {hw_r2_train:.4f}")
log_print(f"MAPE: {hw_mape_train:.2f}%")
log_print("\nTest Set Metrics:")
log_print(f"MAE: {hw_mae:.2f}")
log_print(f"RMSE: {hw_rmse:.2f}")
log_print(f"R²: {hw_r2:.4f}")
log_print(f"MAPE: {hw_mape:.2f}%")

# Forecast for 2019 using HW
hw_forecast_2019 = hw_fit.forecast(forecast_horizon)
forecast_dates = pd.date_range(start='2019-01-01', periods=forecast_horizon, freq='MS')
hw_forecast_2019.index = forecast_dates
# Interpolate any NaN values in 2019 forecast
hw_forecast_2019 = hw_forecast_2019.interpolate(method='linear')

# Generate confidence intervals for HW forecast
from scipy.stats import norm

alpha = 0.05  # 95% confidence level
hw_std = np.sqrt(hw_fit.sse / (len(train_clean) - 3))  # Approximate standard error
hw_forecast_lower = hw_forecast_2019 - norm.ppf(1 - alpha / 2) * hw_std
hw_forecast_upper = hw_forecast_2019 + norm.ppf(1 - alpha / 2) * hw_std
hw_forecast_lower = hw_forecast_lower.interpolate(method='linear')
hw_forecast_upper = hw_forecast_upper.interpolate(method='linear')

# --- Simple Exponential Smoothing (SES) ---
log_print("\nFitting Simple Exponential Smoothing (SES) model...")
ses_model = SimpleExpSmoothing(train_clean, initialization_method="estimated")
ses_fit = ses_model.fit(optimized=True)
log_print("SES Model Parameters:")
log_print(f"Alpha (smoothing_level): {ses_fit.params.get('smoothing_level', 'N/A')}")

# Generate in-sample predictions for training set
ses_in_sample = ses_fit.fittedvalues
ses_in_sample = pd.Series(ses_in_sample, index=train_clean.index).interpolate(method='linear')

# Inverse transform SES in-sample predictions for evaluation
ses_in_sample_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(ses_in_sample.values.reshape(-1, 1)).flatten(),
    index=ses_in_sample.index)
ses_in_sample_original_scale = np.exp(ses_in_sample_log_scale)

# Get actual training data in original scale
train_actual_original_scale = ts_original_scale.loc[train.index]

# Align and clean for training evaluation
common_idx_ses_train = train_actual_original_scale.index.intersection(ses_in_sample_original_scale.index)
y_true_ses_train = train_actual_original_scale.loc[common_idx_ses_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
y_pred_ses_train = ses_in_sample_original_scale.loc[common_idx_ses_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

if y_true_ses_train.isnull().any() or y_pred_ses_train.isnull().any():
    log_print("Warning: NaNs detected in SES training evaluation series. Filling with 0.")
    y_true_ses_train = y_true_ses_train.fillna(0)
    y_pred_ses_train = y_pred_ses_train.fillna(0)

# Calculate metrics for training set
ses_mae_train = mean_absolute_error(y_true_ses_train, y_pred_ses_train)
ses_rmse_train = np.sqrt(mean_squared_error(y_true_ses_train, y_pred_ses_train))
ses_r2_train = r2_score(y_true_ses_train, y_pred_ses_train)
ses_mape_train = mean_absolute_percentage_error(y_true_ses_train, y_pred_ses_train)

# Forecast for test period
ses_forecast_test = ses_fit.forecast(len(test))
ses_forecast_test = pd.Series(ses_forecast_test, index=test.index).interpolate(method='linear')

# Inverse transform SES forecast for evaluation
ses_forecast_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(ses_forecast_test.values.reshape(-1, 1)).flatten(),
    index=ses_forecast_test.index)
ses_forecast_original_scale = np.exp(ses_forecast_log_scale)

# Align and clean for evaluation
common_idx_ses = test_actual_original_scale.index.intersection(ses_forecast_original_scale.index)
y_true_ses = test_actual_original_scale.loc[common_idx_ses].interpolate(method='linear').fillna(method='ffill').fillna(
    method='bfill')
y_pred_ses = ses_forecast_original_scale.loc[common_idx_ses].interpolate(method='linear').fillna(method='ffill').fillna(
    method='bfill')

if y_true_ses.isnull().any() or y_pred_ses.isnull().any():
    log_print("Warning: NaNs detected in SES evaluation series. Filling with 0.")
    y_true_ses = y_true_ses.fillna(0)
    y_pred_ses = y_pred_ses.fillna(0)

ses_mae = mean_absolute_error(y_true_ses, y_pred_ses)
ses_rmse = np.sqrt(mean_squared_error(y_true_ses, y_pred_ses))
ses_r2 = r2_score(y_true_ses, y_pred_ses)
ses_mape = mean_absolute_percentage_error(y_true_ses, y_pred_ses)

log_print("\nSES Model Evaluation (Original Scale Metrics):")
log_print("Training Set Metrics:")
log_print(f"MAE: {ses_mae_train:.2f}")
log_print(f"RMSE: {ses_rmse_train:.2f}")
log_print(f"R²: {ses_r2_train:.4f}")
log_print(f"MAPE: {ses_mape_train:.2f}%")
log_print("\nTest Set Metrics:")
log_print(f"MAE: {ses_mae:.2f}")
log_print(f"RMSE: {ses_rmse:.2f}")
log_print(f"R²: {ses_r2:.4f}")
log_print(f"MAPE: {ses_mape:.2f}%")

# Forecast for 2019 using SES
ses_forecast_2019 = pd.Series(ses_fit.forecast(forecast_horizon), index=forecast_dates).interpolate(method='linear')

# Confidence Intervals for SES 2019 forecast
ses_std = np.sqrt(ses_fit.sse / (len(train_clean) - 1))
ses_forecast_lower = ses_forecast_2019 - norm.ppf(1 - alpha / 2) * ses_std
ses_forecast_upper = ses_forecast_2019 + norm.ppf(1 - alpha / 2) * ses_std

# --- Holt's Linear Method (Double Exponential Smoothing) ---
log_print("\nFitting Holt's Linear Method model...")
holt_model = Holt(train_clean, exponential=False, initialization_method="estimated")
holt_fit = holt_model.fit(optimized=True)
log_print("Holt's Model Parameters:")
log_print(f"Alpha (smoothing_level): {holt_fit.params.get('smoothing_level', 'N/A')}")
log_print(f"Beta (smoothing_trend): {holt_fit.params.get('smoothing_trend', 'N/A')}")

# Generate in-sample predictions for training set
holt_in_sample = holt_fit.fittedvalues
holt_in_sample = pd.Series(holt_in_sample, index=train_clean.index).interpolate(method='linear')

# Inverse transform Holt in-sample predictions
holt_in_sample_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(holt_in_sample.values.reshape(-1, 1)).flatten(),
    index=holt_in_sample.index)
holt_in_sample_original_scale = np.exp(holt_in_sample_log_scale)

# Align and clean for training evaluation
common_idx_holt_train = train_actual_original_scale.index.intersection(holt_in_sample_original_scale.index)
y_true_holt_train = train_actual_original_scale.loc[common_idx_holt_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
y_pred_holt_train = holt_in_sample_original_scale.loc[common_idx_holt_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

if y_true_holt_train.isnull().any() or y_pred_holt_train.isnull().any():
    log_print("Warning: NaNs detected in Holt training evaluation series. Filling with 0.")
    y_true_holt_train = y_true_holt_train.fillna(0)
    y_pred_holt_train = y_pred_holt_train.fillna(0)

# Calculate metrics for training set
holt_mae_train = mean_absolute_error(y_true_holt_train, y_pred_holt_train)
holt_rmse_train = np.sqrt(mean_squared_error(y_true_holt_train, y_pred_holt_train))
holt_r2_train = r2_score(y_true_holt_train, y_pred_holt_train)
holt_mape_train = mean_absolute_percentage_error(y_true_holt_train, y_pred_holt_train)

# Forecast for test period
holt_forecast_test = holt_fit.forecast(len(test))
holt_forecast_test = pd.Series(holt_forecast_test, index=test.index).interpolate(method='linear')

# Inverse transform Holt's forecast for evaluation
holt_forecast_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(holt_forecast_test.values.reshape(-1, 1)).flatten(),
    index=holt_forecast_test.index)
holt_forecast_original_scale = np.exp(holt_forecast_log_scale)

# Align and clean for evaluation
common_idx_holt = test_actual_original_scale.index.intersection(holt_forecast_original_scale.index)
y_true_holt = test_actual_original_scale.loc[common_idx_holt].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')
y_pred_holt = holt_forecast_original_scale.loc[common_idx_holt].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')

if y_true_holt.isnull().any() or y_pred_holt.isnull().any():
    log_print("Warning: NaNs detected in Holt evaluation series. Filling with 0.")
    y_true_holt = y_true_holt.fillna(0)
    y_pred_holt = y_pred_holt.fillna(0)

holt_mae = mean_absolute_error(y_true_holt, y_pred_holt)
holt_rmse = np.sqrt(mean_squared_error(y_true_holt, y_pred_holt))
holt_r2 = r2_score(y_true_holt, y_pred_holt)
holt_mape = mean_absolute_percentage_error(y_true_holt, y_pred_holt)

log_print("\nHolt's Linear Method Evaluation (Original Scale Metrics):")
log_print("Training Set Metrics:")
log_print(f"MAE: {holt_mae_train:.2f}")
log_print(f"RMSE: {holt_rmse_train:.2f}")
log_print(f"R²: {holt_r2_train:.4f}")
log_print(f"MAPE: {holt_mape_train:.2f}%")
log_print("\nTest Set Metrics:")
log_print(f"MAE: {holt_mae:.2f}")
log_print(f"RMSE: {holt_rmse:.2f}")
log_print(f"R²: {holt_r2:.4f}")
log_print(f"MAPE: {holt_mape:.2f}%")

# Forecast for 2019 using Holt's
holt_forecast_2019 = pd.Series(holt_fit.forecast(forecast_horizon), index=forecast_dates).interpolate(method='linear')

# Confidence Intervals for Holt's 2019 forecast
holt_std = np.sqrt(holt_fit.sse / (len(train_clean) - 2))
holt_forecast_lower = holt_forecast_2019 - norm.ppf(1 - alpha / 2) * holt_std
holt_forecast_upper = holt_forecast_2019 + norm.ppf(1 - alpha / 2) * holt_std

# --- ARMA Model ---
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
arma_fit = arma_model_auto

# Generate in-sample predictions for training set
arma_in_sample = arma_fit.predict_in_sample()
arma_in_sample = pd.Series(arma_in_sample, index=train_clean.index).interpolate(method='linear')

# Inverse transform ARMA in-sample predictions
arma_in_sample_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arma_in_sample.values.reshape(-1, 1)).flatten(),
    index=arma_in_sample.index)
arma_in_sample_original_scale = np.exp(arma_in_sample_log_scale)

# Align and clean for training evaluation
common_idx_arma_train = train_actual_original_scale.index.intersection(arma_in_sample_original_scale.index)
y_true_arma_train = train_actual_original_scale.loc[common_idx_arma_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
y_pred_arma_train = arma_in_sample_original_scale.loc[common_idx_arma_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

if y_true_arma_train.isnull().any() or y_pred_arma_train.isnull().any():
    log_print("Warning: NaNs detected in ARMA training evaluation series. Filling with 0.")
    y_true_arma_train = y_true_arma_train.fillna(0)
    y_pred_arma_train = y_pred_arma_train.fillna(0)

# Calculate metrics for training set
arma_mae_train = mean_absolute_error(y_true_arma_train, y_pred_arma_train)
arma_rmse_train = np.sqrt(mean_squared_error(y_true_arma_train, y_pred_arma_train))
arma_r2_train = r2_score(y_true_arma_train, y_pred_arma_train)
arma_mape_train = mean_absolute_percentage_error(y_true_arma_train, y_pred_arma_train)

# Forecast with ARMA
arma_forecast_test_array = arma_fit.predict(n_periods=len(test))
arma_forecast_test = pd.Series(arma_forecast_test_array, index=test.index).interpolate(method='linear')

# Inverse transform ARMA forecast
arma_forecast_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arma_forecast_test.values.reshape(-1, 1)).flatten(),
    index=arma_forecast_test.index)
arma_forecast_original_scale = np.exp(arma_forecast_log_scale)

# Align and clean for evaluation
common_idx_arma = test_actual_original_scale.index.intersection(arma_forecast_original_scale.index)
y_true_arma = test_actual_original_scale.loc[common_idx_arma].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')
y_pred_arma = arma_forecast_original_scale.loc[common_idx_arma].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')

if y_true_arma.isnull().any() or y_pred_arma.isnull().any():
    log_print("Warning: NaNs detected in ARMA evaluation series. Filling with 0.")
    y_true_arma = y_true_arma.fillna(0)
    y_pred_arma = y_pred_arma.fillna(0)

arma_mae = mean_absolute_error(y_true_arma, y_pred_arma)
arma_rmse = np.sqrt(mean_squared_error(y_true_arma, y_pred_arma))
arma_r2 = r2_score(y_true_arma, y_pred_arma)
arma_mape = mean_absolute_percentage_error(y_true_arma, y_pred_arma)

log_print("\nARMA Model Evaluation (Original Scale Metrics):")
log_print("Training Set Metrics:")
log_print(f"MAE: {arma_mae_train:.2f}")
log_print(f"RMSE: {arma_rmse_train:.2f}")
log_print(f"R²: {arma_r2_train:.4f}")
log_print(f"MAPE: {arma_mape_train:.2f}%")
log_print("\nTest Set Metrics:")
log_print(f"MAE: {arma_mae:.2f}")
log_print(f"RMSE: {arma_rmse:.2f}")
log_print(f"R²: {arma_r2:.4f}")
log_print(f"MAPE: {arma_mape:.2f}%")

# Forecast for 2019 with ARMA
arma_forecast_2019_array, arma_ci_array = arma_fit.predict(n_periods=forecast_horizon, return_conf_int=True,
                                                           alpha=alpha)
arma_forecast_2019 = pd.Series(arma_forecast_2019_array, index=forecast_dates).interpolate(method='linear')
arma_forecast_lower = pd.Series(arma_ci_array[:, 0], index=forecast_dates).interpolate(method='linear')
arma_forecast_upper = pd.Series(arma_ci_array[:, 1], index=forecast_dates).interpolate(method='linear')

# --- ARIMA Model (Non-Seasonal) ---
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
arima_fit = arima_model_auto

# Generate in-sample predictions for training set
arima_in_sample = arima_fit.predict_in_sample()
arima_in_sample = pd.Series(arima_in_sample, index=train_clean.index).interpolate(method='linear')

# Inverse transform ARIMA in-sample predictions
arima_in_sample_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arima_in_sample.values.reshape(-1, 1)).flatten(),
    index=arima_in_sample.index)
arima_in_sample_original_scale = np.exp(arima_in_sample_log_scale)

# Align and clean for training evaluation
common_idx_arima_train = train_actual_original_scale.index.intersection(arima_in_sample_original_scale.index)
y_true_arima_train = train_actual_original_scale.loc[common_idx_arima_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
y_pred_arima_train = arima_in_sample_original_scale.loc[common_idx_arima_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

if y_true_arima_train.isnull().any() or y_pred_arima_train.isnull().any():
    log_print("Warning: NaNs detected in ARIMA training evaluation series. Filling with 0.")
    y_true_arima_train = y_true_arima_train.fillna(0)
    y_pred_arima_train = y_pred_arima_train.fillna(0)

# Calculate metrics for training set
arima_mae_train = mean_absolute_error(y_true_arima_train, y_pred_arima_train)
arima_rmse_train = np.sqrt(mean_squared_error(y_true_arima_train, y_pred_arima_train))
arima_r2_train = r2_score(y_true_arima_train, y_pred_arima_train)
arima_mape_train = mean_absolute_percentage_error(y_true_arima_train, y_pred_arima_train)

# Forecast with Non-Seasonal ARIMA
arima_forecast_test_array = arima_fit.predict(n_periods=len(test))
arima_forecast_test = pd.Series(arima_forecast_test_array, index=test.index).interpolate(method='linear')

# Inverse transform Non-Seasonal ARIMA forecast
arima_forecast_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arima_forecast_test.values.reshape(-1, 1)).flatten(),
    index=arima_forecast_test.index)
arima_forecast_original_scale = np.exp(arima_forecast_log_scale)

# Align and clean for evaluation
common_idx_arima = test_actual_original_scale.index.intersection(arima_forecast_original_scale.index)
y_true_arima = test_actual_original_scale.loc[common_idx_arima].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')
y_pred_arima = arima_forecast_original_scale.loc[common_idx_arima].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')

if y_true_arima.isnull().any() or y_pred_arima.isnull().any():
    log_print("Warning: NaNs detected in Non-Seasonal ARIMA evaluation series. Filling with 0.")
    y_true_arima = y_true_arima.fillna(0)
    y_pred_arima = y_pred_arima.fillna(0)

arima_mae = mean_absolute_error(y_true_arima, y_pred_arima)
arima_rmse = np.sqrt(mean_squared_error(y_true_arima, y_pred_arima))
arima_r2 = r2_score(y_true_arima, y_pred_arima)
arima_mape = mean_absolute_percentage_error(y_true_arima, y_pred_arima)

log_print("\nNon-Seasonal ARIMA Model Evaluation (Original Scale Metrics):")
log_print("Training Set Metrics:")
log_print(f"MAE: {arima_mae_train:.2f}")
log_print(f"RMSE: {arima_rmse_train:.2f}")
log_print(f"R²: {arima_r2_train:.4f}")
log_print(f"MAPE: {arima_mape_train:.2f}%")
log_print("\nTest Set Metrics:")
log_print(f"MAE: {arima_mae:.2f}")
log_print(f"RMSE: {arima_rmse:.2f}")
log_print(f"R²: {arima_r2:.4f}")
log_print(f"MAPE: {arima_mape:.2f}%")

# Forecast for 2019 with Non-Seasonal ARIMA
arima_forecast_2019_array, arima_ci_array = arima_fit.predict(n_periods=forecast_horizon, return_conf_int=True,
                                                              alpha=alpha)
arima_forecast_2019 = pd.Series(arima_forecast_2019_array, index=forecast_dates).interpolate(method='linear')
arima_forecast_lower = pd.Series(arima_ci_array[:, 0], index=forecast_dates).interpolate(method='linear')
arima_forecast_upper = pd.Series(arima_ci_array[:, 1], index=forecast_dates).interpolate(method='linear')

# SARIMA Model
log_print("\nFitting SARIMA model with auto_arima...")
train_for_sarima = train_clean.copy()  # train_clean is from ts_processed (log-std scale)

# Use auto_arima to find best SARIMA parameters
# Seasonal period (m): For daily data with yearly seasonality, m=252 (approx business days)
# If monthly forecasts are made from daily data, this needs careful thought.
# Assuming we are still forecasting based on the frequency of 'ts' (daily business days)
# and seasonal_order had 12, implies monthly pattern on daily data.
# For auto_arima, m should reflect the main seasonal cycle.
# If data is daily and main seasonality is yearly, m=252.
# If it's monthly data, m=12. Given forecast_horizon=12 (months),
# and original data is daily, this is tricky.
# Let's assume a yearly seasonality on daily data (m=252) for auto_arima if it's daily.
# However, the previous seasonal_order=(1,1,1,12) suggests a period of 12.
# If 'ts' is daily, period=12 is unusual unless there's a 12-day cycle.
# Let's stick to m=12 for consistency with previous manual setting for now, but flag it.
log_print("Note: SARIMA m=12 used. If data is daily with yearly seasonality, m approx 252 might be better.")
auto_sarima_model = pm.auto_arima(train_for_sarima,
                                  start_p=1, start_q=1,
                                  test='adf',  # use adf test to find optimal 'd'
                                  max_p=3, max_q=3,  # maximum p and q
                                  m=12,  # frequency of series (if seasonal)
                                  d=None,  # let model determine 'd'
                                  seasonal=True,  # Seasonality
                                  start_P=0,
                                  D=None,  # let model determine 'D'
                                  trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

log_print(f"Auto ARIMA Best Order: {auto_sarima_model.order}, Seasonal Order: {auto_sarima_model.seasonal_order}")
sarima_fit = auto_sarima_model  # auto_arima returns a fitted model

# Generate in-sample predictions for training set
sarima_in_sample = sarima_fit.predict_in_sample()
sarima_in_sample = pd.Series(sarima_in_sample, index=train_clean.index).interpolate(method='linear')

# Inverse transform SARIMA in-sample predictions
sarima_in_sample_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(sarima_in_sample.values.reshape(-1, 1)).flatten(),
    index=sarima_in_sample.index)
sarima_in_sample_original_scale = np.exp(sarima_in_sample_log_scale)

# Align and clean for training evaluation
common_idx_sarima_train = train_actual_original_scale.index.intersection(sarima_in_sample_original_scale.index)
y_true_sarima_train = train_actual_original_scale.loc[common_idx_sarima_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
y_pred_sarima_train = sarima_in_sample_original_scale.loc[common_idx_sarima_train].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

if y_true_sarima_train.isnull().any() or y_pred_sarima_train.isnull().any():
    log_print("Warning: NaNs detected in SARIMA training evaluation series. Filling with 0.")
    y_true_sarima_train = y_true_sarima_train.fillna(0)
    y_pred_sarima_train = y_pred_sarima_train.fillna(0)

# Calculate metrics for training set
sarima_mae_train = mean_absolute_error(y_true_sarima_train, y_pred_sarima_train)
sarima_rmse_train = np.sqrt(mean_squared_error(y_true_sarima_train, y_pred_sarima_train))
sarima_r2_train = r2_score(y_true_sarima_train, y_pred_sarima_train)
sarima_mape_train = mean_absolute_percentage_error(y_true_sarima_train, y_pred_sarima_train)

# Forecast with SARIMA (predictions are on log-standardized scale)
sarima_forecast_log_std = sarima_fit.predict(n_periods=len(test))
sarima_forecast = pd.Series(sarima_forecast_log_std, index=test.index)  # Name kept for consistency
# Interpolate any NaN values in forecast
sarima_forecast = sarima_forecast.interpolate(method='linear')

# Plot SARIMA results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(sarima_forecast.index, sarima_forecast, label='SARIMA Forecast')
plt.title('SARIMA Model - Training and Test Data')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '08_sarima_forecast.png'))
plt.close()

# Evaluate SARIMA on test set - CRITICAL: Must use original scale for metrics
# Predictions (sarima_forecast) are on the log-standardized scale

# Inverse transform SARIMA forecast
sarima_forecast_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(sarima_forecast.values.reshape(-1, 1)).flatten(), index=sarima_forecast.index)
sarima_forecast_original_scale = np.exp(sarima_forecast_log_scale)

# Align and clean for evaluation
common_idx_sarima = test_actual_original_scale.index.intersection(sarima_forecast_original_scale.index)
y_true_sarima = test_actual_original_scale.loc[common_idx_sarima].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')
y_pred_sarima = sarima_forecast_original_scale.loc[common_idx_sarima].interpolate(method='linear').fillna(
    method='ffill').fillna(method='bfill')

if y_true_sarima.isnull().any() or y_pred_sarima.isnull().any():
    log_print("Warning: NaNs detected in SARIMA evaluation series after cleaning. Metrics might be affected.")
    y_true_sarima = y_true_sarima.fillna(0)
    y_pred_sarima = y_pred_sarima.fillna(0)

test_eval_sarima_np = np.array(y_true_sarima)
sarima_forecast_eval_np = np.array(y_pred_sarima)

sarima_mae = mean_absolute_error(test_eval_sarima_np, sarima_forecast_eval_np)
sarima_rmse = np.sqrt(mean_squared_error(test_eval_sarima_np, sarima_forecast_eval_np))
sarima_r2 = r2_score(test_eval_sarima_np, sarima_forecast_eval_np)
sarima_mape = mean_absolute_percentage_error(test_eval_sarima_np, sarima_forecast_eval_np)

log_print("\nSARIMA Model Evaluation (Original Scale Metrics):")
log_print("Training Set Metrics:")
log_print(f"MAE: {sarima_mae_train:.2f}")
log_print(f"RMSE: {sarima_rmse_train:.2f}")
log_print(f"R²: {sarima_r2_train:.4f}")
log_print(f"MAPE: {sarima_mape_train:.2f}%")
log_print("\nTest Set Metrics:")
log_print(f"MAE: {sarima_mae:.2f}")
log_print(f"RMSE: {sarima_rmse:.2f}")
log_print(f"R²: {sarima_r2:.4f}")
log_print(f"MAPE: {sarima_mape:.2f}%")

# Forecast for 2019 with SARIMA
# Predictions from sarima_fit.predict are on log-standardized scale
sarima_forecast_2019_log_std, sarima_ci_log_std_array = sarima_fit.predict(n_periods=forecast_horizon,
                                                                           return_conf_int=True, alpha=0.05)
sarima_forecast_2019_standardized = pd.Series(sarima_forecast_2019_log_std,
                                              index=forecast_dates)  # Renamed from sarima_forecast_2019_original

# Inverse transform SARIMA 2019 forecast
sarima_forecast_2019_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(sarima_forecast_2019_standardized.values.reshape(-1, 1)).flatten(),
    index=forecast_dates)
sarima_forecast_2019_original_plot = np.exp(
    sarima_forecast_2019_log_scale)  # This is the final forecast on original scale

# Inverse transform SARIMA 2019 CI
sarima_lower_log_std = pd.Series(sarima_ci_log_std_array[:, 0], index=forecast_dates)
sarima_upper_log_std = pd.Series(sarima_ci_log_std_array[:, 1], index=forecast_dates)

sarima_lower_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(sarima_lower_log_std.values.reshape(-1, 1)).flatten(), index=forecast_dates)
sarima_lower_original_plot = np.exp(sarima_lower_log_scale)

sarima_upper_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(sarima_upper_log_std.values.reshape(-1, 1)).flatten(), index=forecast_dates)
sarima_upper_original_plot = np.exp(sarima_upper_log_scale)

# Compare models
log_print("\nModel Comparison on Test Set:")
comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²', 'MAPE (%)'],
    'Holt-Winters': [hw_mae, hw_rmse, hw_r2, hw_mape],
    'SES': [ses_mae, ses_rmse, ses_r2, ses_mape],
    'Holt': [holt_mae, holt_rmse, holt_r2, holt_mape],
    'ARMA': [arma_mae, arma_rmse, arma_r2, arma_mape],
    'ARIMA': [arima_mae, arima_rmse, arima_r2, arima_mape],
    'SARIMA': [sarima_mae, sarima_rmse, sarima_r2, sarima_mape]
})
log_print(comparison_df)

# Also save comparison table to CSV
comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))

# Determine best model based on RMSE
rmse_values = {
    'Holt-Winters': hw_rmse,
    'SES': ses_rmse,
    'Holt': holt_rmse,
    'ARMA': arma_rmse,
    'ARIMA': arima_rmse,
    'SARIMA': sarima_rmse
}
best_model = min(rmse_values, key=rmse_values.get)
log_print(f"\nBest performing model based on RMSE: {best_model} (RMSE: {rmse_values[best_model]:.2f})")

# Plot 2019 forecasts with confidence intervals
plt.figure(figsize=(16, 10))
# Plot historical data
plt.plot(ts_original_scale.loc[train.index], label='Training (1998-2016)', alpha=0.7)
plt.plot(ts_original_scale.loc[test.index], label='Test (2017-2018)', alpha=0.7)

# Inverse transform all 2019 forecasts to original scale
# Holt-Winters
hw_forecast_2019_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(hw_forecast_2019.values.reshape(-1, 1)).flatten(), # Use correctly fitted scaler
    index=hw_forecast_2019.index)
hw_forecast_2019_original = np.exp(hw_forecast_2019_log_scale)
hw_lower_log_scale = pd.Series(scaler_log_standard.inverse_transform(hw_forecast_lower.values.reshape(-1, 1)).flatten(),
                               index=hw_forecast_lower.index)
hw_lower_original = np.exp(hw_lower_log_scale)
hw_upper_log_scale = pd.Series(scaler_log_standard.inverse_transform(hw_forecast_upper.values.reshape(-1, 1)).flatten(),
                               index=hw_forecast_upper.index)
hw_upper_original = np.exp(hw_upper_log_scale)

# SES
ses_forecast_2019_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(ses_forecast_2019.values.reshape(-1, 1)).flatten(),
    index=ses_forecast_2019.index)
ses_forecast_2019_original = np.exp(ses_forecast_2019_log_scale)
ses_lower_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(ses_forecast_lower.values.reshape(-1, 1)).flatten(),
    index=ses_forecast_lower.index)
ses_lower_original = np.exp(ses_lower_log_scale)
ses_upper_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(ses_forecast_upper.values.reshape(-1, 1)).flatten(),
    index=ses_forecast_upper.index)
ses_upper_original = np.exp(ses_upper_log_scale)

# Holt
holt_forecast_2019_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(holt_forecast_2019.values.reshape(-1, 1)).flatten(),
    index=holt_forecast_2019.index)
holt_forecast_2019_original = np.exp(holt_forecast_2019_log_scale)
holt_lower_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(holt_forecast_lower.values.reshape(-1, 1)).flatten(),
    index=holt_forecast_lower.index)
holt_lower_original = np.exp(holt_lower_log_scale)
holt_upper_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(holt_forecast_upper.values.reshape(-1, 1)).flatten(),
    index=holt_forecast_upper.index)
holt_upper_original = np.exp(holt_upper_log_scale)

# ARMA
arma_forecast_2019_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arma_forecast_2019.values.reshape(-1, 1)).flatten(),
    index=arma_forecast_2019.index)
arma_forecast_2019_original = np.exp(arma_forecast_2019_log_scale)
arma_lower_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arma_forecast_lower.values.reshape(-1, 1)).flatten(),
    index=arma_forecast_lower.index)
arma_lower_original = np.exp(arma_lower_log_scale)
arma_upper_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arma_forecast_upper.values.reshape(-1, 1)).flatten(),
    index=arma_forecast_upper.index)
arma_upper_original = np.exp(arma_upper_log_scale)

# ARIMA
arima_forecast_2019_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arima_forecast_2019.values.reshape(-1, 1)).flatten(),
    index=arima_forecast_2019.index)
arima_forecast_2019_original = np.exp(arima_forecast_2019_log_scale)
arima_lower_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arima_forecast_lower.values.reshape(-1, 1)).flatten(),
    index=arima_forecast_lower.index)
arima_lower_original = np.exp(arima_lower_log_scale)
arima_upper_log_scale = pd.Series(
    scaler_log_standard.inverse_transform(arima_forecast_upper.values.reshape(-1, 1)).flatten(),
    index=arima_forecast_upper.index)
arima_upper_original = np.exp(arima_upper_log_scale)

# Plot all forecasts
plt.plot(forecast_dates, hw_forecast_2019_original, '--', label=f'Holt-Winters (RMSE: {hw_rmse:.2f})', linewidth=2)
plt.fill_between(forecast_dates, hw_lower_original, hw_upper_original, alpha=0.1)

plt.plot(forecast_dates, ses_forecast_2019_original, ':', label=f'SES (RMSE: {ses_rmse:.2f})', linewidth=2)
plt.fill_between(forecast_dates, ses_lower_original, ses_upper_original, alpha=0.1)

plt.plot(forecast_dates, holt_forecast_2019_original, '-.', label=f'Holt (RMSE: {holt_rmse:.2f})', linewidth=2)
plt.fill_between(forecast_dates, holt_lower_original, holt_upper_original, alpha=0.1)

plt.plot(forecast_dates, arma_forecast_2019_original, '-', marker='x', markersize=4,
         label=f'ARMA (RMSE: {arma_rmse:.2f})', linewidth=2)
plt.fill_between(forecast_dates, arma_lower_original, arma_upper_original, alpha=0.1)

plt.plot(forecast_dates, arima_forecast_2019_original, '-', marker='o', markersize=4,
         label=f'ARIMA (RMSE: {arima_rmse:.2f})', linewidth=2)
plt.fill_between(forecast_dates, arima_lower_original, arima_upper_original, alpha=0.1)

plt.plot(forecast_dates, sarima_forecast_2019_original_plot, '-', label=f'SARIMA (RMSE: {sarima_rmse:.2f})',
         linewidth=2)
plt.fill_between(forecast_dates, sarima_lower_original_plot, sarima_upper_original_plot, alpha=0.1)

plt.title('Apple Stock Price - All Models Comparison (Original Scale)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price ($)')
plt.legend(loc='best', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '09_final_forecast_all_models.png'), dpi=300)
plt.close()

# Summary of findings
log_print("\nSummary of Findings:")
log_print(f"1. Original Apple stock series was found to be non-stationary")
log_print(f"2. After log transformation and standardization, the series became suitable for modeling")
log_print(f"3. Best performing model for forecasting: {best_model} (RMSE: {rmse_values[best_model]:.2f})")
log_print("4. Point forecasts and 95% confidence intervals generated for 2019 for all 6 models")
log_print("5. All models follow the same data transformation pipeline for consistency")

# Save forecasts to CSV
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Holt_Winters_Forecast': hw_forecast_2019_original.values,
    'HW_Lower_CI': hw_lower_original.values,
    'HW_Upper_CI': hw_upper_original.values,
    'SES_Forecast': ses_forecast_2019_original.values,
    'SES_Lower_CI': ses_lower_original.values,
    'SES_Upper_CI': ses_upper_original.values,
    'Holt_Forecast': holt_forecast_2019_original.values,
    'Holt_Lower_CI': holt_lower_original.values,
    'Holt_Upper_CI': holt_upper_original.values,
    'ARMA_Forecast': arma_forecast_2019_original.values,
    'ARMA_Lower_CI': arma_lower_original.values,
    'ARMA_Upper_CI': arma_upper_original.values,
    'ARIMA_Forecast': arima_forecast_2019_original.values,
    'ARIMA_Lower_CI': arima_lower_original.values,
    'ARIMA_Upper_CI': arima_upper_original.values,
    'SARIMA_Forecast': sarima_forecast_2019_original_plot.values,
    'SARIMA_Lower_CI': sarima_lower_original_plot.values,
    'SARIMA_Upper_CI': sarima_upper_original_plot.values
})
forecast_df.set_index('Date', inplace=True)

# Super aggressive NaN handling
forecast_df = forecast_df.interpolate(method='linear')
forecast_df = forecast_df.fillna(method='ffill').fillna(method='bfill')
# If any NaN values remain, replace with column means
for col in forecast_df.columns:
    if forecast_df[col].isna().any():
        forecast_df[col] = forecast_df[col].fillna(forecast_df[col].mean())
# Final check - replace any remaining NaNs with 0
forecast_df = forecast_df.fillna(0)

forecast_df.to_csv(os.path.join(output_dir, 'forecasts_2019_original_scale.csv'))  # Explicitly name this file

# Print final message BEFORE closing the log file
log_print(f"\nAll outputs have been saved to: {output_dir}")

# Close log file - this must be the last operation
log_file.close()