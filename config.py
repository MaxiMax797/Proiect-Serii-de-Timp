import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration settings
OUTPUT_DIR = 'G:/Romana + Istorie/Proiect Serii de timp/var semifin/dataOut'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
FORECAST_HORIZON = 12  # 12 months for 2019
CONFIDENCE_LEVEL = 0.05  # 95% confidence interval

# Data splitting dates
TRAIN_END_DATE = '2016-12-31'
TEST_START_DATE = '2017-01-01'
TEST_END_DATE = '2018-11-30'

# Seasonal periods
BUSINESS_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

# Plot settings
FIGURE_SIZE_LARGE = (15, 12)
FIGURE_SIZE_MEDIUM = (12, 6)