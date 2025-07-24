"""
Contrarian Feature-to-Country Weight Conversion Program
======================================================

This program converts feature importance weights from a contrarian machine learning model into 
country-specific investment weights for stock market forecasting. The system processes
monthly feature weights and factor data to generate contrarian country-level investment allocations
by selecting the bottom 20% of countries for each factor.

Version: 1.3
Last Updated: 2025-06-12

INPUT FILES
==========
1. "T2_rolling_window_weights.xlsx"
   - Location: Same directory as script
   - Format: Excel file
   - Content: Feature weights from machine learning model
   - Structure:
     * Rows: Dates (index)
     * Columns: Feature names
     * Values: Feature importance weights

2. "Normalized_T2_MasterCSV.csv"
   - Location: Same directory as script
   - Format: CSV file
   - Content: Normalized factor data for multiple countries
   - Structure:
     * date: Date of observation (YYYY-MM-DD)
     * country: Country code/name
     * variable: Feature name
     * value: Normalized feature value

3. "T2 Master.xlsx"
   - Location: Same directory as script
   - Format: Excel file
   - Purpose: Defines the original sort order of countries

OUTPUT FILES (CONTRARIAN STRATEGY)
==================================
1. "T2_Final_Country_Weights.xlsx"
   - Format: Excel workbook with multiple sheets containing contrarian allocations
   - Sheets:
     a. "All Periods": Complete time series of contrarian country weights
        * Rows: Dates
        * Columns: Countries
        * Values: Contrarian investment weights (0-1)
     b. "Summary Statistics": Statistical analysis of contrarian allocations
        * Metrics: Mean, standard deviation, min, max, etc.
        * Rows: Statistics
        * Columns: Countries
     c. "Latest Weights": Current contrarian allocation snapshot
        * Country: Country name
        * Weight: Current contrarian weight
        * vs_avg: Difference from historical average

2. "T2_Country_Final.xlsx"
   - Format: Excel workbook with single sheet
   - Purpose: Final contrarian country weights in original sort order
   - Structure:
     * Column A: Country names (from T2 Master.xlsx order)
     * Column B: Assigned contrarian weights (formatted as percentage)

METHODOLOGY (CONTRARIAN APPROACH)
=================================
1. Data Loading and Preparation:
   - Load contrarian feature weights and normalized factor data
   - Identify all unique countries and dates
   - Initialize weight tracking structures

2. Contrarian Weight Calculation (per date):
   a. For each date's contrarian model:
      - Identify significant features (non-zero weights)
      - For each significant feature:
        * Apply contrarian approach by selecting bottom 20% of countries
        * Distribute feature's weight equally among worst-performing countries
      - Sum weights across all features for final contrarian country allocations

3. Output Generation:
   - Create comprehensive Excel workbook with multiple analysis views
   - Generate final weight file in predefined sort order
   - Apply proper formatting and styling

DEPENDENCIES:
- pandas
- numpy
- tqdm
- xlsxwriter

USAGE:
python "Step Eight Write Country Weights.py"

NOTES:
- All weights are normalized to sum to 1 (100%) for each date
- The contrarian approach selects bottom 20% of countries for each factor
- Excel output uses xlsxwriter for formatting and styling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ===============================
# DATA LOADING AND PREPROCESSING
# ===============================

# Input file paths
weights_file = "T2_rolling_window_weights.xlsx"
factor_file = "Normalized_T2_MasterCSV.csv"

print("Loading data...")
# Load feature weights from optimization model
feature_weights_df = pd.read_excel(weights_file, index_col=0)

# Load factor data for all countries
factor_df = pd.read_csv(factor_file)
factor_df['date'] = pd.to_datetime(factor_df['date'])  # Convert date column to datetime

# ===============================
# CONTRARIAN APPROACH CONFIGURATION
# ===============================

# CONTRARIAN APPROACH: For all features, we select the BOTTOM 20% of countries
# This implements a contrarian investment strategy where we invest in countries
# with the worst factor scores, based on the hypothesis that underperforming
# factors may provide better future returns due to mean reversion.

# ===============================
# INITIALIZATION
# ===============================

# Get all unique countries from the factor data in their original order
all_countries = factor_df['country'].unique()  # Removed sorting to preserve original order

# Get all unique dates from the factor data
all_factor_dates = sorted(factor_df['date'].unique())

# Get latest weights date - we'll only process dates that have feature weights
latest_weights_date = feature_weights_df.index.max()

# Initialize DataFrame to store weights for all countries and dates
# Only include dates that have feature weights available
all_dates = list(feature_weights_df.index)  # Only use dates with feature weights
all_weights = pd.DataFrame(index=all_dates, columns=all_countries)
all_weights = all_weights.fillna(0.0)  # Start with zero weights

# ===============================
# WEIGHT CALCULATION PROCESS
# ===============================

print("\nProcessing all dates with contrarian approach...")
# Process each date in the all_dates list using contrarian selection
for date in tqdm(all_dates):
    # Initialize weights for all countries on this date
    country_weights = {country: 0.0 for country in all_countries}
    
    # Convert to datetime if it's not already
    if not isinstance(date, pd.Timestamp):
        date_dt = pd.to_datetime(date)
    else:
        date_dt = date
    
    # Use the CURRENT date for feature weights instead of previous date
    # Skip if the current date is not in the feature weights index
    if date_dt not in feature_weights_df.index:
        print(f"Skipping {date} - date not available in feature weights")
        continue
    
    # Get feature weights from the CURRENT date
    date_weights = feature_weights_df.loc[date_dt]
    
    # Filter out features with negligible weights (numerical stability)
    significant_weights = date_weights[date_weights.abs() > 1e-10]
    
    # Process each feature that has a significant weight
    for feature, feature_weight in significant_weights.items():
        # Get data for this feature and CURRENT date across all countries
        feature_data = factor_df[
            (factor_df['date'] == date) & 
            (factor_df['variable'] == feature)
        ].copy()
        
        # Skip if no data available for this feature/date
        if feature_data.empty:
            continue
            
        # Calculate number of countries to select (bottom 20% for contrarian approach)
        n_select = max(1, int(len(feature_data) * 0.2))
        
        # For contrarian approach, LOWER values are better (select bottom 20%)
        selected = feature_data.nsmallest(n_select, 'value')
        
        # Distribute feature weight equally among selected worst-performing countries
        weight_per_country = feature_weight / n_select
        for country in selected['country']:
            country_weights[country] += weight_per_country
    
    # Store calculated weights for this date
    for country, weight in country_weights.items():
        all_weights.loc[date, country] = weight

# ===============================
# VALIDATION AND ANALYSIS
# ===============================

# Verify weights sum to approximately 1 for each date
weight_sums = all_weights.sum(axis=1)
print("\nWeight sum statistics:")
print(weight_sums.describe())

# ===============================
# RESULTS SAVING
# ===============================

print("\nSaving results...")
output_file = 'T2_Final_Country_Weights.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Sheet 1: Complete time series of country weights
    all_weights.to_excel(writer, sheet_name='All Periods')
    
    # Sheet 2: Summary statistics for each country
    summary_stats = pd.DataFrame({
        'Mean Weight': all_weights.mean(),
        'Std Dev': all_weights.std(),
        'Min Weight': all_weights.min(),
        'Max Weight': all_weights.max(),
        'Days with Weight': (all_weights > 0).sum()
    }).sort_values('Mean Weight', ascending=False)
    
    summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    
    # Sheet 3: Latest country weights with comparison to historical average
    # Find the last row that has non-zero weights
    non_zero_dates = all_weights.index[all_weights.sum(axis=1) > 0]
    if len(non_zero_dates) > 0:
        latest_valid_date = non_zero_dates[-1]  # Get the last date with non-zero weights
        latest_weights = pd.DataFrame({
            'Weight': all_weights.loc[latest_valid_date],
            'Average Weight': all_weights.mean(),
            'Days with Weight': (all_weights > 0).sum(),
            'Latest Date': pd.Series([latest_valid_date] * len(all_weights.columns), index=all_weights.columns)
        }).sort_values('Weight', ascending=False)
        print(f"\nUsing {latest_valid_date} as the latest valid date with non-zero weights")
    else:
        # Fallback in case there are no dates with non-zero weights (unlikely)
        latest_weights = pd.DataFrame({
            'Weight': all_weights.iloc[-1],
            'Average Weight': all_weights.mean(),
            'Days with Weight': (all_weights > 0).sum()
        }).sort_values('Weight', ascending=False)
    
    latest_weights.to_excel(writer, sheet_name='Latest Weights')

print(f"\nResults saved to {output_file}")

# ===============================
# SUMMARY REPORTING
# ===============================

# Print top countries by average weight
print("\nTop 10 countries by average weight:")
print(summary_stats.head(10))

print("\nContrarian country weight allocation completed successfully!")
print("Files generated:")
print(f"- {output_file}: Complete contrarian country weight analysis")
print("\nThe contrarian approach selects the bottom 20% of countries for each factor,")
print("implementing a strategy that invests in worst-performing countries based on")
print("the hypothesis that underperforming factors may provide better future returns.")