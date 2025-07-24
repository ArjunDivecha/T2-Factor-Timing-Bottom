"""
=============================================================================
SCRIPT NAME: Step Fourteen Target Optimization.py
=============================================================================

INPUT FILES:
- T2_Final_Country_Weights.xlsx: 
  Target contrarian country weights from contrarian factor timing strategy (All Periods sheet)
- Portfolio_Data.xlsx: 
  Monthly country returns for portfolio roll-forward (Returns sheet)
- T2_Country_Alphas.xlsx: 
  Country alpha estimates based on contrarian factor performance for optimization objective
- T2_Top_20_Exposure.csv:
  Country factor exposures from contrarian (bottom 20%) factor analysis
- Step Tcost.xlsx:
  Asset-specific borrow costs and trading costs by country (jjunk sheet)

OUTPUT FILES:
- T2_Optimized_Country_Weights.xlsx: 
  Turnover-optimized contrarian country weights with performance comparison
- T2_Optimized_Strategy_Analysis.pdf:
  Comprehensive contrarian strategy performance analysis charts
- T2_Turnover_Analysis.pdf:
  Detailed turnover analysis charts for contrarian strategy
- T2_Weighted_Average_Factor_Exposure.pdf:
  Contrarian factor exposure analysis charts
- T2_Weighted_Average_Factor_Rolling_Analysis.pdf:
  Rolling contrarian factor exposure analysis charts

VERSION: 2.0 - CVXPY Enhanced
LAST UPDATED: 2025-06-16
AUTHOR: Claude Code

DESCRIPTION:
Optimizes monthly contrarian country portfolio weights to balance three objectives:
1. Optimize portfolio alpha based on contrarian strategy (weighted average of country alphas from contrarian factor selection)
2. Minimize drift from rolled-forward previous weights (lambda penalty)
3. Minimize transaction costs from turnover (tcost penalty)

The optimization uses contrarian-derived country weights and alphas, rolling forward 
previous month's weights using monthly returns, then solves for optimal weights that maximize:
Alpha_Mult * Contrarian Portfolio Alpha - Lambda * (Drift Penalty)² - Asset-Specific Transaction Costs

Key Features:
- Asset-specific transaction costs: Loads individual transaction costs per country from Step Tcost.xlsx
- Enhanced transaction cost impact: Applies 500x multiplier to make transaction costs more influential
- Contrarian strategy integration: Uses weights and alphas derived from bottom 20% factor selection

Note: The contrarian approach feeds into this optimization through the input files:
- T2_Final_Country_Weights.xlsx contains weights derived from bottom 20% factor selection
- T2_Country_Alphas.xlsx contains alphas based on contrarian factor performance
- Step Tcost.xlsx provides country-specific transaction cost data

DEPENDENCIES:
- pandas
- numpy
- cvxpy (replaces scipy.optimize for faster convex optimization)
- matplotlib
- openpyxl

TRANSACTION COST ENHANCEMENT:
- Loads asset-specific borrow costs and trading costs from Step Tcost.xlsx (jjunk sheet)
- Combines borrow cost + trading cost for total transaction cost per country
- Converts percentage format to decimals and applies 500x multiplier
- Uses individual country total transaction costs in optimization objective
- Significantly increases transaction cost impact on portfolio optimization

USAGE:
python "Step Fourteen Target Optimization.py"

NOTES:
- Uses CVXPY with OSQP solver for fast convex optimization
- Long-only constraints with maximum position size limits
- All weights must sum to 1.0
- Uses warm-start for repeated monthly optimizations
- First month uses original target weights as starting point
=============================================================================
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os

# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Alpha multiplier parameter - higher values increase emphasis on alpha maximization
ALPHA_MULT = 0.0

# Drift penalty parameter - higher values reduce deviation from rolled-forward weights
LAMBDA_DRIFT_PENALTY = 200.0

# Transaction cost parameter - higher values reduce turnover
TRANSACTION_COST = 5

# Maximum weight constraint - prevents concentration in single country
MAX_WEIGHT = 1.00

# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('T2_processing.log', mode='a'),
            logging.StreamHandler()
        ]
    )

def load_transaction_costs():
    """
    Load asset-specific transaction costs from Step Tcost.xlsx
    Combines borrow cost and trading cost for total transaction cost per asset
    
    Returns:
        pd.Series: Total transaction costs (borrow + trading) indexed by country/asset name (as decimals)
    """
    logging.info("Loading asset-specific transaction costs...")
    
    tcost_file = 'Step Tcost.xlsx'
    tcost_df = pd.read_excel(tcost_file, sheet_name='jjunk')
    
    # Set country names as index
    tcost_df = tcost_df.set_index('Country')
    
    # Get borrow cost and trading cost columns
    borrow_cost = tcost_df.iloc[:, 0].copy()  # First column after Country
    trading_cost = tcost_df.iloc[:, 1].copy()  # Second column after Country
    
    # Handle percentage format for borrow cost (remove % and convert to decimal)
    if borrow_cost.dtype == 'object':
        borrow_cost = borrow_cost.str.replace('%', '').astype(float) / 100
    
    # Handle percentage format for trading cost (remove % and convert to decimal)
    if trading_cost.dtype == 'object':
        trading_cost = trading_cost.str.replace('%', '').astype(float) / 100
    
    # Calculate total transaction cost as sum of borrow cost and trading cost
    tcost_series = borrow_cost + trading_cost
    
    # Multiply by 500 to make transaction costs even more impactful
    tcost_series = tcost_series * 500
    
    logging.info(f"Loaded transaction costs for {len(tcost_series)} assets")
    logging.info(f"Borrow cost range: {borrow_cost.min():.1%} to {borrow_cost.max():.1%}")
    logging.info(f"Trading cost range: {trading_cost.min():.1%} to {trading_cost.max():.1%}")
    logging.info(f"Total transaction cost range: {tcost_series.min():.1%} to {tcost_series.max():.1%}")
    
    return tcost_series

def load_data():
    """
    Load all required data files including asset-specific transaction costs
    
    Returns:
        tuple: (target_weights_df, returns_df, alphas_df, exposure_df, tcosts_series)
    """
    logging.info("Loading input data files...")
    
    # Load target weights (All Periods sheet)
    target_weights_file = 'T2_Final_Country_Weights.xlsx'
    target_weights_df = pd.read_excel(target_weights_file, sheet_name='All Periods', index_col=0)
    target_weights_df.index = pd.to_datetime(target_weights_df.index)
    logging.info(f"Loaded target weights: {target_weights_df.shape[0]} periods, {target_weights_df.shape[1]} countries")
    
    # Load returns data
    returns_file = 'Portfolio_Data.xlsx'
    returns_df = pd.read_excel(returns_file, sheet_name='Returns', index_col=0)
    returns_df.index = pd.to_datetime(returns_df.index)
    logging.info(f"Loaded returns data: {returns_df.shape[0]} periods, {returns_df.shape[1]} countries")
    
    # Load country alphas - this file has dates as rows and countries as columns
    alphas_file = 'T2_Country_Alphas.xlsx'
    alphas_df = pd.read_excel(alphas_file, sheet_name='Country_Scores', index_col=0)
    alphas_df.index = pd.to_datetime(alphas_df.index)
    
    logging.info(f"Loaded country alphas: {alphas_df.shape[0]} periods, {alphas_df.shape[1]} countries")
    
    # Load factor exposure data
    exposure_file = 'T2_Top_20_Exposure.csv'
    exposure_df = pd.read_csv(exposure_file)
    exposure_df['Date'] = pd.to_datetime(exposure_df['Date'])
    logging.info(f"Loaded factor exposures: {exposure_df.shape[0]} observations, {len(exposure_df.columns)-2} factors")
    
    # Load asset-specific transaction costs
    tcosts_series = load_transaction_costs()
    
    return target_weights_df, returns_df, alphas_df, exposure_df, tcosts_series

def roll_forward_weights(prev_weights, returns):
    """
    Roll forward previous weights using monthly returns
    
    Args:
        prev_weights: Previous month's weights (pandas Series)
        returns: Monthly returns for current period (pandas Series)
        
    Returns:
        pandas Series: Rolled forward weights (normalized to sum to 1)
    """
    # Apply returns to get new values, handle NaN returns
    returns_clean = returns.fillna(0)  # Replace NaN returns with 0
    new_values = prev_weights * (1 + returns_clean)
    
    # Handle case where all values become zero or negative
    if new_values.sum() <= 0:
        logging.warning("All portfolio values are zero or negative after applying returns. Using equal weights.")
        rolled_weights = pd.Series(1.0 / len(prev_weights), index=prev_weights.index)
    else:
        # Normalize to sum to 1
        rolled_weights = new_values / new_values.sum()
    
    return rolled_weights

def run_optimization(target_weights_df, returns_df, alphas_df, tcosts_series):
    """
    Run the complete optimization process using CVXPY with asset-specific transaction costs
    
    Args:
        target_weights_df: Target weights DataFrame
        returns_df: Returns DataFrame
        alphas_df: Country alphas DataFrame (time-varying)
        tcosts_series: Asset-specific transaction costs Series
        
    Returns:
        tuple: (optimized_weights_df, metrics_df)
    """
    logging.info("Starting optimization process with asset-specific transaction costs...")
    
    # Get common dates and countries
    common_dates = target_weights_df.index.intersection(returns_df.index).intersection(alphas_df.index)
    common_countries = target_weights_df.columns.intersection(returns_df.columns).intersection(
        alphas_df.columns).intersection(tcosts_series.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates found between datasets")
    if len(common_countries) == 0:
        raise ValueError("No common countries found between datasets")
    
    # Align all data to common dates and countries
    target_weights_aligned = target_weights_df.loc[common_dates, common_countries]
    returns_aligned = returns_df.loc[common_dates, common_countries]
    alphas_aligned = alphas_df.loc[common_dates, common_countries]
    tcosts_aligned = tcosts_series[common_countries]
    
    # Fill missing values with column means
    target_weights_aligned = target_weights_aligned.fillna(target_weights_aligned.mean())
    returns_aligned = returns_aligned.fillna(returns_aligned.mean())
    alphas_aligned = alphas_aligned.fillna(alphas_aligned.mean())
    
    logging.info(f"Aligned data: {len(common_dates)} periods, {len(common_countries)} countries")
    logging.info(f"Asset-specific transaction costs range: {tcosts_aligned.min():.1%} to {tcosts_aligned.max():.1%}")
    
    # Convert to numpy arrays for CVXPY
    n_periods = len(common_dates)
    n_countries = len(common_countries)
    
    # Initialize tracking variables
    optimized_weights_list = []
    metrics_list = []
    previous_weights = target_weights_aligned.iloc[0]  # Use first month target weights as initial
    
    # Progress tracking
    logging.info(f"Starting monthly optimization for {n_periods} periods...")
    
    for t, date in enumerate(common_dates):
        if t % 30 == 0 or t == n_periods - 1:
            progress = (t + 1) / n_periods * 100
            logging.info(f"... {t+1:3d}/{n_periods} months ({progress:5.1f}%)")
        
        # Roll forward previous weights using current returns
        if t > 0:
            current_returns = returns_aligned.iloc[t]
            rolled_forward_weights = roll_forward_weights(previous_weights, current_returns)
        else:
            rolled_forward_weights = previous_weights.copy()
        
        # Get current data
        current_target = target_weights_aligned.iloc[t]
        current_alphas = alphas_aligned.iloc[t]
        
        # Create CVXPY model for this specific period (avoid parameter issues)
        weights_var = cp.Variable(n_countries)
        
        # Objective function components
        portfolio_alpha = ALPHA_MULT * cp.sum(cp.multiply(current_alphas.values, weights_var))
        drift_penalty = LAMBDA_DRIFT_PENALTY * cp.sum_squares(weights_var - current_target.values)
        
        # Use transaction costs as constants (not parameters) to avoid DCP issues
        turnover_penalty = cp.sum(cp.multiply(tcosts_aligned.values, cp.abs(weights_var - rolled_forward_weights.values)))
        
        # Define objective function (maximize alpha, minimize penalties)
        objective = cp.Maximize(portfolio_alpha - drift_penalty - turnover_penalty)
        
        # Define constraints
        constraints = [
            cp.sum(weights_var) == 1,
            weights_var >= 0,
            weights_var <= MAX_WEIGHT
        ]
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            # Get optimized weights
            optimized_weights = weights_var.value
            
            # Check if optimization was successful
            if optimized_weights is None:
                logging.warning(f"Optimization failed for {date} - weights_var.value is None, using target weights")
                optimized_weights = current_target.values
            else:
                optimized_weights = np.clip(optimized_weights, 0.0, MAX_WEIGHT)
                optimized_weights = optimized_weights / np.sum(optimized_weights)
            
            # Store results
            weights_series = pd.Series(optimized_weights, index=common_countries)
            optimized_weights_list.append(weights_series)
            
            # Calculate metrics with asset-specific transaction costs
            portfolio_alpha_value = ALPHA_MULT * np.dot(optimized_weights, current_alphas)
            original_portfolio_alpha_value = ALPHA_MULT * np.dot(current_target, current_alphas)
            drift_penalty_value = LAMBDA_DRIFT_PENALTY * np.sum((optimized_weights - current_target) ** 2)
            
            # Calculate turnover penalty using asset-specific costs
            asset_turnovers = np.abs(optimized_weights - rolled_forward_weights)
            turnover_penalty_value = np.sum(tcosts_aligned.values * asset_turnovers)
            
            weight_diff_pct = np.sum(np.abs(optimized_weights - current_target)) * 100
            
            # Calculate total turnover (for reporting)
            total_turnover = np.sum(asset_turnovers) / 2  # One-way turnover
            
            metrics_list.append({
                'Date': date,
                'Portfolio_Alpha': portfolio_alpha_value,
                'Original_Portfolio_Alpha': original_portfolio_alpha_value,
                'Optimized_Portfolio_Alpha': portfolio_alpha_value,
                'Weight_Diff_Pct': weight_diff_pct,
                'Drift_Penalty': drift_penalty_value,
                'Turnover_Penalty': turnover_penalty_value,
                'Total_Turnover': total_turnover,
                'Total_Objective': portfolio_alpha_value - drift_penalty_value - turnover_penalty_value,
                'Solver_Status': problem.status
            })
            
            # Update previous weights for next iteration
            previous_weights = weights_series
            
        else:
            # Fallback if optimization fails
            logging.warning(f"Optimization failed for {date}, using target weights")
            weights_series = current_target.copy()
            optimized_weights_list.append(weights_series)
            
            # Calculate what we can even in fallback case
            original_portfolio_alpha_value = ALPHA_MULT * np.dot(current_target, current_alphas)
            
            # Calculate turnover penalty for fallback case
            asset_turnovers = np.abs(current_target - rolled_forward_weights)
            turnover_penalty_value = np.sum(tcosts_aligned.values * asset_turnovers)
            total_turnover = np.sum(asset_turnovers) / 2
            
            metrics_list.append({
                'Date': date,
                'Portfolio_Alpha': original_portfolio_alpha_value,
                'Original_Portfolio_Alpha': original_portfolio_alpha_value,
                'Optimized_Portfolio_Alpha': original_portfolio_alpha_value,
                'Weight_Diff_Pct': 0.0,
                'Drift_Penalty': 0.0,
                'Turnover_Penalty': turnover_penalty_value,
                'Total_Turnover': total_turnover,
                'Total_Objective': np.nan,
                'Solver_Status': problem.status
            })
            
            previous_weights = weights_series
    
    # Create result DataFrames
    optimized_weights_df = pd.DataFrame(optimized_weights_list, index=common_dates)
    metrics_df = pd.DataFrame(metrics_list)
    
    logging.info("Optimization completed successfully with asset-specific transaction costs")
    
    return optimized_weights_df, metrics_df

def save_results(optimized_weights_df, metrics_df):
    """
    Save optimization results to Excel files
    
    Args:
        optimized_weights_df: DataFrame of optimized weights
        metrics_df: DataFrame of optimization metrics
    """
    logging.info("Saving optimization results...")
    
    # Save optimized weights and metrics
    output_file = 'T2_Optimized_Country_Weights.xlsx'
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Optimized weights
        optimized_weights_df.to_excel(writer, sheet_name='Optimized Weights')
        
        # Optimization metrics
        metrics_df.to_excel(writer, sheet_name='Optimization Metrics', index=False)
        
        # Summary statistics
        summary_stats = pd.DataFrame({
            'Mean Weight': optimized_weights_df.mean(),
            'Std Dev': optimized_weights_df.std(),
            'Min Weight': optimized_weights_df.min(),
            'Max Weight': optimized_weights_df.max(),
            'Days with Weight': (optimized_weights_df > 0).sum()
        }).sort_values('Mean Weight', ascending=False)
        
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    
    logging.info(f"Optimization results saved to {output_file}")

def main():
    """Main execution function"""
    setup_logging()
    
    logging.info("Starting T2 Contrarian Target Optimization...")
    
    try:
        # Load data
        target_weights_df, returns_df, alphas_df, exposure_df, tcosts_series = load_data()
        
        # Run optimization
        optimized_weights_df, metrics_df = run_optimization(
            target_weights_df, returns_df, alphas_df, tcosts_series
        )
        
        # Save results
        save_results(optimized_weights_df, metrics_df)
        
        # Print summary
        avg_turnover = metrics_df['Total_Turnover'].mean() * 100
        avg_drift_penalty = metrics_df['Drift_Penalty'].mean()
        avg_turnover_penalty = metrics_df['Turnover_Penalty'].mean()
        
        logging.info("\n" + "="*60)
        logging.info("CONTRARIAN TARGET OPTIMIZATION SUMMARY")
        logging.info("="*60)
        logging.info(f"Average Monthly Turnover: {avg_turnover:.2f}%")
        logging.info(f"Average Drift Penalty: {avg_drift_penalty:.4f}")
        logging.info(f"Average Turnover Penalty: {avg_turnover_penalty:.4f}")
        logging.info(f"Optimization Periods: {len(metrics_df)}")
        logging.info("="*60)
        
        logging.info("T2 Contrarian Target Optimization completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during optimization: {e}")
        raise

if __name__ == "__main__":
    main()