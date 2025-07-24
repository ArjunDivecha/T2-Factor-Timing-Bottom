# T2 Factor Timing Bottom - Contrarian Strategy

A comprehensive contrarian factor timing strategy implementation that selects the bottom 20% performing factors for portfolio construction, based on the hypothesis that underperforming factors may provide better future returns due to mean reversion or market inefficiencies.

## Overview

This repository contains a complete contrarian investment pipeline that:
- Selects the **bottom 20%** of countries for each factor (contrarian approach)
- Calculates returns based on worst-performing countries
- Optimizes portfolios to minimize expected returns while managing risk
- Generates country-specific investment weights using contrarian factor selection

## Strategy Philosophy

The contrarian approach is based on the hypothesis that:
- Underperforming factors may provide better future returns due to mean reversion
- Market inefficiencies create opportunities in poorly-performing assets
- Traditional momentum strategies may be overcrowded, creating alpha in contrarian approaches

## Pipeline Architecture

```
Input Data → Portfolio Construction → Return Calculation → Optimization → Country Weights
     ↓              ↓                      ↓               ↓              ↓
Normalized_T2   Step 3: Bottom 20%    Step 4: Contrarian  Step 5:       Step 8:
MasterCSV.csv   Portfolio Creation    Return Calculation  CVXPY Opt     Country Weights
```

## Key Files

### Core Processing Scripts
- **Step Three Top20 Portfolios.py** - Contrarian portfolio construction (bottom 20% selection)
- **Step Four Create Monthly Top20 Returns.py** - Contrarian return calculation
- **Step Five FAST.py** - Contrarian portfolio optimization using CVXPY
- **Step Eight Write Country Weights.py** - Country weight allocation using contrarian factors
- **Step Fourteen Target Optimization.py** - Turnover-optimized contrarian portfolio weights

### Backup Files
- **backup/** - Original top 20% strategy files for comparison and rollback

### Configuration
- **.kiro/specs/contrarian-portfolio-conversion/** - Complete specification and task documentation

## Key Changes from Long to Short Strategy

| Component | Original (Long) | Contrarian (Short) | Purpose |
|-----------|----------------|-------------------|---------|
| Portfolio Construction | `nlargest()` - Top 20% | `nsmallest()` - Bottom 20% | Select worst performers |
| Return Calculation | Top performer returns | Bottom performer returns | Calculate contrarian returns |
| Optimization Objective | `Maximize(returns - penalties)` | `Maximize(-returns - penalties)` | Minimize expected returns |
| Country Weight Allocation | Top 20% countries | Bottom 20% countries | Allocate to worst performers |

## Mathematical Framework

### Contrarian Portfolio Selection
For each factor and date:
```python
n_select = max(1, int(len(countries) * 0.2))
selected_countries = countries.nsmallest(n_select, 'factor_value')
```

### Contrarian Optimization Objective
```python
objective = cp.Maximize(-portfolio_return - risk_penalty - concentration_penalty)
```
Where:
- `-portfolio_return`: Minimizes expected returns (contrarian approach)
- `risk_penalty`: Maintains risk control
- `concentration_penalty`: Ensures diversification

### Transaction Cost Model
```python
total_transaction_cost = borrow_cost + trading_cost
turnover_penalty = sum(total_transaction_cost * abs(weight_changes))
```

## Usage

### Basic Pipeline Execution

1. **Portfolio Construction**: `python "Step Three Top20 Portfolios.py"`
2. **Return Calculation**: `python "Step Four Create Monthly Top20 Returns.py"`
3. **Factor Optimization**: `python "Step Five FAST.py"`
4. **Country Weight Generation**: `python "Step Eight Write Country Weights.py"`
5. **Turnover Optimization**: `python "Step Fourteen Target Optimization.py"`

## Dependencies

```python
pandas >= 2.0.0
numpy >= 1.24.0
cvxpy >= 1.3.0
matplotlib >= 3.5.0
openpyxl >= 3.1.0
xlsxwriter >= 3.0.0
```

## Performance Features

- **CVXPY Optimization**: 50-100x speed improvement over scipy.optimize
- **Warm-start Capabilities**: 2-3x additional speedup for repeated optimizations
- **Vectorized Operations**: Efficient matrix operations for large datasets
- **Asset-specific Transaction Costs**: Individual borrow and trading costs per country
- **Memory Optimization**: Efficient handling of large time series datasets

## License

This project is proprietary and confidential.