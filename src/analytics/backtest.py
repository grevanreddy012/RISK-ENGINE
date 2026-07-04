import numpy as np
import pandas as pd
from scipy.stats import chi2
from .metrics import get_returns, historical_var_cvar

def kupiec_pof_test(breaches, total_obs, conf_level):
    """
    Kupiec's Proportion of Failures (POF) test.
    """
    p_expected = 1 - conf_level
    p_observed = breaches / total_obs
    
    # If there are no breaches, or breaches equal total_obs, handle edge cases
    if breaches == 0:
        lr_stat = -2 * np.log((1 - p_expected)**total_obs)
    elif breaches == total_obs:
        lr_stat = -2 * np.log(p_expected**total_obs)
    else:
        num = ((1 - p_expected)**(total_obs - breaches)) * (p_expected**breaches)
        den = ((1 - p_observed)**(total_obs - breaches)) * (p_observed**breaches)
        lr_stat = -2 * np.log(num / den)
        
    # Degrees of freedom = 1
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    
    # Well-calibrated if p-value > 0.05 (using 95% significance level for the test itself)
    is_valid = p_value > 0.05
    
    return lr_stat, p_value, is_valid

def run_backtest(returns_df, weights, conf_level=0.99, window=252):
    """
    Run rolling VaR backtest over the history.
    Args:
        returns_df: DataFrame of asset returns
        weights: Portfolio weights
        conf_level: VaR confidence level
        window: Rolling window size for VaR calculation
    """
    port_ret = returns_df.dot(weights)
    
    if len(port_ret) <= window:
        return None
        
    actual_returns = []
    var_forecasts = []
    dates = []
    
    # Compute rolling Historical VaR
    # (Using historical as the baseline for backtesting)
    for i in range(window, len(port_ret)):
        # Data available up to day i-1 to forecast day i
        window_returns = returns_df.iloc[i-window:i]
        
        # Calculate 1-day VaR
        # Note: historical_var_cvar expects returns_df, so we pass the window slice
        var, _ = historical_var_cvar(window_returns, weights, conf_level, horizon=1)
        
        # actual return on day i
        actual = port_ret.iloc[i]
        
        var_forecasts.append(var)
        actual_returns.append(actual)
        dates.append(port_ret.index[i])
        
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual Return': actual_returns,
        'VaR Forecast': var_forecasts
    }).set_index('Date')
    
    # A breach occurs when the actual return is worse (more negative) than the -VaR threshold
    # Since VaR is positive, we compare actual_return < -VaR
    results_df['Breach'] = results_df['Actual Return'] < -results_df['VaR Forecast']
    
    total_obs = len(results_df)
    breaches = results_df['Breach'].sum()
    
    lr_stat, p_value, is_valid = kupiec_pof_test(breaches, total_obs, conf_level)
    
    test_results = {
        'Total Observations': total_obs,
        'Expected Breaches': round(total_obs * (1 - conf_level), 2),
        'Actual Breaches': breaches,
        'LR Statistic': lr_stat,
        'p-value': p_value,
        'Is Calibrated (95% sig)': is_valid
    }
    
    return results_df, test_results
