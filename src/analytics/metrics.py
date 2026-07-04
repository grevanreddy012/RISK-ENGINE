import numpy as np
import pandas as pd
from scipy.stats import norm

def get_returns(prices_df):
    """Calculate daily logarithmic returns from prices."""
    return np.log(prices_df / prices_df.shift(1)).dropna()

def compute_portfolio_returns(returns_df, weights):
    """Calculate portfolio returns based on weights."""
    return returns_df.dot(weights)

def scale_horizon(value, horizon):
    """Scale risk metric by sqrt(t)."""
    return value * np.sqrt(horizon)

def historical_var_cvar(returns_df, weights, conf_level=0.99, horizon=1):
    """
    Historical Simulation VaR and CVaR.
    """
    port_ret = compute_portfolio_returns(returns_df, weights)
    alpha = 1 - conf_level
    
    # Percentile of the return distribution
    var_percentile = np.percentile(port_ret, alpha * 100)
    
    # VaR is expressed as a positive loss
    var = -var_percentile
    
    # CVaR is the average of returns worse than the VaR threshold
    cvar_returns = port_ret[port_ret <= var_percentile]
    cvar = -cvar_returns.mean() if len(cvar_returns) > 0 else var
    
    return scale_horizon(var, horizon), scale_horizon(cvar, horizon)

def parametric_var_cvar(returns_df, weights, conf_level=0.99, horizon=1):
    """
    Parametric (Variance-Covariance) VaR and CVaR assuming Normal distribution.
    """
    cov_matrix = returns_df.cov()
    mean_returns = returns_df.mean()
    
    port_mean = np.dot(weights, mean_returns)
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_std = np.sqrt(port_var)
    
    alpha = 1 - conf_level
    z_score = norm.ppf(alpha)
    
    var = -(port_mean + z_score * port_std)
    
    # CVaR for normal distribution
    phi = norm.pdf(z_score)
    cvar = -(port_mean) + port_std * (phi / alpha)
    
    return scale_horizon(var, horizon), scale_horizon(cvar, horizon)

def monte_carlo_var_cvar(returns_df, weights, conf_level=0.99, horizon=1, n_paths=10000):
    """
    Monte Carlo VaR and CVaR using Cholesky decomposition.
    """
    np.random.seed(42) # For reproducibility
    cov_matrix = returns_df.cov()
    mean_returns = returns_df.mean()
    
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback if covariance matrix is not positive definite
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        
    # Generate correlated random normal variables
    Z = np.random.standard_normal((n_paths, len(weights)))
    simulated_returns = mean_returns.values + Z @ L.T
    
    port_sim_returns = simulated_returns.dot(weights)
    
    alpha = 1 - conf_level
    var_percentile = np.percentile(port_sim_returns, alpha * 100)
    var = -var_percentile
    
    cvar_returns = port_sim_returns[port_sim_returns <= var_percentile]
    cvar = -cvar_returns.mean() if len(cvar_returns) > 0 else var
    
    return scale_horizon(var, horizon), scale_horizon(cvar, horizon)

def calculate_all_metrics(returns_df, weights, conf_level=0.99, horizon=1):
    """Calculate all VaR and CVaR metrics."""
    metrics = {}
    
    h_var, h_cvar = historical_var_cvar(returns_df, weights, conf_level, horizon)
    p_var, p_cvar = parametric_var_cvar(returns_df, weights, conf_level, horizon)
    m_var, m_cvar = monte_carlo_var_cvar(returns_df, weights, conf_level, horizon)
    
    metrics['Historical'] = {'VaR': h_var, 'CVaR': h_cvar}
    metrics['Parametric'] = {'VaR': p_var, 'CVaR': p_cvar}
    metrics['Monte Carlo'] = {'VaR': m_var, 'CVaR': m_cvar}
    
    return metrics
