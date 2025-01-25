import numpy as np
import pandas as pd

def investment_strategy(strat: str, states: pd.DataFrame, trades: pd.DataFrame, w: int):
    """
    Computes investment weights based on historical cluster performance using the tangency portfolio approach.
    
    Parameters:
    states (pd.DataFrame): DataFrame of size (w, 1) indicating the market state for each day in the window.
    trades (pd.DataFrame): DataFrame of size (w, n) containing log returns for each stock at the end of the day.
    w (int): Size of the rolling window.

    Returns:
    np.array: Investment weights for the current day, ensuring they sum to one.
    """

    # Get the cluster for the current day (excluding the last day)
    current_state = states.iloc[-2].values[0]
    
    # Filter trades belonging to the same cluster (excluding the last day)
    cluster_trades = trades.iloc[:-1][states.iloc[:-1, 0] == current_state]

    # Compute mean and covariance matrix of log returns
    mean_log_returns = cluster_trades.mean()
    cov_matrix = cluster_trades.cov()

    if strat == "ten":        
        # Compute tangency portfolio weights
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(len(mean_log_returns))
        tangency_weights = inv_cov_matrix @ mean_log_returns / (ones.T @ inv_cov_matrix @ mean_log_returns)
        
        # Normalize to ensure weights sum to one
        investment_weights = tangency_weights / tangency_weights.sum()
    elif strat == "w_ret":
        max_pos_return = mean_log_returns[mean_log_returns > 0].max()
        min_neg_return = mean_log_returns[mean_log_returns < 0].min()
        investment_weights = np.where(mean_log_returns > 0, mean_log_returns / max_pos_return, mean_log_returns / abs(min_neg_return))
    else:
        investment_weights = np.where(mean_log_returns > 0, 1, -1)
    
    return investment_weights.values

def compute_portfolio_return(weights: np.array, returns: pd.DataFrame):
    """
    Computes the portfolio return based on investment weights and stock returns.
    
    Parameters:
    weights (np.array): Array of investment weights for each stock.
    returns (pd.DataFrame): DataFrame containing log returns for each stock.

    Returns:
    float: Portfolio return computed as the dot product of weights and returns.
    """
    portfolio_return = np.dot(weights, returns.iloc[-1].values)
    return portfolio_return
