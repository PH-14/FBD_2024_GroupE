import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def compute_tangency_weights(mean_log_returns, cov_matrix):
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones = np.ones(len(mean_log_returns))
    tangency_weights = inv_cov_matrix @ mean_log_returns / (ones.T @ inv_cov_matrix @ mean_log_returns)
    return tangency_weights / tangency_weights.sum()

def compute_w_ret_weights(mean_log_returns):
    max_pos_return = mean_log_returns[mean_log_returns > 0].max()
    min_neg_return = mean_log_returns[mean_log_returns < 0].min()
    weights = np.where(mean_log_returns > 0, mean_log_returns / max_pos_return, mean_log_returns / abs(min_neg_return))
    weights = weights / np.sum(np.abs(weights))
    return weights 

def compute_top_bottom_weights(mean_log_returns, quantile):
    top_threshold = mean_log_returns.quantile(1 - quantile)
    bottom_threshold = mean_log_returns.quantile(quantile)
    weights = np.zeros(len(mean_log_returns))
    weights[mean_log_returns >= top_threshold] = 1 / (mean_log_returns >= top_threshold).sum()
    weights[mean_log_returns <= bottom_threshold] = -1 / (mean_log_returns <= bottom_threshold).sum()
    return weights 

def compute_momentum_weights(mean_log_returns):
    weights = np.where(mean_log_returns > 0, mean_log_returns / mean_log_returns.sum(), 0)
    weights = np.where(mean_log_returns < 0, mean_log_returns / abs(mean_log_returns).sum(), weights)
    return weights / weights.sum()

def compute_risk_parity_weights(cluster_trades):
    risk_contributions = 1 / cluster_trades.std()
    return risk_contributions / risk_contributions.sum()

def investment_strategy(strat: str, df_states: pd.DataFrame, df_trades: pd.DataFrame, quantile: float = 0.25, use_cluster: bool = True): 
    """
    Computes investment weights based on historical cluster performance using various strategies.
    
    Parameters:
    states (pd.DataFrame): DataFrame indicating the market state for each day in the window.
    trades (pd.DataFrame): DataFrame containing log returns for each stock at the end of the day.
    w (int): Size of the rolling window.
    quantile (float): Percentage of top and bottom stocks to long and short, default is 10%.

    Returns:
    np.array: Investment weights for the current day, ensuring they sum to one.
    """

    # Get the cluster for the current day 
    states = df_states.copy()
    trades = df_trades.copy()
    current_state = states['Cluster'].iloc[-1]


    if use_cluster:
        # Filter trades belonging to the same cluster (excluding the last day)
        cluster_trades = trades[states['Cluster'] == current_state]
    else:
        cluster_trades = trades

    # Compute mean and covariance matrix of log returns
    mean_log_returns = cluster_trades.mean()
    cov_matrix = cluster_trades.cov()

    if strat == "tan":
        investment_weights = compute_tangency_weights(mean_log_returns, cov_matrix)
    elif strat == "w_ret":
        investment_weights = compute_w_ret_weights(mean_log_returns)
    elif strat == "top_bottom":
        investment_weights = compute_top_bottom_weights(mean_log_returns, quantile)
    elif strat == "momentum":
        investment_weights = compute_momentum_weights(mean_log_returns)
    elif strat == "risk_parity":
        investment_weights = compute_risk_parity_weights(cluster_trades)
    else:
        investment_weights = np.where(mean_log_returns > 0, 1, -1)
    
    return investment_weights

def compute_portfolio_return(weights: np.array, returns: pd.DataFrame):
    """
    Computes the portfolio return based on investment weights and stock returns.
    
    Parameters:
    weights (np.array): Array of investment weights for each stock.
    returns (pd.DataFrame): DataFrame containing log returns for each stock on that day.

    Returns:
    float: Portfolio return computed as the dot product of weights and returns.
    """
    # Ensure weight sum to one
    
    portfolio_return = np.dot(weights, returns.values)
    return portfolio_return

def compute_log_returns_eod(df, fill_method='ffill'):

    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Handle missing values based on the chosen fill method
    if fill_method == 'ffill':
        df = df.ffill()
    elif fill_method == 'interpolate':
        df = df.interpolate(method='linear')

    # Group by date and compute intraday log return
    intraday_returns = df.groupby(df.index.date).apply(lambda x: np.log(x.iloc[-1] / x.iloc[0]))

    # Remove the multi-index introduced by groupby
    #intraday_returns.index = intraday_returns.index.droplevel(0)
    
    return intraday_returns


def compute_returns_eod(processed_data, fill_method='ffill'):
    df = processed_data.copy()
    df = df.pivot_table(index='xltime', columns='stock', values='price', aggfunc='mean')
    df.index = pd.to_datetime(df.index)
    df = df[df.index.time >= datetime.strptime('15:30', '%H:%M').time()]

    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Handle missing values based on the chosen fill method
    if fill_method == 'ffill':
        df = df.ffill()
    elif fill_method == 'interpolate':
        df = df.interpolate(method='linear')

    # Group by date and compute intraday log return
    intraday_returns = df.groupby(df.index.date).apply(lambda x: x.iloc[-1]/ x.iloc[0] -1 )
    
    intraday_returns.index = intraday_returns.index.rename('Day')
    intraday_returns.index = pd.to_datetime(intraday_returns.index)
    
    return intraday_returns


def rolling_backtest(df_states, df_trades, min_window_size=30, quantile=0.25):
    """
    Performs a rolling window backtest of the investment strategy.

    Parameters:
    - strategy_func: Function to apply the investment strategy
    - df_states: DataFrame with market states for each day
    - df_trades: DataFrame with log returns for each stock per day
    - window_size: Number of past days to consider in the rolling window
    - strat: The investment strategy to use (e.g., 'tan', 'w_ret', etc.)
    - quantile: Percentage of top and bottom stocks to long and short (default 10%)

    Returns:
    - results_df: DataFrame containing portfolio returns for each day in the test period
    """
    results =  pd.DataFrame(index=df_states.index[min_window_size:])
    strats = ['tan', 'w_ret', 'top_bottom', 'momentum', 'risk_parity']

    for strat in strats:
        for i in tqdm(range(min_window_size, len(df_states))):
            
            # Select rolling window for states and trades
            rolling_states = df_states.iloc[: i + 1]
            rolling_return = df_trades.iloc[: i + 1]

            # Call the investment strategy function
            weights = investment_strategy(strat, rolling_states, rolling_return, quantile)

            # Compute portfolio return using today's actual returns (excluding the last day)
            portfolio_return = compute_portfolio_return(weights, rolling_return.iloc[-1])

            # Add a column to the results DataFrame with name strat + '_return'
            results.loc[df_states.index[i], 'c_' + strat + '_return'] = portfolio_return

            # Redo without clusters
            weights = investment_strategy(strat, rolling_states, rolling_return, quantile, use_cluster=False)

            # Compute portfolio return using today's actual returns (excluding the last day)
            portfolio_return = compute_portfolio_return(weights, rolling_return.iloc[-1])

            results.loc[df_states.index[i], strat + '_return'] = portfolio_return


        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        

    return results_df


def compute_returns(processed_data, clusters_results, days):
    """
    computes the returns for all the strategies
    """
    stock_returns = compute_returns_eod(processed_data)

    clusters_results.set_index('Day', inplace=True)
    clusters_results.index = pd.to_datetime(clusters_results.index)

     # drop lines in trades in for which we don't have clusters
    stock_returns = stock_returns[stock_returns.index.isin(clusters_results.index)]

    results = rolling_backtest(clusters_results, stock_returns, min_window_size=days)

    return results