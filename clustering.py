import pandas as pd
from datetime import datetime
import numpy as np
import tarfile
import community
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from sklearn.metrics.pairwise import cosine_similarity

## Loads the cleaned data from the preprocessing
def load_cleaned_data(tar_file_path):
    with tarfile.open(tar_file_path, "r") as tar:
        member = tar.getmembers()[0]  
        with tar.extractfile(member) as f:
            df = pd.read_parquet(f)
    return df

## Process data for louvain clustering
def get_sortest_eig(C):
    '''
    input 
        C: correlation matrix
        
    output: 
        l: eigenvalues
        v: eigenvectors 
    '''
    
    l,v = np.linalg.eigh(C)
    ordn = np.argsort(l)
    l,v = l[ordn],v[:,ordn]
    return l,v

## Creates the Louvain graph from the correlation matrix
def create_louvain_graph(correlation_matrix):
    """
    Create a Louvain graph from a correlation matrix
    """

    l, v = get_sortest_eig(correlation_matrix)

    lambda_limit = 1e-1
    len(l[l<lambda_limit]) 
    selected_indices = [i for i, l in enumerate(l) if l <= lambda_limit]

    C_r = np.zeros_like(correlation_matrix)

    for i in selected_indices:
        v_i = v[:, i]  # Get the i-th eigenvector
        outer_product = np.outer(v_i, v_i)  # Compute outer product
        C_r += l[i] * outer_product  # Add scaled matrix to the sum

    C_m = l[-1] * np.outer(v[-1, :],v[-1, :])
    C_0 = C_r + C_m
    C = abs(correlation_matrix - C_0)
    return nx.from_pandas_adjacency(C)

## Computes the log returns of each stock and minute
def compute_log_returns(df):

    filtered_df = df.copy()
    filtered_df = filtered_df.pivot_table(index='xltime', columns='stock', values='price', aggfunc='mean')
    filtered_df.index = pd.to_datetime(filtered_df.index)
    filtered_df = filtered_df[filtered_df.index.time < datetime.strptime('15:30', '%H:%M').time()]
    
    # Calculate log-returns for each stock
    log_returns = filtered_df.copy()
    log_returns.iloc[:, :] = log_returns.apply(lambda col: np.log(col).diff())

    log_returns['Date'] = log_returns.index.date
    daily_data = log_returns.groupby('Date').apply(lambda x: x.drop(columns='Date').values.flatten())
    daily_data_df = pd.DataFrame(daily_data.tolist(), index=daily_data.index)
    daily_data_df = daily_data_df.fillna(0)
    
    return daily_data_df

## Plots the found clusters
def plot_clusters(G, partition):

    #Visualize the clusters 
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    return None

## Creates and visualizes all the clusters from a dataframe of the period_data 
def create_clusters(df):

    daily_data_df = compute_log_returns(df)
    
    print(f"Daily data shape: {daily_data_df.shape}")

    correlation_matrix = daily_data_df.T.corr()
    G = create_louvain_graph(correlation_matrix)

    # Apply Louvain clustering
    partition = community.community_louvain.best_partition(G, weight='weight', random_state = 42)

    # Convert results to a DataFrame
    clusters = pd.DataFrame({'Day': list(partition.keys()), 'Cluster': list(partition.values())})
    print("There are", len(clusters['Cluster'].unique()), "clusters")
    print("The length of each clusters are", clusters.groupby('Cluster').size())

    plot_clusters(G, partition)

    return clusters, daily_data_df

## Creates and visualizes the clusters from the .tar file.
def create_clusters_file(tar_file_path):
    # Argument parsing
    print(f"Processing tar file: {tar_file_path}")
    period_data = load_cleaned_data(tar_file_path)
    print(f"Combined DataFrame shape: {period_data.shape}")

    daily_data_df = compute_log_returns(period_data)
    
    print(f"Daily data shape: {daily_data_df.shape}")

    correlation_matrix = daily_data_df.T.corr()
    G = create_louvain_graph(correlation_matrix)

    # Apply Louvain clustering
    partition = community.community_louvain.best_partition(G, weight='weight')

    # Convert results to a DataFrame
    clusters = pd.DataFrame({'Day': list(partition.keys()), 'Cluster': list(partition.values())})
    print("There are", len(clusters['Cluster'].unique()), "clusters")
    print("The length of each clusters are", clusters.groupby('Cluster').size())

    plot_clusters(G, partition)

    return clusters, daily_data_df

def classify_new_day(new_day_data, daily_data_df, clusters):
    """
    Classify a new day into an existing cluster based on correlation similarity with historical clusters.

    Parameters:
        - new_day_data: DataFrame containing log-returns of the new day to classify.
        - daily_data_df: DataFrame of historical daily data used for clustering.
        - clusters: DataFrame containing cluster assignments for the historical data.

    Returns:
        - assigned_cluster: The cluster the new day is assigned to.
    """
        
    # Align new_day_data to match the shape of daily_data_df
    new_day_data = new_day_data.fillna(0).squeeze()
    daily_data_df = daily_data_df.fillna(0)
    
    # Compute cosine similarity
    similarities = cosine_similarity([new_day_data], daily_data_df.values)[0]
    
    # Assign the cluster based on the highest average similarity
    cluster_similarities = {}
    for cluster in clusters['Cluster'].unique():
        cluster_days = clusters[clusters['Cluster'] == cluster]['Day']
        cluster_indices = daily_data_df.index.isin(cluster_days)
        cluster_similarities[cluster] = similarities[cluster_indices].mean()
    
    # Find the cluster with the highest similarity
    assigned_cluster = max(cluster_similarities, key=cluster_similarities.get)
    print(f"The new day is assigned to cluster {assigned_cluster} with similarity {cluster_similarities[assigned_cluster]}")
    return assigned_cluster

def compute_test_train(period_data, threshold):
    ## Compute the 90 percent for training and the rest for testing
    period_data['date'] = period_data['xltime'].dt.date

    start_date = period_data['date'].iloc[0]  # First date
    end_date = period_data['date'].iloc[-1]   # Last date

    period_data = period_data.drop(columns=['date'])

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    stop_date = (start_date + (end_date - start_date) * threshold).normalize()

    # Filter the DataFrame to keep rows before the stop date
    train_data = period_data[period_data["xltime"] < stop_date]
    test_data = period_data[period_data["xltime"] >= stop_date]

    days = len(test_data["xltime"].dt.date.unique())

    return train_data, test_data, days

def classify_test_data(df, threshold=0.8):
    train, test, days = compute_test_train(df, threshold)
    
    clusters, historical_log = create_clusters(train)

    unique_dates = test["xltime"].dt.date.unique()

    # Iterate through each unique date
    for date in unique_dates:
        # Filter the DataFrame for the current date
        date_df = test[test["xltime"].dt.date == date]
        print(date)
        if(pd.to_datetime(date).date() == pd.to_datetime("2012-12-24").date()):
            continue
        new_log_returns = compute_log_returns(date_df)
        try:
            cluster = classify_new_day(new_log_returns, historical_log, clusters)
        except Exception as e: 
            continue
        new_row_data = {'Day': [date], 'Cluster': [cluster]}
        # Create a new DataFrame from the row data
        new_row = pd.DataFrame(new_row_data)
        # Add the new rows to the existing DataFrame
        clusters = pd.concat([clusters, new_row], ignore_index=True)

    return clusters, days

def main():

    parser = argparse.ArgumentParser(description="Cluster ETF data days")
    parser.add_argument('main_tar_file', type=str, help="Path to the main ETF tar file")
    args = parser.parse_args()
    create_clusters(args.main_tar_file)

if __name__ == '__main__':
    main()
