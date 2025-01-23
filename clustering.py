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

# process data for louvain clustering
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

def create_clusters(tar_file_path):
    # Argument parsing
    combined_df = pd.DataFrame()

    print(f"Processing tar file: {tar_file_path}")

    # Open the tar file
    with tarfile.open(tar_file_path, 'r') as tar:
        # Iterate through each member in the tar file
        for member in tar.getmembers():
            # Check if the file is a .parquet file
            if member.isfile() and member.name.endswith('.parquet'):
                # Extract the file
                extracted_file = tar.extractfile(member)
                if extracted_file:
                    # Load the parquet file into a DataFrame
                    daily_df = pd.read_parquet(extracted_file)
                    # Reset the index and keep it as a column
                    daily_df.reset_index(inplace=True)
                    # Append to the combined DataFrame
                    combined_df = pd.concat([combined_df, daily_df], ignore_index=True)

    print(f"Combined DataFrame shape: {combined_df.shape}")

    filtered_df = combined_df.copy()
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
    
    print(f"Daily data shape: {daily_data_df.shape}")

    correlation_matrix = daily_data_df.T.corr()
    G = create_louvain_graph(correlation_matrix)

    # Apply Louvain clustering
    partition = community.community_louvain.best_partition(G, weight='weight')

    # Convert results to a DataFrame
    clusters = pd.DataFrame({'Day': list(partition.keys()), 'Cluster': list(partition.values())})
    print("There are", len(clusters['Cluster'].unique()), "clusters")
    print("The length of each clusters are", clusters.groupby('Cluster').size())

    print("The clusters are:", clusters)

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
        
        # Compute correlation of the new day's vector with the existing days' vectors (daily_data_df)
        new_day_corr = daily_data_df.corrwith(pd.Series(new_day_data))
        # Assign the new day to the cluster with the highest correlation
        # First, calculate the average correlation for each cluster
        cluster_correlations = {}
        for cluster in clusters['Cluster'].unique():
            cluster_days = clusters[clusters['Cluster'] == cluster]['Day']
            cluster_corrs = new_day_corr[cluster_days.index]
            cluster_correlations[cluster] = cluster_corrs.mean()

        # Assign the new day to the cluster with the highest average correlation
        assigned_cluster = max(cluster_correlations, key=cluster_correlations.get)
        print(f"The new day is assigned to cluster {assigned_cluster} with correlation {cluster_correlations[assigned_cluster]}")
        return assigned_cluster

def main():

    parser = argparse.ArgumentParser(description="Cluster ETF data days")
    parser.add_argument('main_tar_file', type=str, help="Path to the main ETF tar file")
    args = parser.parse_args()
    create_clusters(args.main_tar_file)

if __name__ == '__main__':
    main()
