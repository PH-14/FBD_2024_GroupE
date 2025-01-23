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

def main():
    # Argument parsing

    parser = argparse.ArgumentParser(description="Cluster ETF data days")
    parser.add_argument('main_tar_file', type=str, help="Path to the main ETF tar file")
    args = parser.parse_args()

    tar_file_path = args.main_tar_file
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

if __name__ == '__main__':
    main()
