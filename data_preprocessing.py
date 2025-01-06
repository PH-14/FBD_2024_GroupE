import os
import tarfile
import pandas as pd
import xlrd
from datetime import datetime
import argparse

# Function to convert xltime to datetime using xlrd
def xltime_to_datetime(xltime):
    return xlrd.xldate_as_datetime(xltime, 0)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process ETF data for stock trading.")
    parser.add_argument('main_tar_file', type=str, help="Path to the main ETF tar file")
    parser.add_argument('N', type=int, help="Number of stocks to include in the dataset")
    parser.add_argument('T', type=int, help="Number of parquet files (days) per stock")
    parser.add_argument('TIME_THRESHOLD', type=str, help="Time threshold (e.g., '16:00')")
    
    args = parser.parse_args()

    main_tar_file = args.main_tar_file
    data_dir = main_tar_file.split("/")[0]
    N = args.N
    T = args.T
    TIME_THRESHOLD = args.TIME_THRESHOLD
    sep = "/"

    # Initialize lists to store the training and validation data
    train_data = []
    val_data = []

    # Open the main ETF tar file
    with tarfile.open(main_tar_file, "r") as main_tar:
        stock_tars = [f for f in main_tar.getnames() if f.endswith('.tar') and 'trade' in f]
        
        # Limit the number of stocks to N
        for i, stock_tar in enumerate(stock_tars[:N]):
            print(f"Processing stock {i+1}/{min(N, len(stock_tars))}: {stock_tar}")
            
            with main_tar.extractfile(stock_tar) as sub_tar_file:
                with tarfile.open(fileobj=sub_tar_file, mode='r') as stock_tar:
                    parquet_files = [f for f in stock_tar.getnames() if f.endswith('.parquet') and 'trade' in f]
                    
                    # Only process the first T parquet files
                    for parquet_file in parquet_files[:T]:
                        with stock_tar.extractfile(parquet_file) as f:
                            stock_data = pd.read_parquet(f)
                            
                            # Convert xltime to datetime
                            stock_data['Timestamp'] = stock_data['xltime'].apply(xltime_to_datetime)
                            
                            # Add the stock's symbol and date to the dataframe
                            stock_symbol = parquet_file.split(sep)[-2]
                            stock_data['Stock'] = stock_symbol
                            
                            # Drop unnecessary column
                            stock_data = stock_data.drop(columns=['xltime'])
                            
                            # Add the data to the train and validation lists
                            stock_data['Timestamp'] = pd.to_datetime(stock_data['Timestamp'])

                            # Create a column to indicate whether the trade is before or after 4pm
                            stock_data['Time_of_day'] = stock_data['Timestamp'].dt.time
                            stock_data[f'Before_{TIME_THRESHOLD}'] = stock_data['Time_of_day'] <= datetime.strptime(TIME_THRESHOLD, '%H:%M').time()

                            # Separate into before 4pm and after 4pm
                            before = stock_data[stock_data[f'Before_{TIME_THRESHOLD}']]
                            after = stock_data[~stock_data[f'Before_{TIME_THRESHOLD}']]
                            
                            # Split 80% train, 20% val for each stock's before and after data
                            train_before = before.sample(frac=0.8, random_state=42)
                            val_before = before.drop(train_before.index)
                            
                            train_after = after.sample(frac=0.8, random_state=42)
                            val_after = after.drop(train_after.index)
                            
                            # Append the data to the train and validation sets
                            train_data.append(pd.concat([train_before, train_after]))
                            val_data.append(pd.concat([val_before, val_after]))

    # Concatenate all stocks into the final train and validation sets
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)

    # Save the datasets
    train_output_path = f"{data_dir}/train_data_N{N}_T{T}_2012.parquet"
    val_output_path = f"{data_dir}/val_data_N{N}_T{T}_2012.parquet"

    train_df.to_parquet(train_output_path)
    val_df.to_parquet(val_output_path)
    
    assert len(train_df) + len(val_df) == len(train_df) + len(val_df)
    print(f"Training dataset saved to {train_output_path}")
    print(f"Validation dataset saved to {val_output_path}")

if __name__ == '__main__':
    main()
