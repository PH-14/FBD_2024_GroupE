from datetime import datetime
import pandas as pd
import tarfile
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from clustering import *
import os
from tkinter import font
import matplotlib.ticker as ticker
import seaborn as sns

## Loads the data cleaned from the exploration
def load_data(tar_file):
    combined_df = pd.DataFrame()

    with tarfile.open(tar_file, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.parquet'):
                # Extract the file
                extracted_file = tar.extractfile(member)
                if extracted_file:
                    # Load the parquet file into a DataFrame
                    daily_df = pd.read_parquet(extracted_file)
                    daily_df.reset_index(inplace=True)
                    combined_df = pd.concat([combined_df, daily_df], ignore_index=True)

    return combined_df

## Computes the percentage of missing minutes (over the maximum number of minutes per day) for each stock.
def calculate_ratio_missing_minutes_per_stocks(df): 

    # Create an empty dictionary to store results
    missing_minutes_ratio = {}

    stock_minutes = {}
    max_minutes = 0

    # List of unique stocks in the dataset
    unique_stocks = df['stock'].unique()

    # Count the number of minutes for each stock
    for stock in unique_stocks:
        # Filter the DataFrame for the current stock and count the rows (minutes)
        nbr_minutes = df[df['stock'] == stock].shape[0]

        # Update max minutes if the current stock has more
        if nbr_minutes > max_minutes:
            max_minutes = nbr_minutes

        # Add the minutes for the stock to the dictionary
        stock_minutes[stock] = nbr_minutes

    # Iterate through the stock data
    for stock, minutes in stock_minutes.items():
        if minutes < max_minutes:
            missing_percentage = 100 * (max_minutes - minutes) / (max_minutes)
            missing_minutes_ratio[stock] = round(missing_percentage, 2)
        else:
            missing_minutes_ratio[stock] = 0.0  # Complete data means 0% missing

    return missing_minutes_ratio

## Adds a new row with Nans values for each missing minute in the dataframe.
def fill_bid_price_by_day_and_stock(df):
    # Ensure xltime is consistent and timezone-naive
    df['xltime'] = pd.to_datetime(df['xltime']).dt.tz_localize(None)

    # Extract the date from xltime for grouping
    df['date'] = df['xltime'].dt.date

    # Define the valid range of trading minutes (09:30 to 15:59)
    trading_minutes = pd.date_range("09:30", "15:59", freq="min").time

    # Initialize an empty list to store the augmented rows
    augmented_rows = []

    # Group by stock and date
    for (stock, date), group in df.groupby(['stock', 'date']):
        # Get the set of minutes already present for the stock on this date
        existing_minutes = group['xltime'].dt.time

        # Find missing minutes by comparing with the full range of trading minutes
        missing_minutes = set(trading_minutes) - set(existing_minutes)

        # Create rows for the missing minutes
        for missing_minute in missing_minutes:
            augmented_rows.append({
                'stock': stock,
                'xltime': pd.Timestamp(f"{date} {missing_minute}"),
                # Fill the rest of the columns with NaN
                **{col: np.nan for col in df.columns if col not in ['stock', 'xltime']}
            })

    # Create a DataFrame from the augmented rows
    augmented_df = pd.DataFrame(augmented_rows)

    # Combine the original DataFrame with the new rows and sort by stock, date, and time
    df = pd.concat([df, augmented_df], ignore_index=True)
    df = df.sort_values(by=['stock', 'xltime']).reset_index(drop=True)

    # Drop the temporary 'date' column
    df = df.drop(columns=['date'])

    return df

## Fills the Nan values with the weighted average of the previous 5 minutes.
def fill_missing_values_by_stock_and_date(df):

    # Ensure xltime is consistent and timezone-naive
    df['xltime'] = pd.to_datetime(df['xltime']).dt.tz_localize(None)

    # Extract the date from xltime for grouping
    df['date'] = df['xltime'].dt.date

    def fill_group(group):
        # Sort the group by time to ensure proper ordering
        group = group.sort_values('xltime')

        # Fill missing prices using weighted average of previous 5 minutes
        def compute_weighted_price(row, idx):
            # Select the previous 5 rows
            previous_rows = group.iloc[max(0, idx - 5):idx]
            # Calculate the total weights
            total_weights = previous_rows['bid_volume'] + previous_rows['ask_volume']
            # Weighted average formula
            if total_weights.sum() > 0:
                return np.sum(previous_rows['price'] * total_weights) / total_weights.sum()
            else:
                return np.nan

        # Fill missing bid_volume and ask_volume using the simple average of the last 5 minutes
        def compute_average_volume(column, idx):
            previous_rows = group.iloc[max(0, idx - 5):idx]
            return round(previous_rows[column].mean(),2)

        # Iterate through each row and fill missing values
        for idx, row in group.iterrows():
            if pd.isna(row['price']):
                group.loc[idx, 'price'] = compute_weighted_price(row, idx)
            if pd.isna(row['bid_volume']):
                group.loc[idx, 'bid_volume'] = compute_average_volume('bid_volume', idx)
            if pd.isna(row['ask_volume']):
                group.loc[idx, 'ask_volume'] = compute_average_volume('ask_volume', idx)

        return group

    # Apply the filling logic to each group (by stock and date)
    df = df.groupby(['stock', 'date'], group_keys=False).apply(fill_group)

    # Drop the temporary 'date' column
    df = df.drop(columns=['date'])

    return df

## Saves the obtained cleaned file to the data folder
def save_to_parquet_and_tar(df, parquet_filename, tar_filename):

    # Save the DataFrame to a .parquet file
    df.to_parquet(parquet_filename, index=False)
    print(f"Saved DataFrame to {parquet_filename}")

    # Compress the .parquet file into a .tar archive
    with tarfile.open(tar_filename, "w") as tar:
        tar.add(parquet_filename, arcname=os.path.basename(parquet_filename))
    print(f"Compressed {parquet_filename} into {tar_filename}")

    # Optionally, remove the temporary .parquet file
    os.remove(parquet_filename)
    print(f"Removed temporary file: {parquet_filename}")
    return None

## OVERALL FUNCTION USED FOR THE PREPROCESSING
def preprocessing(filename, save = False, demo = True):
    # Load the right data to a dataset
    period_data = load_data(filename)

    # Update the dataframe, especially take the minutes before 15:30 for the analysis
    df_pivot = period_data.copy()
    df_pivot = df_pivot.pivot_table(index='xltime', columns='stock', values='price', aggfunc='mean')
    df_pivot.index = pd.to_datetime(df_pivot.index)
    df_pivot = df_pivot[df_pivot.index.time < datetime.strptime('15:30', '%H:%M').time()]
    
    dates = df_pivot.copy()
    dates.index = pd.to_datetime(dates.index)
    # Get the first and last index (times)
    first_time = dates.index.min()
    last_time = dates.index.max()

    # Extract only the dates in YYYY-MM-DD format
    first_date = first_time.date()  # Or first_time.strftime('%Y-%m-%d')
    last_date = last_time.date()

    # Plot the minutes in each stock for the period
    daily_data_count = df_pivot.resample('D').count() 
    sns.set(font_scale=1)

    pivot_table = daily_data_count.transpose()

    plt.figure(figsize=(25, 12))
    ax = sns.heatmap(pivot_table, cmap="BuPu", linewidths=0, linecolor=None, cbar_kws={"shrink": .5}, yticklabels=True, xticklabels=False)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_xlabel(f'Time from {first_date} to {last_date}', fontsize=15)
    ax.set_ylabel('Stocks', fontsize=15)

    plt.title(f'Heatmap of number of data per day for each stock from {first_date} to {last_date}', fontsize=20)

    plt.show()
    
    # For each stock plot the ratio of missing values compared to the total maximum 
    missing_min = calculate_ratio_missing_minutes_per_stocks(period_data)
    critical_stocks = {key: value for key, value in missing_min.items() if value > 25}
    perc_high = len(critical_stocks)
    print("We remove the " + str(perc_high) + " stocks with more than 25% missing minutes: " + ", ".join(critical_stocks.keys()))

    stocks_to_remove = list(critical_stocks.keys())
    period_data = period_data[~period_data['stock'].isin(stocks_to_remove)]

    num_unique_stocks = period_data['stock'].nunique()
    print(f"We are now working with : {num_unique_stocks} stocks.")

    period_nans = fill_bid_price_by_day_and_stock(period_data)
    period_nans = fill_missing_values_by_stock_and_date(period_nans)
    
    print("The missing minutes values were now filled.")

    if save and demo:
        parquet_filename = "data/cleaned_data_demo.parquet"
        tar_filename = "data/cleaned_data_demo.tar"
        save_to_parquet_and_tar(period_nans, parquet_filename, tar_filename) 

    if save and not(demo):
        parquet_filename = "data/cleaned_data.parquet"
        tar_filename = "data/cleaned_data.tar"
        save_to_parquet_and_tar(period_nans, parquet_filename, tar_filename)

    return period_nans