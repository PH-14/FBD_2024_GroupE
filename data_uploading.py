import pandas as pd
import pytz
import tarfile
import io
from datetime import datetime
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from clustering import *
import os

## Counts the total number of days of data available for each stock in a given year.
def count_days_per_stock_per_year(year):
    
    tar_file_path = f"data/ETFs-{year}.tar"
    stock_days = {}
    
    # Extract the main yearly tar file
    with tarfile.open(tar_file_path, "r") as year_tar:
        stock_filenames = year_tar.getmembers()

        # Iterate through the nested stock tar files
        for stock_file in stock_filenames:
            if stock_file.isfile() and stock_file.name.endswith(f"_{year}.tar"):
                stock_tar_name = os.path.basename(stock_file.name)
                ticker = stock_tar_name.split('.')[0]  # Extract the ticker from the file name

                # The stock SOYB is missing a file for one day, we therefore, do not use this stock
                if(ticker == 'SOYB'):
                    continue
                
                target_stock_file = year_tar.extractfile(stock_file)

                # Open the stock tar file
                with tarfile.open(fileobj=target_stock_file) as stock_tar:
                    dates_filenames = stock_tar.getmembers()
                    for file in dates_filenames:
                        if file.isfile() and file.name.endswith(".parquet"):
                            # Extract the date from the parquet file name
                            parquet_name = os.path.basename(file.name)
                            date_part = '-'.join(parquet_name.split('-')[:3])
                            # Initialize the ticker in the dictionary if not present
                            if ticker not in stock_days:
                                stock_days[ticker] = set()
                            
                            # Add the date to the ticker's set
                            stock_days[ticker].add(date_part)

    # Convert sets to counts
    stock_days_count = {ticker: len(dates) for ticker, dates in stock_days.items()}
    
    return stock_days_count, stock_days

## Computes and prints the stats for each year
# How many stocks ? How many have missing dates ? 
def stats_per_year(dictionary, year):

    opening_days = 252
    # For some reason, market opened for 251 days in 2012
    if(year == 2012):
        opening_days = 251

    num_tickers = len(dictionary)

    incomplete_stocks = sum(1 for days in dictionary.values() if days < opening_days)
    incomplete_percentage = (incomplete_stocks / num_tickers) * 100
    
    print("YEAR " + str(year))
    print(f"There are {num_tickers} stocks in the year {str(year)}.")
    print(f"{incomplete_stocks} of which have incomplete data (less than {opening_days} days of data), corresponding to {incomplete_percentage:.2f}% of the stocks. \n")
    
## Calculates missing days and the number of stocks with missing data per month for a given year,
# based on the maximum number of available days for any stock in that month. Plots the results.
def calculate_missing_days_stats(yearly_days, year):
    """
    Returns:
    - expected_days_per_month (dict): Maximum available days for any stock in each month (YYYY-MM).
    - total_missing_days_per_month (dict): Total missing days across all stocks for each month.
    - stocks_with_missing_data_per_month (dict): Number of stocks with missing data for each month.
    """
    # Extract stock days for the specified year
    stock_days_count = yearly_days[year]

    stocks_with_missing_data_per_month = {}

    # Group days by month for each stock
    monthly_stock_days = defaultdict(lambda: defaultdict(set))
    for ticker, days in stock_days_count.items():
        stocks_with_missing_data_per_month[ticker] = set()
        for day in days:
            month = day[:7]  # Extract YYYY-MM
            monthly_stock_days[month][ticker].add(day)
    
    # Determine the expected days per month (based on max for any stock)
    expected_days_per_month = {}
    for month, stocks in monthly_stock_days.items():
        max_days = max(len(days) for days in stocks.values())
        expected_days_per_month[month] = max_days

    # Calculate total missing days and number of stocks with missing data per month
    total_missing_days_per_month = defaultdict(int)
    stocks_with_missing_data_per_month_count = defaultdict(int)
    

    for month, stocks in monthly_stock_days.items():
        expected_days = expected_days_per_month[month]
        for ticker, days in stocks.items():
            available_days = len(days)
            missing_days = expected_days - available_days
            if missing_days > 0:
                total_missing_days_per_month[month] += missing_days
                stocks_with_missing_data_per_month_count[month] += 1
                stocks_with_missing_data_per_month[ticker].add(month)
                
    # Prepare data for plotting
    months = sorted(expected_days_per_month.keys())
    missing_days = [total_missing_days_per_month.get(month, 0) for month in months]
    stocks_with_missing = [stocks_with_missing_data_per_month_count.get(month, 0) for month in months]
    max_days_per_month = [expected_days_per_month.get(month, 0) for month in months]

    # Plot results with the addition of maximum days per month
    plt.figure(figsize=(12, 6))

    # Plot total missing days as bars
    plt.bar(months, missing_days, label="Total Missing Days", alpha=0.7)

    # Plot number of stocks with missing data as a line
    plt.plot(months, stocks_with_missing, label="Stocks with Missing Data", color="red", marker="o")
    # Add numbers next to each point
    for month, value in zip(months, stocks_with_missing):
        if value > 0:  # Only add text for non-zero values
            plt.text(month, value + 1, f"{value}", ha='center', fontsize=10, color="black")


    # Plot maximum days for each month as a line
    plt.plot(months, max_days_per_month, label="Maximum Days in Month", color="blue", linestyle="--", marker="x")

    # Formatting the plot
    plt.title(f"Missing Data Statistics for {year} (Including Max Days Per Month)")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    filtered_stocks_with_missing_data_per_month = {
        key: value
        for key, value in stocks_with_missing_data_per_month.items()
        if len(value) > 0  # Explicitly checks if the list is non-empty
    }
    return filtered_stocks_with_missing_data_per_month

# Create a dictionary of dates with their corresponding list of stocks.
def filter_days_for_period(start_date, end_date):
    """
    Returns:
    - tickers_per_date (dict): Dictionary with dates as keys and lists of tickers as values.
                                A date is not included if no tickers are available.
    """
    
    start_year = start_date.year  # First 4 characters for the year
    end_year = end_date.year
    
    # Generate the range of years
    years = list(range(start_year, end_year + 1))

    yearly_count_dict = {}
    yearly_days = {}

    for year in years:
        count , days = count_days_per_stock_per_year(year)
        yearly_count_dict[year] = dict(sorted(count.items()))
        yearly_days[year] = dict(sorted(days.items()))

    # Convert start and end dates to datetime objects
    #start = datetime.strptime(start_date, "%Y-%m-%d")
    #end = datetime.strptime(end_date, "%Y-%m-%d")

    # Initialize the dictionary for tickers per date
    tickers_per_date = defaultdict(list)

    # Iterate through the years and filter dates
    for year, stocks in yearly_days.items():
        for ticker, dates in stocks.items():
            # Filter dates within the specified period
            for date in dates:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if start_date <= date_obj <= end_date:
                    tickers_per_date[date].append(ticker)

    # Return tickers per date, sorted by date
    return dict(sorted(tickers_per_date.items()))

## Function to clean timestamp (From Excel formatting to usual dates)
def clean_timestamp(df, timezone = 'America/New_York'):
    df.index = pd.to_datetime(df["xltime"],unit="d",origin="1899-12-30",utc=True)
    df.index = df.index.tz_convert(timezone)  # .P stands for Arca, which is based at New York
    df.drop(columns="xltime",inplace=True)

    # Verify that the time are within the market opening hours
    df=df.between_time("09:30:00","16:00:00")
    df.index = df.index.floor('min')
    return df

## Function to verify types of data
def clean_prices(df, columns_to_check):
    for column in columns_to_check:
        df[column].apply(lambda value: float(value) if pd.api.types.is_number(value) else np.nan)
    return df

## CLEANING THE BBO DATAFRAME
# Careful: If the date does not exist for stock then returns None
def load_bbo(ticker, date, save = False):

    year = date.strftime('%Y')    
    file_path_year = 'data/' + 'ETFs-' + year + '.tar'
    date_str = date.strftime('%Y-%m-%d')

    # Read the .parquet file into a Pandas DataFrame
    with tarfile.open(file_path_year, 'r') as tar:
        # Get the list of all files in the tar archive
        filenames = sorted(tar.getmembers(), key=lambda x: x.name)
        ticker_bbo = ticker + ".P_bbo"
        target_filename = next((m for m in filenames if ticker_bbo in m.name), None)
        if(target_filename == None):
            return None
        target_file = tar.extractfile(target_filename)
        with tarfile.open(fileobj=target_file) as inner_tar:
            members_bbo = sorted(inner_tar.getmembers(), key=lambda x: x.name)
            target_member = next((m for m in members_bbo if date_str in m.name), None)
            if(target_member == None):
                return None
            # Read the parquet file into a DataFrame
            df_bbo_raw = pd.read_parquet(io.BytesIO(inner_tar.extractfile(target_member).read()))
    
    df_bbo_cleaned = clean_timestamp(df_bbo_raw)

    columns_to_check_bbo = ["bid-price", "bid-volume", "ask-price", "ask-volume"]
    df_bbo_cleaned = clean_prices(df_bbo_cleaned, columns_to_check_bbo)

    # Remove duplicate time: take the mean of the prices and the sum of the volumes
    df_bbo_cleaned=df_bbo_cleaned.groupby(df_bbo_cleaned.index).agg(bid_price=pd.NamedAgg(column='bid-price', aggfunc='mean'),
                                                                bid_volume=pd.NamedAgg(column='bid-volume', aggfunc='sum'),
                                                                ask_price=pd.NamedAgg(column='ask-price', aggfunc='mean'),
                                                                ask_volume=pd.NamedAgg(column='ask-volume', aggfunc='sum'))
    
    # Adds a column containing the average prices (based on the prices and volumes of ask and bid)
    df_bbo_cleaned['price'] = ((df_bbo_cleaned['bid_volume'] * df_bbo_cleaned['bid_price'] + df_bbo_cleaned['ask_volume'] * df_bbo_cleaned['ask_price']) /
                                    (df_bbo_cleaned['bid_volume'] + df_bbo_cleaned['ask_volume']))
    
    return df_bbo_cleaned

## CLEANING THE TRADE DATAFRAME OBSOLETE BC MOSTLY NANS
# Careful: If the date does not exist for Ticker then returns None
#def load_trade(ticker, date, save = False):

    year = date.strftime('%Y')    
    file_path_year = 'data/' + 'ETFs-' + year + '.tar'
    date_str = date.strftime('%Y-%m-%d')

    with tarfile.open(file_path_year, 'r') as tar:
        # Get the list of all files in the tar archive
        filenames = sorted(tar.getmembers(), key=lambda x: x.name)
        ticker_trade = ticker + ".P_trade"
        target_filename = next((m for m in filenames if ticker_trade in m.name), None)
        if(target_filename == None):
            return None
        target_file = tar.extractfile(target_filename)
        with tarfile.open(fileobj=target_file) as inner_tar:
            members_trade = sorted(inner_tar.getmembers(), key=lambda x: x.name)
            target_member = next((m for m in members_trade if date_str in m.name), None)
            if(target_member == None):
                return None
            # Read the parquet file into a DataFrame
            df_trade_raw = pd.read_parquet(io.BytesIO(inner_tar.extractfile(target_member).read()))

    df_trade_cleaned = clean_timestamp(df_trade_raw)

    # Keep only the traditional trades
    df_trade_cleaned = df_trade_cleaned[df_trade_cleaned["trade-stringflag"]=="uncategorized"]
    df_trade_cleaned.drop(columns=["trade-rawflag","trade-stringflag"],axis=1,inplace=True)

    # Remove duplicate time: take the mean of the prices and the sum of the volumes
    columns_to_check_trade = ["trade-price", "trade-volume"]
    df_trade_cleaned = clean_prices(df_trade_cleaned, columns_to_check_trade)
    df_trade_cleaned=df_trade_cleaned.groupby(df_trade_cleaned.index).agg(trade_price=pd.NamedAgg(column='trade-price', aggfunc='mean'),
                                                                      trade_volume=pd.NamedAgg(column='trade-volume', aggfunc='sum'))
    return df_trade_cleaned

## OBSOLETE BC TRADE IS MOSTLY NANS
## Create a Dataframe for all events on a given ticker on a given date
#def merge_bbo_trade(ticker, date, save = False):
    df_trade = load_trade(ticker, date)
    df_bbo = load_bbo(ticker, date)
    if(df_bbo is None):
        return None
    
    if (df_trade is None):
        df_bbo["trade_price"]=np.nan
        df_bbo["trade_volume"]=np.nan
        return df_bbo
        
    df_events = df_trade.join(df_bbo,how="outer")
    return df_events

## Create a Dataframe that contains all the events for a ticker between two dates
#def load_all_dates(ticker, start_date, end_date):
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date)
    dates.to_pydatetime().tolist()

    combined_df = pd.DataFrame()

    for date in dates:
        df_date = merge_bbo_trade(ticker, date)
        if (df_date is not None):
            combined_df = pd.concat([combined_df, df_date], ignore_index=True)
    
    return combined_df

## Create a Dataframe that contains all the events between two dates
def load_all(start_date, end_date, save = False, compress_tar = False):  

    days_for_period = filter_days_for_period(start_date, end_date)    

    for date_str, tickers in days_for_period.items():
        print(f"Processing date: {date_str}, Tickers: {tickers}")
        date = pd.to_datetime(date_str)
        
        # Initialize a list to hold DataFrames for the date
        daily_data = []
        
        for ticker in tickers:
            try:
                # Call the merge function to get data for the ticker at the date
                df = load_bbo(ticker, date)
                df["stock"] = ticker
                if df is not None and not df.empty:
                    daily_data.append(df)
            except Exception as e:
                print(f"Error processing ticker {ticker} for date {date}: {e}")
        
        # Combine all DataFrames for the date
        if daily_data:
            combined_df = pd.concat(daily_data)

        if save:
            output_dir = "data/dates"

            output_path = os.path.join(output_dir, f"{date_str}.parquet")
            combined_df.to_parquet(output_path)

    if compress_tar:
        input_dir = "data/dates/"
        output_file = "data/period_data.tar"
        compress_to_tar(input_dir, output_file)

    return combined_df

def compress_to_tar(input_folder, output_file):
    """
    Compress an entire folder into a .tar file.

    Arguments:
    - input_folder (str): Path to the folder to compress.
    - output_tar (str): Path to the resulting .tar file.
    """
    # Ensure the input folder exists
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist or is not a directory.")

    print(f"Compressing folder '{input_folder}' into '{output_file}'...")
    
    # Create a .tar file
    with tarfile.open(output_file, "w") as tar:
        tar.add(input_folder, arcname=os.path.basename(input_folder))
    
    print(f"Folder successfully compressed into '{output_file}'.")
    return None
