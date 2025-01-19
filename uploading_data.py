import pandas as pd
import pytz
import tarfile
import io
from data_exploration import *

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
# Careful: If the date does not exist for Ticker then returns None
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

## Create a Dataframe that contains all the events for a list of ticker between two dates
# Careful: If the ticker is None then consider all tickers available between the two dates
def load_all(start_date, end_date, save = False, compress_to_tar = False):  

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

            if compress_to_tar:
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

def fill_missing_minutes(group):
    # Generate the complete minute index for the group
    complete_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq='min')
    group = group.reindex(complete_index)
    group['bid_price'].apply(longest_nan_gap)

    # Iterate over the rows with missing values and fill them
    for idx in group[group.isnull().any(axis=1)].index:
        # Get the past 5 rows or all available previous rows
        past_data = group.loc[:idx].iloc[:-1].tail(5)

        # If there are no past data, skip filling (shouldn't happen in most cases)
        if past_data.empty:
            print("PAST DATA IS EMPTY FOR " + str(idx) + "FOR STOCK" + str(group['stock']))
            continue

        # Calculate the weights based on the total volume
        total_volume = past_data['bid_volume'] + past_data['ask_volume']
        weights = total_volume / total_volume.sum()

        # Compute the weighted average for each column to fill
        group.loc[idx, 'price'] = np.average(past_data['price'], weights=weights)

    return group

def fill_with_nans(group):
    # Generate the complete minute index for the group
    complete_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq='min')
    # Reindex the group to fill missing minutes
    group = group.reindex(complete_index)
    return group

# Function to find the longest gap of NaNs in a series
def longest_nan_gap(series):
    is_nan = series.isna()
    groups = (is_nan != is_nan.shift()).cumsum()  # Group consecutive values
    return is_nan.groupby(groups).sum().max()    # Get the max size of NaN groups



