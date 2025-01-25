from datetime import datetime
import pandas as pd
import pytz
import tarfile
import io
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import os
import re

## Count the total number of days of data available for each stock in a given year.
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
    
# Calculates missing days and the number of stocks with missing data per month for a given year,
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

# Create a dictionary of dates with their corresponding list of tickers.
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

def count_minutes_per_stock_per_months(month:int):
    # fill month with 0 to have two digits
    tar_file_path = f"data/period_data.tar"
    stock_minutes = {}
    max = 0 
    counter = 0

    # Number of expected minutes per day is 
    # Number of days is 211 

    # Extract the main daily tar file
    with tarfile.open(tar_file_path, "r") as month_tar:

        # for each parquet file in the month file
        day_filenames = month_tar.getmembers()

       

        # Iterate through the nested daily tar files
        for day_filename in day_filenames:
            
            if day_filename.isfile() and day_filename.name.endswith(f".parquet"):
                # open the corresponding parquet file
                
                print(day_filename)
                counter +=1


                # Read parquet file
                day_trades = pd.read_parquet(month_tar.extractfile(day_filename))

                
                # List of stock for that day 
                # TODO: change to stock 
                stocks = day_trades['ticker'].unique()
               

                # count the number of minutes for each stock
                for stock in stocks:
                    nbr_minutes = day_trades[day_trades['ticker'] == stock].shape[0]
                    
                    if nbr_minutes > max:
                        max = nbr_minutes
                        print(max)
                    if stock not in stock_minutes.keys():
                        stock_minutes[stock] = nbr_minutes
                    else: 
                        stock_minutes[stock] += nbr_minutes
    print(counter)

    return stock_minutes, max


def stats_per_stock(dictionary, max):

    # Number of days 
    nbr_days = 213

    num_stocks = len(dictionary)
    incomplete_stocks = sum(1 for minutes in dictionary.values() if minutes < max * nbr_days)
    incomplete_percentage = (incomplete_stocks / num_stocks) * 100
    
    print(f"There are {num_stocks} stocks.")
    print(f"{incomplete_stocks} of which have incomplete data (less than {max* nbr_days} minutes of data), corresponding to {incomplete_percentage:.2f}% of the stocks. \n")

def calculate_ratio_missing_minutes_per_stocks(dictionary, max): 
    # Number of days 
    nbr_days = 213

    for stock, minutes in dictionary.items():
        if minutes < max * nbr_days:
            print(f"Stock {stock} has {100 *(max * nbr_days - minutes)/ (max * nbr_days):.2f} % missing minutes of data.")
        else:
            print(f"Stock {stock} has complete data.")