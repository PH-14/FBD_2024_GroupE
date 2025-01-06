# FBD_2024_GroupE
EPFL Financial Big Data final project

## Requirements

Use the following command to install the requirements of this project:

'''
pip install -r requirements.txt
'''

## Data pre-processing

To create training and validation sets on ETF data, place the ETF tar archive in the 'data/' directory and use the following command:

'''
python data_preprocessing.py data/ETFs-2012.tar N T "16:00"
'''

Where N is the number of stocks to include and T the number of days (max 1 year for now) and "16:00" is the cutoff time.
This will create two files: 'data/train_data_N_T_2012.parquet' and 'data/val_data_N_T_2012.parquet' (an example is provided for N=10 and T=100). 
The format of the dataset is:


| **Column Name**    | **Description**                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| `trade-price`      | The price at which the trade occurred.                                           |
| `trade-volume`     | The volume of shares traded in this transaction.                                |
| `trade-stringflag` | The string flag indicating the type of trade (e.g., auction, marketclosed).     |
| `Timestamp`        | The timestamp of the trade (in datetime format).                                |
| `Stock`            | The ticker symbol of the stock.                                                  |
| `Date`             | The date of the trade (in YYYY-MM-DD format).                                   |
| `Time_of_day`      | The time portion of the `Timestamp` indicating when the trade happened.         |
| `Before_XX:00`     | Boolean indicating whether the trade occurred before XX o'clock (`True` if before, `False` if after). |


