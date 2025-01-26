# FBD_2024_GroupE
EPFL Financial Big Data final project

## Requirements

Use the following command to install the requirements of this project:

```
pip install -r requirements.txt
```

## Data Exploration

To explore the distribution of data over the years, upload all the ETFs-"year".tar files in the "data/" directory, then, run the "exploring.ipynb" file. 

This allows us to observe the distribution of stocks within the years and months available. 

## Data Uploading 

Running the file "uploading.ipynb" will create a new .tar file named "period_data.tar" that contains .parquet files for each day of a chosen period. 

## Data pre-processing

To create training and validation sets on ETF data, place the ETF tar archive in the 'data/' directory and use the following command:

```
python data_preprocessing.py data/ETFs-2012.tar N T "16:00"
```

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

## Structure of the repository 

FBD_project/
│
├── [data_uploading.py](data_uploading.py)                       # Functions to explore the provided data & upload
├── [data_preprocessing.py](data_preprocessing.py)              # Various utils functions
├── [data_preprocessing.py](data_preprocessing.py)              # Functions to further clean the data for pre-processing
├── [market_state_strategy.py](market_state_strategy.py)        # Functions to implement the clustering and the strategies
├── [data_cleaning.ipynb](data_cleaning.ipynb)                  # Notebook to clean the data
├── [data_preprocessing.ipynb](data_preprocessing.ipynb)        # Notebook of the preprocessing of the data
├── [data_observation.ipynb](data_observation.ipynb)            # Notebook to observe the clean and processed data and to compute statistics 
├── [strateg_test.ipynb](strateg_test.ipynb)                    # Notebook containing all the strategies and their performance
├── ResearchNotebooks/                                          # Folder with deprecated notebooks used for exploration and various tests
|
├── [requirements.txt](requirements.txt)                        # Requirement files
|
└── README.md 