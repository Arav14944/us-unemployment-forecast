import pandas as pd
from pandas_datareader import data as web
from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# download us unemployment data from fred
unemployment = web.DataReader(
    "UNRATE",
    "fred",
    start = "2000-01-01"
)

# save raw data
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

unemployment.to_csv(
    RAW_DATA_DIR / "unemployment_rate.csv"
)
