
import pandas as pd
from pandas_datareader import data as web
from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


# FRED series to download
series = [
    "PAYEMS",     # Nonfarm payroll employment
    "ICSA",       # Initial jobless claims
    "FEDFUNDS",   # Federal Funds Rate
    "CPIAUCSL"    # Consumer Price Index
]


# Download data from FRED
macro_data = web.DataReader(
    series,
    "fred",
    start="2000-01-01"
)


# Convert all variables to monthly frequency
# Weekly initial claims are converted to monthly averages.
macro_data = macro_data.resample("MS").mean()


# Create output directory if necessary
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Save raw macroeconomic data
output_path = RAW_DATA_DIR / "macro_data.csv"

macro_data.to_csv(output_path)


# Display results
print("Monthly Macroeconomic Data")
print(macro_data.tail(12))
print()
print(f"Observations: {len(macro_data)}")
print(f"Missing values by variable:\n{macro_data.isna().sum()}")
print()
print(f"Data saved to: {output_path}")
