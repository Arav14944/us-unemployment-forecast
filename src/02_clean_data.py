import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


unemployment = pd.read_csv(
    RAW_DATA_DIR / "unemployment_rate.csv",
    index_col="DATE",
    parse_dates=True
)

unemployment = unemployment.dropna()

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

unemployment.to_csv(
    PROCESSED_DATA_DIR / "unemployment_rate_clean.csv"
)