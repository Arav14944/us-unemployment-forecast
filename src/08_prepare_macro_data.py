import pandas as pd
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Load monthly macroeconomic data
macro_data = pd.read_csv(
    RAW_DATA_DIR / "macro_data.csv",
    index_col="DATE",
    parse_dates=True
)

# Create transformed variables
macro_data["PAYROLL_GROWTH"] = (
    macro_data["PAYEMS"].pct_change() * 100
)

macro_data["CPI_INFLATION"] = (
    macro_data["CPIAUCSL"].pct_change(periods=12) * 100
)

# Keep the four modelling variables
prepared_data = macro_data[
    [
        "PAYROLL_GROWTH",
        "ICSA",
        "FEDFUNDS",
        "CPI_INFLATION"
    ]
].copy()


# Save processed data
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DATA_DIR / "macro_data_prepared.csv"

prepared_data.to_csv(output_path)


# Display results
print("Prepared Monthly Macroeconomic Variables")
print(prepared_data.tail(15))
print()
print("Missing values by variable:")
print(prepared_data.isna().sum())
print()
print(f"Saved to: {output_path}")
