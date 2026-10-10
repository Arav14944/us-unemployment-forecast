import pandas as pd
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Load cleaned unemployment data
unemployment = pd.read_csv(
    PROCESSED_DATA_DIR / "unemployment_rate_clean.csv",
    index_col="DATE",
    parse_dates=True
)

# Load prepared macroeconomic data
macro_data = pd.read_csv(
    PROCESSED_DATA_DIR / "macro_data_prepared.csv",
    index_col="DATE",
    parse_dates=True
)

# Combine datasets by month
model_data = unemployment.join(
    macro_data,
    how="inner"
)

# Save the combined dataset
output_path = PROCESSED_DATA_DIR / "model_dataset.csv"
model_data.to_csv(output_path)

# Display results
print("Combined modelling dataset")
print(model_data.tail(15))
print()
print(f"Observations: {len(model_data)}")
print("\nMissing values by variable:")
print(model_data.isna().sum())
print()
print(f"Saved to: {output_path}")
