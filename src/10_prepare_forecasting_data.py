import pandas as pd
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Load combined dataset
data = pd.read_csv(
    PROCESSED_DATA_DIR / "model_dataset.csv",
    index_col="DATE",
    parse_dates=True
)

# Create lagged predictors using the previous month's values
predictors = [
    "PAYROLL_GROWTH",
    "ICSA",
    "FEDFUNDS",
    "CPI_INFLATION"
]

for variable in predictors:
    data[f"{variable}_LAG1"] = data[variable].shift(1)

# Keep unemployment and lagged predictors
forecast_data = data[
    [
        "UNRATE",
        "PAYROLL_GROWTH_LAG1",
        "ICSA_LAG1",
        "FEDFUNDS_LAG1",
        "CPI_INFLATION_LAG1"
    ]
].copy()

# Save forecasting dataset
output_path = PROCESSED_DATA_DIR / "forecasting_dataset.csv"
forecast_data.to_csv(output_path)

# Display results
print("Forecasting dataset")
print(forecast_data.tail(15))
print()
print("Missing values:")
print(forecast_data.isna().sum())
print()
print(f"Saved to: {output_path}")