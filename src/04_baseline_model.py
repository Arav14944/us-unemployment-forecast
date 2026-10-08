import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "tables"

unemployment = pd.read_csv(
    PROCESSED_DATA_DIR / "unemployment_rate_clean.csv",
    index_col="DATE",
    parse_dates=True
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

test_size = 24

train = unemployment.iloc[:-test_size]
test = unemployment.iloc[-test_size:]

naive_forecast = test["UNRATE"].shift(1)
naive_forecast.iloc[0] = train["UNRATE"].iloc[-1]

# mean absolute error, gives the average size of our forecasting error
mae = np.mean(np.abs(test["UNRATE"] - naive_forecast))

# root mean squared error - squared errors
rmse = np.sqrt(np.mean((test["UNRATE"] - naive_forecast) ** 2))

results = pd.DataFrame({
    "Model": ["Naive"],
    "MAE": [mae],
    "RMSE": [rmse]
})

results.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)