# starting off with an ar(1) model, u(t) = a + b*u(t-1) + e(t)

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg

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

ar1_model = AutoReg(
    train["UNRATE"],
    lags=1,
    trend="c"
).fit()

ar1_forecast = ar1_model.predict(
    start=len(train),
    end=len(unemployment) - 1
)

ar1_forecast.index = test.index

ar1_mae = np.mean(np.abs(test["UNRATE"] - ar1_forecast))
ar1_rmse = np.sqrt(np.mean((test["UNRATE"] - ar1_forecast) ** 2))

# AR(2) model
ar2_model = AutoReg(
    train["UNRATE"],
    lags=2,
    trend="c"
).fit()

ar2_forecast = ar2_model.predict(
    start=len(train),
    end=len(unemployment) - 1
)

ar2_forecast.index = test.index

ar2_mae = np.mean(np.abs(test["UNRATE"] - ar2_forecast))
ar2_rmse = np.sqrt(np.mean((test["UNRATE"] - ar2_forecast) ** 2))

print(f"AR(1) MAE: {ar1_mae:.4f}")
print(f"AR(1) RMSE: {ar1_rmse:.4f}")

print(f"AR(2) MAE: {ar2_mae:.4f}")
print(f"AR(2) RMSE: {ar2_rmse:.4f}")