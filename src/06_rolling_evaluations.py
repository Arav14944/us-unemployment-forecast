import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "tables"

unemployment = pd.read_csv(
    PROCESSED_DATA_DIR / "unemployment_rate_clean.csv"
)
unemployment_values = unemployment["UNRATE"].reset_index(drop=True)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# data has 320 observations, we use 320 - 24 = 296 data points for training and forecast on next 24
test_size = 24
initial_train_size = len(unemployment) - test_size

# storing the data so the model can be trained on a rolling basis
# after say sept 2024 is realised, the model is trained on that data as well
actuals = []
naive_forecasts = []
ar1_forecasts = []
ar2_forecasts = []
ar3_forecasts = []
ar6_forecasts = []
ar12_forecasts = []

for i in range(initial_train_size, len(unemployment_values)):
    train = unemployment_values.iloc[:i]
    actual = unemployment_values.iloc[i]

# naive forecast here means the forecast for next month is the same as last month    
    naive_forecast = train.iloc[-1]
    naive_forecasts.append(naive_forecast)

    actuals.append(actual)

    ar1_model = AutoReg(
        train,
        lags=1,
        trend="c"
    ).fit()

    ar1_forecast = ar1_model.predict(
        start=len(train),
        end=len(train)
    ).iloc[0]

    ar1_forecasts.append(float(ar1_forecast))

    ar2_model = AutoReg(
        train,
        lags=2,
        trend="c"
    ).fit()

    ar2_forecast = ar2_model.predict(
        start=len(train),
        end=len(train)
    ).iloc[0]

    ar2_forecasts.append(float(ar2_forecast))

        # AR(3)
    ar3_model = AutoReg(
        train,
        lags=3,
        trend="c"
    ).fit()

    ar3_forecast = ar3_model.predict(
        start=len(train),
        end=len(train),
        dynamic=False
    ).iloc[0]

    ar3_forecasts.append(float(ar3_forecast))


    # AR(6)
    ar6_model = AutoReg(
        train,
        lags=6,
        trend="c"
    ).fit()

    ar6_forecast = ar6_model.predict(
        start=len(train),
        end=len(train),
        dynamic=False
    ).iloc[0]

    ar6_forecasts.append(float(ar6_forecast))


    # AR(12)
    ar12_model = AutoReg(
        train,
        lags=12,
        trend="c"
    ).fit()

    ar12_forecast = ar12_model.predict(
        start=len(train),
        end=len(train),
        dynamic=False
    ).iloc[0]

    ar12_forecasts.append(float(ar12_forecast))

# Convert forecasts and actuals to arrays
actuals = np.array(actuals)
naive_forecasts = np.array(naive_forecasts)
ar1_forecasts = np.array(ar1_forecasts)
ar2_forecasts = np.array(ar2_forecasts)
ar3_forecasts = np.array(ar3_forecasts)
ar6_forecasts = np.array(ar6_forecasts)
ar12_forecasts = np.array(ar12_forecasts)


results = pd.DataFrame({
    "Model": ["Naive", "AR(1)", "AR(2)", "AR(3)", "AR(6)", "AR(12)"],
    "MAE": [
        np.mean(np.abs(actuals - naive_forecasts)),
        np.mean(np.abs(actuals - ar1_forecasts)),
        np.mean(np.abs(actuals - ar2_forecasts)),
        np.mean(np.abs(actuals - ar3_forecasts)),
        np.mean(np.abs(actuals - ar6_forecasts)),
        np.mean(np.abs(actuals - ar12_forecasts))
    ],
    "RMSE": [
        np.sqrt(np.mean((actuals - naive_forecasts) ** 2)),
        np.sqrt(np.mean((actuals - ar1_forecasts) ** 2)),
        np.sqrt(np.mean((actuals - ar2_forecasts) ** 2)),
        np.sqrt(np.mean((actuals - ar3_forecasts) ** 2)),
        np.sqrt(np.mean((actuals - ar6_forecasts) ** 2)),
        np.sqrt(np.mean((actuals - ar12_forecasts) ** 2))
    ]
})

print(results)