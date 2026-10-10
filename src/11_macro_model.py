import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "tables"

# Load the forecasting dataset
data = pd.read_csv(
    PROCESSED_DATA_DIR / "forecasting_dataset.csv",
    index_col="DATE",
    parse_dates=True
)

predictors = [
    "PAYROLL_GROWTH_LAG1",
    "ICSA_LAG1",
    "FEDFUNDS_LAG1",
    "CPI_INFLATION_LAG1"
]

# Keep observations with complete predictors
data = data.dropna(subset=predictors + ["UNRATE"])

# Use the final 24 eligible observations for evaluation
test_size = 24
initial_train_size = len(data) - test_size

actuals = []
naive_forecasts = []
macro_forecasts = []

# Expanding-window one-step-ahead evaluation
for i in range(initial_train_size, len(data)):

    train = data.iloc[:i]
    test = data.iloc[i]

    # Naive forecast: latest observed unemployment rate
    naive_forecast = train["UNRATE"].iloc[-1]
    naive_forecasts.append(naive_forecast)

    # Fit scaling only on the training data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[predictors])
    X_test = scaler.transform(
        test[predictors].to_frame().T
    )

    y_train = train["UNRATE"]

    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    macro_forecast = model.predict(X_test)[0]
    macro_forecasts.append(macro_forecast)

    actuals.append(test["UNRATE"])

# Convert to arrays
actuals = np.array(actuals)
naive_forecasts = np.array(naive_forecasts)
macro_forecasts = np.array(macro_forecasts)

# Calculate evaluation metrics
results = pd.DataFrame({
    "Model": ["Naive", "Macroeconomic Regression"],
    "MAE": [
        np.mean(np.abs(actuals - naive_forecasts)),
        np.mean(np.abs(actuals - macro_forecasts))
    ],
    "RMSE": [
        np.sqrt(np.mean((actuals - naive_forecasts) ** 2)),
        np.sqrt(np.mean((actuals - macro_forecasts) ** 2))
    ]
})

# Save results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results.to_csv(
    RESULTS_DIR / "macro_model_comparison.csv",
    index=False
)

print(results)
