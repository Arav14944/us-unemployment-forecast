import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "tables"

# Load combined monthly dataset
data = pd.read_csv(
    PROCESSED_DATA_DIR / "model_dataset.csv",
    index_col="DATE",
    parse_dates=True
).sort_index()

# Restore the complete monthly calendar.
# Missing months remain missing; no observations are invented.
monthly_dates = pd.date_range(
    start=data.index.min(),
    end=data.index.max(),
    freq="MS"
)

data = data.reindex(monthly_dates)
data.index.name = "DATE"

# Create calendar-correct one-month lags
data["UNRATE_LAG1"] = data["UNRATE"].shift(1)
data["PAYROLL_GROWTH_LAG1"] = data["PAYROLL_GROWTH"].shift(1)
data["ICSA_LAG1"] = data["ICSA"].shift(1)
data["FEDFUNDS_LAG1"] = data["FEDFUNDS"].shift(1)
data["CPI_INFLATION_LAG1"] = data["CPI_INFLATION"].shift(1)

# Define predictors
predictors = [
    "UNRATE_LAG1",
    "PAYROLL_GROWTH_LAG1",
    "ICSA_LAG1",
    "FEDFUNDS_LAG1",
    "CPI_INFLATION_LAG1"
]

# Evaluate the final 24 calendar months
test_dates = pd.date_range(
    end=data.index.max(),
    periods=24,
    freq="MS"
)

actuals = []
naive_forecasts = []
macro_ar1_forecasts = []
evaluation_dates = []
skipped_dates = []

# Expanding-window one-month-ahead evaluation
for date in test_dates:

    # Target month must have an observed unemployment rate
    if pd.isna(data.loc[date, "UNRATE"]):
        skipped_dates.append((date, "Unemployment unavailable"))
        continue

    # The previous calendar month's unemployment must be observed
    if pd.isna(data.loc[date, "UNRATE_LAG1"]):
        skipped_dates.append((date, "Previous month's unemployment unavailable"))
        continue

    # All predictors must be available
    if data.loc[date, predictors].isna().any():
        skipped_dates.append((date, "Missing predictor"))
        continue

    # Use only observations dated before the forecast month
    train = data.loc[data.index < date].copy()

    # Remove incomplete historical training observations
    train = train.dropna(subset=predictors + ["UNRATE"])

    test = data.loc[[date]]

    # Naive forecast: previous calendar month's unemployment
    naive_forecast = float(test["UNRATE_LAG1"].iloc[0])

    # Standardise predictors using training data only
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[predictors])
    X_test = scaler.transform(test[predictors])

    y_train = train["UNRATE"]

    # Fit macroeconomic regression with AR(1) unemployment
    model = LinearRegression()
    model.fit(X_train, y_train)

    macro_forecast = float(model.predict(X_test)[0])

    # Store forecasts for the same target month
    actuals.append(float(test["UNRATE"].iloc[0]))
    naive_forecasts.append(naive_forecast)
    macro_ar1_forecasts.append(macro_forecast)
    evaluation_dates.append(date)

# Convert forecasts to arrays
actuals = np.array(actuals)
naive_forecasts = np.array(naive_forecasts)
macro_ar1_forecasts = np.array(macro_ar1_forecasts)

# Calculate evaluation metrics
results = pd.DataFrame({
    "Model": ["Naive", "Macro Regression + AR(1)"],
    "MAE": [
        np.mean(np.abs(actuals - naive_forecasts)),
        np.mean(np.abs(actuals - macro_ar1_forecasts))
    ],
    "RMSE": [
        np.sqrt(np.mean((actuals - naive_forecasts) ** 2)),
        np.sqrt(np.mean((actuals - macro_ar1_forecasts) ** 2))
    ]
})

# Save model comparison
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results.to_csv(
    RESULTS_DIR / "macro_ar1_model_comparison.csv",
    index=False
)

# Save individual forecasts for verification
forecast_results = pd.DataFrame({
    "Actual": actuals,
    "Naive_Forecast": naive_forecasts,
    "Macro_AR1_Forecast": macro_ar1_forecasts
}, index=pd.DatetimeIndex(evaluation_dates, name="DATE"))

forecast_results.to_csv(
    RESULTS_DIR / "macro_ar1_forecasts.csv"
)

# Display results
print("Model comparison")
print(results.round(6))

print(f"\nEligible evaluation observations: {len(actuals)}")

if evaluation_dates:
    print(f"First forecast date: {evaluation_dates[0].date()}")
    print(f"Last forecast date: {evaluation_dates[-1].date()}")

print("\nSkipped dates:")
for date, reason in skipped_dates:
    print(f"{date.date()}: {reason}")

print("\nResults saved to:")
print("results/tables/macro_ar1_model_comparison.csv")
print("results/tables/macro_ar1_forecasts.csv")
