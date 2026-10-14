import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "tables"

rng = np.random.default_rng(42)

# Load the combined monthly dataset
data = pd.read_csv(
    PROCESSED_DATA_DIR / "model_dataset.csv",
    index_col="DATE",
    parse_dates=True
).sort_index()

# Restore the complete monthly calendar
monthly_dates = pd.date_range(
    start=data.index.min(),
    end=data.index.max(),
    freq="MS"
)

data = data.reindex(monthly_dates)
data.index.name = "DATE"

# Create calendar-correct lags
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

# Identify the latest observed unemployment month
observed_unemployment = data["UNRATE"].dropna()
last_date = observed_unemployment.index[-1]
last_unemployment = float(observed_unemployment.iloc[-1])

# Train only on complete historical observations
train = data.dropna(subset=predictors + ["UNRATE"])

scaler = StandardScaler()
X_train = scaler.fit_transform(train[predictors])
y_train = train["UNRATE"]

model = LinearRegression()
model.fit(X_train, y_train)

# Get the latest available macroeconomic observations
# No partial observations from a later calendar month are used.
macro_variables = [
    "PAYROLL_GROWTH",
    "ICSA",
    "FEDFUNDS",
    "CPI_INFLATION"
]

latest_macro = {}

print("Predictor values used:")
for variable in macro_variables:
    available = data.loc[data.index <= last_date, variable].dropna()

    if available.empty:
        raise ValueError(f"No available observations for {variable}")

    latest_macro[variable] = float(available.iloc[-1])

    print(
        f"{variable}: {latest_macro[variable]:.4f} "
        f"(latest observation: {available.index[-1].date()})"
    )

# Load historical out-of-sample forecast errors
error_file = RESULTS_DIR / "macro_ar1_forecasts.csv"

historical_forecasts = pd.read_csv(
    error_file,
    index_col="DATE",
    parse_dates=True
)

forecast_errors = (
    historical_forecasts["Actual"]
    - historical_forecasts["Macro_AR1_Forecast"]
).dropna().to_numpy()

if len(forecast_errors) < 10:
    raise ValueError(
        "Too few historical forecast errors to estimate uncertainty."
    )

# Forecast October 2026 through January 2027
forecast_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=4,
    freq="MS"
)

n_simulations = 10000
simulated_paths = np.zeros(
    (n_simulations, len(forecast_dates))
)

# Simulate future unemployment paths
for simulation in range(n_simulations):

    previous_unemployment = last_unemployment

    for step, date in enumerate(forecast_dates):

        # Assume future macroeconomic predictors stay at
        # their latest available values.
        X_future = pd.DataFrame(
            [[
                previous_unemployment,
                latest_macro["PAYROLL_GROWTH"],
                latest_macro["ICSA"],
                latest_macro["FEDFUNDS"],
                latest_macro["CPI_INFLATION"]
            ]],
            columns=predictors
        )

        X_future_scaled = scaler.transform(X_future)

        point_forecast = float(
            model.predict(X_future_scaled)[0]
        )

        # Add a randomly sampled historical out-of-sample error
        error = rng.choice(forecast_errors)

        simulated_unemployment = point_forecast + error

        simulated_paths[simulation, step] = (
            simulated_unemployment
        )

        # Carry the simulated unemployment rate into
        # the next month's AR(1) prediction.
        previous_unemployment = simulated_unemployment

# Extract the required forecast months
target_dates = [
    pd.Timestamp("2026-11-01"),
    pd.Timestamp("2027-01-01")
]

results = []

for target_date in target_dates:

    step = forecast_dates.get_loc(target_date)
    simulated_values = simulated_paths[:, step]

    results.append({
        "Month": target_date.strftime("%Y-%m"),
        "P10": np.quantile(simulated_values, 0.10),
        "P50": np.quantile(simulated_values, 0.50),
        "P90": np.quantile(simulated_values, 0.90)
    })


forecast_results = pd.DataFrame(results).set_index("Month")

# Save results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_path = RESULTS_DIR / "final_forecasts.csv"

forecast_results.to_csv(output_path)

# Display results
print("\nFinal US unemployment forecasts (%)")
print(forecast_results.round(3))

print(f"\nLatest unemployment observation: {last_date.date()}")
print(f"Historical forecast errors used: {len(forecast_errors)}")
print(f"Simulated paths: {n_simulations}")
print(f"Saved to: {output_path}")