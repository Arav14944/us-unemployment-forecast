import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"

# Load cleaned unemployment data
unemployment = pd.read_csv(
    PROCESSED_DATA_DIR / "unemployment_rate_clean.csv",
    index_col="DATE",
    parse_dates=True
)

# Create figures directory
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Descriptive statistics
print(f"Start date: {unemployment.index.min()}")
print(f"End date: {unemployment.index.max()}")
print(f"Mean unemployment rate: {unemployment['UNRATE'].mean():.2f}%")
print(f"Minimum unemployment rate: {unemployment['UNRATE'].min():.2f}%")
print(f"Maximum unemployment rate: {unemployment['UNRATE'].max():.2f}%")

# Plot unemployment rate
plt.figure(figsize=(12, 6))
plt.plot(unemployment.index, unemployment["UNRATE"])
plt.title("US Unemployment Rate, 2000-2026")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "unemployment_rate.png", dpi=300)
plt.show()
plt.close()

# Augmented Dickey-Fuller test
adf_result = adfuller(unemployment["UNRATE"])

print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"ADF p-value: {adf_result[1]:.4f}")

# Autocorrelation function
plt.figure(figsize=(12, 6))
plot_acf(unemployment["UNRATE"], lags=36)
plt.title("ACF of US Unemployment Rate")
plt.xlabel("Lag (months)")
plt.ylabel("Autocorrelation")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "acf_unemployment.png", dpi=300)
plt.show()
plt.close()

# Partial autocorrelation function
plt.figure(figsize=(12, 6))
plot_pacf(unemployment["UNRATE"], lags=36)
plt.title("PACF of US Unemployment Rate")
plt.xlabel("Lag (months)")
plt.ylabel("Partial Autocorrelation")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "pacf_unemployment.png", dpi=300)
plt.show()
plt.close()

# ADF test on first difference
unemployment_diff = unemployment["UNRATE"].diff().dropna()

adf_diff = adfuller(unemployment_diff)

print(f"ADF statistic (first difference): {adf_diff[0]:.4f}")
print(f"ADF p-value (first difference): {adf_diff[1]:.4f}")