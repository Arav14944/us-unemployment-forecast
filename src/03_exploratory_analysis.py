import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"

unemployment = pd.read_csv(
    PROCESSED_DATA_DIR / "unemployment_rate_clean.csv",
    index_col="DATE",
    parse_dates=True
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6))
plt.plot(unemployment.index, unemployment["UNRATE"])

plt.title("US Unemployment Rate, 2000-2026")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "unemployment_rate.png", dpi=300)

unemployment_diff = unemployment["UNRATE"].diff().dropna()

adf_diff = adfuller(unemployment_diff)

print(f"ADF statistic (first difference): {adf_diff[0]:.4f}")
print(f"ADF p-value (first difference): {adf_diff[1]:.4f}")