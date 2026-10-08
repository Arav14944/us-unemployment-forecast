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