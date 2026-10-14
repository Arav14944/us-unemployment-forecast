US Unemployment Rate Forecasting

Overview

This project forecasts the US unemployment rate for November 2026 and January 2027 using historical unemployment data and key macroeconomic indicators.

The aim was to build a model that could capture the persistence of unemployment while accounting for wider economic conditions, and then evaluate whether it could outperform a simple baseline forecast.

Data

The data is sourced from the Federal Reserve Economic Data (FRED) database and includes:
- UNRATE: US unemployment rate
- PAYEMS: Total nonfarm payroll employment
- ICSA: Initial claims for unemployment insurance
- FEDFUNDS: Federal Funds effective interest rate
- CPIAUCSL: Consumer Price Index for All Urban Consumers

The data was cleaned and combined into a monthly dataset. Payroll employment growth and year-on-year CPI inflation were calculated from the underlying series.

Methodology

I used a multiple linear regression model with an autoregressive component (AR(1)). The model uses the previous month's unemployment rate, payroll employment growth, initial jobless claims, the Federal Funds rate and CPI inflation to predict unemployment.

I chose this approach because unemployment tends to be persistent, while other economic indicators can provide additional information about changes in labour market conditions.

The model was evaluated using an expanding-window approach, comparing its historical one-month-ahead forecasts against a naive benchmark that assumes unemployment remains at the previous month's level.

The regression performed slightly better than the benchmark, with a mean absolute error (MAE) of 0.0785 percentage points compared with 0.0818 for the naive model.

To estimate forecast uncertainty, I used 10,000 simulated forecast paths, resampling historical out-of-sample forecast errors. The 10th, 50th and 90th percentiles of these simulations were used to produce the final forecasts.

Future macroeconomic predictors were held constant at their latest available values.

Forecast Results

The final forecasts, incorporating September 2026 inflation data, are:
Forecast month	P10	P50 (median)	P90
November 2026	3.975%	4.140%	4.306%
January 2027	3.902%	4.102%	4.305%

The median forecasts suggest that US unemployment will remain close to 4.1% over the forecast horizon. The P10 and P90 values provide a range of possible outcomes based on the historical forecast errors.