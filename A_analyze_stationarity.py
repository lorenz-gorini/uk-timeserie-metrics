"""
Analyze stationarity of time series data and fit a Structural VAR model.
This script performs the following steps:
1. Load the merged data from a CSV file.
2. Perform Augmented Dickey-Fuller (ADF) tests to check for weak stationarity.
3. Compute the first differences of the Inflation and Policy Rate series.
4. Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for the series.
5. Fit a Vector Autoregression (VAR) model to determine the lag order.
6. Fit a Structural VAR (SVAR) model with short-run restrictions.
7. Compute and plot the Impulse Response Function (IRF) for the SVAR model.
"""

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.svar_model import SVAR

# =====================
# Check weak stationarity
# =====================
# The null hypothesis of the ADF test is that the time series has a unit root (i.e., it is non-stationary).
# The alternative hypothesis is that the time series is stationary.
merged_df = pd.read_csv("merged_data_rates.csv")


def adf_test(series, series_name, include_critical_values=False):
    result = adfuller(series.dropna())
    print(f"ADF Test for {series_name}:")
    print(f"  ADF Statistic: {result[0]}")
    print(f"  p-value: {result[1]}")
    if include_critical_values:
        print("  Critical Values:")
        for key, value in result[4].items():
            print(f"    {key}: {value}")
    print("")


# Check the weak stationarity of the three time series

# Compute a second difference on GDP to achieve stationarity
merged_df["Inflation_diff"] = merged_df["Inflation"].diff()
merged_df["PolicyRate_diff"] = merged_df["WeightedPolicyRate"].diff()

adf_test(merged_df["GDP_log_change"], f"GDP_log_change", include_critical_values=True)
adf_test(merged_df["Inflation"], f"Inflation")
adf_test(merged_df["WeightedPolicyRate"], f"WeightedPolicyRate")
adf_test(merged_df["Inflation_diff"], f"Inflation")
adf_test(merged_df["PolicyRate_diff"], f"WeightedPolicyRate")


# Define the three timeframes
timeframes = [
    ("Before end of 1992", merged_df[merged_df["Year"] <= 1992]),
    (
        "Between 1993 and 2019",
        merged_df[(merged_df["Year"] >= 1993) & (merged_df["Year"] <= 2019)],
    ),
    ("After beginning of 2020", merged_df[merged_df["Year"] >= 2020]),
]

for label, subset in timeframes:
    adf_test(subset["GDP_log_change"], f"GDP_log_change ({label})")
    adf_test(subset["Inflation"], f"Inflation ({label})")
    adf_test(subset["WeightedPolicyRate"], f"WeightedPolicyRate ({label})")
    adf_test(subset["Inflation_diff"], f"Inflation ({label})")
    adf_test(subset["PolicyRate_diff"], f"WeightedPolicyRate ({label})")

# Drop rows with NaN values in the GDP_log_change and Inflation columns
merged_df = merged_df.dropna(subset=["PolicyRate_diff", "Inflation_diff"])

merged_df.to_csv("merged_data_rates_stationary.csv", index=False)

import matplotlib.pyplot as plt

series_list = [
    ("GDP_log_change", merged_df["GDP_log_change"]),
    ("Inflation_diff", merged_df["Inflation_diff"]),
    ("PolicyRate_diff", merged_df["PolicyRate_diff"]),
]

for name, series in series_list:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=axes[0])
    axes[0].set_title(f"ACF of {name}")
    plot_pacf(series.dropna(), ax=axes[1])
    axes[1].set_title(f"PACF of {name}")
    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt

    # Order the series so that the policy rate is the first variable
    model_data = merged_df[["WeightedPolicyRate", "GDP_log_change", "Inflation"]]

    # Fit a VAR to determine the lag order (using AIC)
    var_model = VAR(model_data)
    var_results = var_model.fit(maxlags=4, ic="aic")

    # Specify short-run restrictions matrix. Here the structure assumes that only the policy
    # equation (first row) is contemporaneously exogenous, i.e. the policy rate is not affected
    # by GDP changes or Inflation in the same period.
    A_matrix = np.array([[np.nan, 0, 0], [np.nan, np.nan, 0], [np.nan, np.nan, np.nan]])

    # Estimate the SVAR model using the lag order from the VAR
    svar_model = SVAR(model_data, svar_type="A", A=A_matrix)
    svar_results = svar_model.fit(maxlags=var_results.k_ar)

    # Compute the impulse response function for 10 periods ahead
    irf = svar_results.irf(10)

    # Plot the IRF to a shock in the policy rate (first variable in the ordered model)
    irf.plot(impulse=0)
    plt.tight_layout()
    plt.show()
