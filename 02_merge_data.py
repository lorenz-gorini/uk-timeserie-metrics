import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# %%
# ------------------ GDP DATA --------------------
# GDP at market prices: Current price: Seasonally adjusted £m
# Data from: https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/ybha/ukea
# PreUnit: £
# Unit: m
# Source dataset ID: UKEA

# Load the GDP data
gdp_path = "/Users/lorenzogorini/Library/CloudStorage/OneDrive-UniversitàCommercialeLuigiBocconi/PhD/Courses/Metrics/Metrics 3-Time Series/uk-timeserie-metrics/data/gdp_series-filtered.csv"
gdp_df = pd.read_csv(gdp_path)

gdp_df = gdp_df.iloc[7:].reset_index(drop=True)
gdp_df.rename(columns={"Title": "Date"}, inplace=True)
# The GDP data contain 2 timeseries:
# 1. One is observed on a yearly basis
# 2. One is observed on a quarterly basis
# We want to keep only the quarterly data, which is in the format "YYYY QN" where N is
# the quarter number (1, 2, 3, or 4).

# Filter rows with quarterly data using a regex pattern
quarter_mask = gdp_df["Date"].str.match(r"^\d{4} Q[1-4]$")
gdp_df = gdp_df[quarter_mask].reset_index(drop=True)

gdp_df.rename(
    columns={
        "Date": "Quarter",
        "Gross Domestic Product at market prices: Current price: Seasonally adjusted £m": "GDP_m",
    },
    inplace=True,
)
# Convert the GPD values to numeric, coercing errors to NaN
gdp_df["GDP_m"] = pd.to_numeric(gdp_df["GDP_m"], errors="coerce")
# Convert the 'Quarter' column (e.g. "1955 Q1") to a pandas PeriodIndex with quarterly frequency
gdp_df["Quarter"] = pd.PeriodIndex(gdp_df["Quarter"].str.replace(" ", ""), freq="Q")

# Show the rows where the conversion failed
failed_rows = gdp_df[gdp_df["GDP_m"].isna()]
if not failed_rows.empty:
    print("Rows with non-convertible GDP values:")
    print(failed_rows[["Quarter", "GDP_m"]])

# ------------------ CPI DATA --------------------
# Inflation (CPIH INDEX 00- ALL ITEMS 2015=100)
# Data from:
# https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23
# Year base: 2015 -> set to 100

# Load the CPI data
cpi_path = "/Users/lorenzogorini/Library/CloudStorage/OneDrive-UniversitàCommercialeLuigiBocconi/PhD/Courses/Metrics/Metrics 3-Time Series/uk-timeserie-metrics/data/inflation_CPIH INDEX 00- ALL ITEMS 2015=100_filtered.csv"
cpi_df = pd.read_csv(cpi_path)
cpi_df = cpi_df.iloc[7:].reset_index(drop=True)
cpi_df.rename(
    columns={"Title": "Date", "CPIH INDEX 00: ALL ITEMS 2015=100": "CPI_index_2015"},
    inplace=True,
)

# In CPI data, there are actually 3 timeseries:
# 1. One is observed on a yearly basis
# 2. One is observed on a quarterly basis
# 3. One is observed on a monthly basis
# We want to keep only the monthly data.
# The CPI data is in the format "YYYY MMM" where MMM is a 3-letter string of month name
# and the data is observed on a monthly basis. We need to convert the CPI data to a
# quarterly basis, by computing the average of the three months of each quarter.
# Convert the 'Date' column from the CPI data (in "YYYY MMM" format) to datetime; errors become NaT
cpi_df["Date"] = pd.to_datetime(cpi_df["Date"], format="%Y %b", errors="coerce")
# Keep only rows that successfully converted (i.e. the monthly series)
cpi_df = cpi_df[cpi_df["Date"].notna()]

# Create a new column 'quarter' in the format "YYYY QN" from the datetime
cpi_df["Quarter"] = cpi_df["Date"].dt.to_period("Q")

# Convert the CPI values to numeric, coercing errors to NaN
cpi_df["CPI_index_2015"] = pd.to_numeric(cpi_df["CPI_index_2015"], errors="coerce")
# Show the rows where the conversion failed
failed_rows = cpi_df[cpi_df["CPI_index_2015"].isna()]
if not failed_rows.empty:
    print("Rows with non-convertible CPI values:")
    print(failed_rows[["Date", "CPI_index_2015"]])

# Drop rows with non-convertible CPI values
cpi_df = cpi_df[cpi_df["CPI_index_2015"].notna()]

# Group by the new 'quarter' column and compute the average of CPI values
cpi_df = cpi_df.groupby("Quarter", as_index=False)["CPI_index_2015"].mean()

# Convert the 'Quarter' column to a pandas PeriodIndex with quarterly frequency
cpi_df["Quarter"] = pd.PeriodIndex(cpi_df["Quarter"].astype(str), freq="Q")

cpi_df["CPI_index_2015"].isna().sum()
# ------------------ POLICY DATA --------------------
# Load the policy rate data
policy_path = "/Users/lorenzogorini/Library/CloudStorage/OneDrive-UniversitàCommercialeLuigiBocconi/PhD/Courses/Metrics/Metrics 3-Time Series/uk-timeserie-metrics/data/policy_rate_weighted_avg_quarter.csv"
policy_df = pd.read_csv(policy_path)
# Drop Quarter and Year columns
policy_df.drop(columns=["Quarter", "Year"], inplace=True)
# Rename the columns
policy_df.rename(
    columns={"Date": "Quarter"},
    inplace=True,
)
# Convert the 'Quarter' column to a pandas PeriodIndex with quarterly frequency
policy_df["Quarter"] = pd.PeriodIndex(
    policy_df["Quarter"].str.replace(" ", ""), freq="Q"
)

# %%

# Merge the datasets on the common column 'date'
merged_df = pd.merge(gdp_df, cpi_df, on="Quarter", how="outer")
merged_df = pd.merge(merged_df, policy_df, on="Quarter", how="outer")

merged_df["GDP_m"].isna().sum(), merged_df["CPI_index_2015"].isna().sum(), merged_df[
    "WeightedPolicyRate"
].isna().sum()
# %%
merged_df = merged_df[merged_df["Quarter"] >= pd.Period("1988Q1", freq="Q")]

merged_df = merged_df.dropna()
merged_df["Year"] = merged_df["Quarter"].apply(lambda p: p.year)
merged_df["Quarter_Num"] = merged_df["Quarter"].apply(lambda p: p.quarter)
merged_df.to_csv("merged_data.csv", index=False)
# %%
# I want to run a SVAR model with the three time series, so I need
# to assume weak stationarity. For this reason, I transform the cpi and gdp
# columns into inflation and gdp growth rate as the difference between the
# logarithm of the value of period t and the one of period t-1.
merged_df = pd.read_csv("data/merged_data.csv")

merged_df["Inflation"] = np.log(merged_df["CPI_index_2015"]) - np.log(
    merged_df["CPI_index_2015"].shift(1)
)

# Compute the first log difference of GDP
merged_df["GDP_log_change"] = np.log(merged_df["GDP_m"]) - np.log(
    merged_df["GDP_m"].shift(1)
)
merged_df[["GDP_log_change", "GDP_m"]]

# After checking weak stationarity in analyze_stationarity.py, I found that policy rate
# serie is also not stationary (ADFuller test p-value=0.116). I will take the first
# difference of the policy rate series to achieve stationarity.
print(
    "ADF Test - p-value: "
    f"{adfuller(merged_df.iloc[0:]["WeightedPolicyRate"].dropna())[1]}"
)
print(
    "ADF Test - p-value: "
    f"{adfuller(merged_df.iloc[1:]["WeightedPolicyRate"].dropna())[1]}"
)
"""
Even though it is unexpected, dropping the first row of the dataframe seems to 
significantly change the p-value of the ADF test (from 0.057 to 0.119).
But since we need to take the first difference of the other series
(so that the first row will contain NaNs), we will consider
the p-value = 0.119 after dropping the first row. Therefore we will have to take the
first difference of the policy rate series to achieve stationarity.
"""
merged_df["PolicyRate_diff"] = merged_df["WeightedPolicyRate"].diff()

# Drop rows with NaN values due to differencing in related columns
initial_row_count = merged_df.shape[0]
merged_df = merged_df.dropna(subset=["GDP_log_change", "Inflation", "PolicyRate_diff"])
print(f"Number of rows dropped: {initial_row_count - merged_df.shape[0]}")

# %%
merged_df.to_csv("merged_data_rates.csv", index=False)
