import pandas as pd

policy_path = "/Users/lorenzogorini/Library/CloudStorage/OneDrive-UniversitàCommercialeLuigiBocconi/PhD/Courses/Metrics/Metrics 3-Time Series/PS1/policy_rate.xls"

# Read the Excel file.
df = pd.read_excel(policy_path, sheet_name="HISTORICAL SINCE 1694")

# Display the original data
print("Original data:")
print(df.head(10))

df = df.iloc[8:]

new_cols = list(df.columns)
new_cols[:4] = ["Year", "Day", "Month", "PolicyRate"]
df.columns = new_cols
# -----------------------------------------------
# Step 1. Fill forward missing years.
# If the year column is missing in some rows, fill it with the last non-null value.
df["Year"] = df["Year"].ffill()

# Some rows only contain some information about the change in what the policy rate
# represents because it is computed differently in different periods.
# We can drop them for now.
df = df[df["PolicyRate"].notnull()]

# Try to convert Year to numeric, displaying rows where conversion fails.
non_convertible = df[pd.to_numeric(df["Year"], errors="coerce").isna()]
if not non_convertible.empty:
    print("Rows with non-convertible Year values:")
    print(non_convertible[["Year"]])
else:
    print("All Year values are convertible.")

# Convert the Year column to integer.
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# Drop rows with Year before 1948.
df = df[df["Year"] >= 1948]

# -----------------------------------------------
# Step 2. Define a month mapping
month_map = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

# Ensure the Month column is string, strip any extra spaces, and convert to camel case.
df["Month"] = df["Month"].astype(str).str.strip().str.capitalize()

# We map the Month names to month numbers and convert to integers.
df["Month_Num"] = df["Month"].map(month_map).astype(pd.Int8Dtype())


# -----------------------------------------------
# Step 3. Build a datetime column.
# Since the day is missing, we can assume day = 1 by default.

# First, ensure Year and Day are numeric. (Month_Num was already created)
for col in ["Year", "Day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Identify rows with invalid/missing Year or Day values.
invalid = df[df[["Year", "Day"]].isna().any(axis=1)]
if not invalid.empty:
    print("Rows with invalid or missing Year or Day values:")
    print(invalid[["Year", "Day"]])

# Also check for invalid Month by verifying Month_Num (which was mapped from Month).
invalid_month = df[df["Month_Num"].isna()]
if not invalid_month.empty:
    print("Rows with invalid Month values:")
    print(invalid_month[["Month"]])

# Drop any rows where Year, Month_Num, or Day are missing/invalid.
df = df.dropna(subset=["Year", "Month_Num", "Day"])

# Convert Year, Month_Num, and Day to integers.
df["Year"] = df["Year"].astype(int)
df["Month_Num"] = df["Month_Num"].astype(int)
df["Day"] = df["Day"].astype(int)

# Combine Year, Month_Num, and Day to create the Date column.
df["Date"] = pd.to_datetime(
    df["Year"].astype(str)
    + "-"
    + df["Month_Num"].astype(str).str.zfill(2)
    + "-"
    + df["Day"].astype(str).str.zfill(2),
    format="%Y-%m-%d",
    errors="coerce",
)

# -----------------------------------------------
# Step 4. Verify the result and drop helper columns if desired.
print("\nData with datetime:")
print(df[["Year", "Month", "Day", "Date"]].head(10))

# If you do not need the intermediate Month_Num, you can drop it
df = df[["Year", "Month", "Day", "Date", "PolicyRate"]]
df.to_csv("policy_rate_clean.csv", index=False)

# -------------------------------------------------
# Step 5. Compute a weighted average of PolicyRate value for each quarter
# The weight must be how long the PolicyRate was kept constant. We compute this
# period by looking at the Date value as the starting date and the Date in the following
# row as the end date, then allocate the duration across the quarters that the period spans.

# Convert Date column to datetime and sort the policy data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['end_date'] = df['Date'].shift(-1)

# Create dictionaries to hold the weighted sum and weights for each quarter
# The key will be a tuple: (year, quarter)
weighted_sums = {}
weights = {}

# Iterate over each policy period and allocate the period duration to intersecting quarters
for _, row in df.iterrows():
    start = row['Date']
    end = row['end_date']
    if pd.isnull(end):
        continue  # Skip the last row if no following date is available

    # Iterate over each year in the period
    for year in range(start.year, end.year + 1):
        # Iterate over the four quarters
        for quarter in range(1, 5):
            # Define quarter start and end
            quarter_start = pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)
            quarter_end = quarter_start + pd.DateOffset(months=3)

            # Compute the intersection of the policy period with the quarter period
            period_start = max(start, quarter_start)
            period_end = min(end, quarter_end)
            delta_days = (period_end - period_start).days

            if delta_days > 0:
                key = (year, quarter)
                weighted_sums[key] = weighted_sums.get(key, 0) + row['PolicyRate'] * delta_days
                weights[key] = weights.get(key, 0) + delta_days

# Compute the weighted average PolicyRate for each quarter
weighted_avg = {key: weighted_sums[key] / weights[key] for key in weighted_sums}

# Convert the result to a DataFrame with columns: Year, Quarter, WeightedPolicyRate
result_rows = [(year, quarter, weighted_avg[(year, quarter)])
               for (year, quarter) in weighted_avg]
weighted_avg_df = pd.DataFrame(result_rows, columns=['Year', 'Quarter', 'WeightedPolicyRate'])

# Create a new column with quarter format (e.g., 2022Q1)
weighted_avg_df["Date"] = (
    weighted_avg_df["Year"].astype(str) + "Q" + weighted_avg_df["Quarter"].astype(str)
)

print(weighted_avg_df)
weighted_avg_df.to_csv("policy_rate_weighted_avg_quarter.csv", index=False)