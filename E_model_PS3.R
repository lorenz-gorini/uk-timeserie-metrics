rm(list = ls()) # Clear workspace
library(KFAS) # for Gaussian state-space models in R
library(vars) # VAR estimation and IRF
library(tseries) # For unit root tests (adf.test)
library(MASS) # For mvrnorm simulation

# The dataset contains: Quarter, GDP, PriceIndex, and PolicyRate.
data_path <- c(
  "/Users/lorenzogorini/Library/CloudStorage/",
  "OneDrive-UniversitàCommercialeLuigiBocconi/PhD/Courses/Metrics/",
  "Metrics 3-Time Series/uk-timeserie-metrics/data/merged_data_rates.csv"
)
data <- read.csv(paste(data_path, collapse = ""), stringsAsFactors = FALSE)

# Convert the Quarter column from "2022Q1" format to Date type by replacing the
# quarter with the corresponding start month.
data$Quarter <- as.Date(paste0(substr(data$Quarter, 1, 4), "-", (as.numeric(
  substr(data$Quarter, 6, 6)
) - 1) * 3 + 1, "-01"))
# Assuming quarterly frequency (adjust start accordingly)
start_year <- as.numeric(format(min(data$Quarter), "%Y"))
start_quarter <- as.numeric(format(min(data$Quarter), "%m")) / 3 + 1
# Order data by Quarter
data <- data[order(data$Quarter), ]

# Create quarterly ts objects:
# policy_ts     : R_t (policy rate)
# dgdp_ts       : y_t (GDP growth)
# inflation_ts  : π_t (inflation rate)

# For GDP, we consider logs and differences, assuming nonstationarity.
dgdp_ts <- ts(data$GDP_log_change,
  start = c(start_year, round(start_quarter)),
  frequency = 4
)
# For the inflation rate
inflation_ts <- ts(data$Inflation,
  start = c(start_year, round(start_quarter)),
  frequency = 4
)
# Policy rate (differenced)
policy_ts <- ts(
  data$PolicyRate_diff,
  start = c(start_year, round(start_quarter)),
  frequency = 4
)

# ==========================
# 1. VAR estimation
# ==========================
# 1. Create the regressor matrix Z_t
intercept_ts <- ts(
  rep(1, length(policy_ts)),
  start = start(policy_ts),
  frequency = frequency(policy_ts)
)
lagged_policy_ts <- stats::lag(policy_ts, k = -1) # R_{t-1}
Zt <- cbind(
  intercept_ts, # intercept
  lagged_policy_ts, # R_{t-1}
  dgdp_ts, # y_t
  inflation_ts # π_t
)

state_names <- c("beta_0", "beta_R_policy", "beta_y_gdp", "beta_pi_inflation")
# 2. Build the SSModel
mod_tvp <- SSModel(
  policy_ts ~ -1 + SSMcustom(
    Z = array(Zt, dim = c(1, 4, length(policy_ts))),
    T = diag(4), # alpha_t = alpha_{t-1} + u_t
    R = diag(4),
    Q = diag(NA, 4), # variances to be estimated
    a1 = rep(0, 4), # initial state mean
    P1 = diag(0.1, 4), # diffuse prior
    state_names = state_names
  ),
  H = matrix(NA), # observation variance to be estimated
)

# 3. Estimate via maximum likelihood
fit_tvp <- fitSSM(mod_tvp,
  inits = rep(0.1, 5), # initial guesses for (σ_ε, σ_0, σ_R, σ_y, σ_π)
  method = "BFGS"
)

# ===========================
# 2. Smoothed parameter estimates
# ===========================
# 4. Extract smoothed parameter estimates
kfs_res <- KFS(fit_tvp$model, smoothing = "state", filtering = "state")
beta_smoothed <- kfs_res$alphahat # matrix T×4 of β̂_{t|T}
kfs_res

library(xtable)

# Create a summary of the smoothed states (each column is a parameter)
smoothed_summary <- data.frame(
  Parameter    = colnames(kfs_res$alphahat),
  Mean         = apply(kfs_res$alphahat, 2, mean),
  Median       = apply(kfs_res$alphahat, 2, median),
  Std_Dev      = apply(kfs_res$alphahat, 2, sd)
)

# Create an xtable object with a caption and label
smoothed_xtable <- xtable(
  smoothed_summary,
  caption = "Summary of smoothed parameter estimates",
  label = "tab:smoothed", digits = 5
)

# Print Latex code to the console
print(smoothed_xtable, include.rownames = FALSE)

# ===========================
# 3. Parameter estimates at the last time point
# ===========================

# "custom" picks all SSMcustom components
ci_states <- confint(kfs_res, parm = "custom")

# 1. Get the last time index
last_index <- nrow(kfs_res$alphahat)

# Aggregate the confidence intervals values for each state_names contained
# in ci_states
lwr <- numeric(length(state_names))
upr <- numeric(length(state_names))

for (i in seq_along(state_names)) {
  name <- state_names[i]
  lwr[i] <- ci_states[[name]][last_index, "lwr"]
  upr[i] <- ci_states[[name]][last_index, "upr"]
}
df_final <- data.frame(
  Parameter = state_names,
  Estimate  = kfs_res$alphahat[last_index, ],
  CI_Lower  = lwr,
  CI_Upper  = upr,
  row.names = NULL
)

# 3. Print via xtable
library(xtable)
print(
  xtable(df_final,
    caption = "Smoothed State Estimates and 95\\% CIs at Final Time",
    label   = "tab:tvp_states_ci",
    digits  = c(0, 4, 4, 4, 4)
  ),
  include.rownames = FALSE,
  caption.placement = "top"
)

# ===========================
# 4. Smoothed estimates of states
# ===========================
# 'alphahat' contains the smoothed estimates of states
smoothed_states <- kfs_res$alphahat # matrix T×4 of smoothed states

# Compute the 95% confidence intervals using the error covariance matrices (kfs_res$V)
# kfs_res$V is an array with dimensions n x m x T (m = number of states)
n <- nrow(smoothed_states)
m <- ncol(smoothed_states)
lower_ci <- matrix(NA, n, m)
upper_ci <- matrix(NA, n, m)
for (t in 1:n) {
  # Diagonal elements contain the variances of the smoothed states at time t
  variances <- diag(kfs_res$V[, , t])
  lower_ci[t, ] <- smoothed_states[t, ] - 1.96 * sqrt(variances)
  upper_ci[t, ] <- smoothed_states[t, ] + 1.96 * sqrt(variances)
}

# Plot each coefficient over time with its 95% CI using polygon shading
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
state_names <- colnames(smoothed_states)
# Use the actual time values from the original policy time series
time_values <- as.numeric(time(policy_ts))

for (j in seq_along(state_names)) {
  # First, plot the smoothed state line
  plot(time_values, smoothed_states[, j],
    type = "l", lwd = 2, col = "blue",
    ylim = range(smoothed_states[, j], lower_ci[, j], upper_ci[, j]),
    xlab = "Time", ylab = state_names[j],
    main = paste("Smoothed", state_names[j]),
    xaxt = "n"
  )
  # Add a custom x-axis with tick marks only (no year labels)
  axis(1, at = pretty(time_values), labels = TRUE)
  # Add a shaded polygon for the 95% confidence interval
  polygon(c(time_values, rev(time_values)),
    c(upper_ci[, j], rev(lower_ci[, j])),
    col = adjustcolor("grey", alpha.f = 0.5), border = NA
  )
  # Redraw the smoothed state line on top for clarity
  lines(time_values, smoothed_states[, j], lwd = 2, col = "blue")
  # Optionally, add dashed lines for CI boundaries
  lines(time_values, lower_ci[, j], lty = 2, col = "red")
  lines(time_values, upper_ci[, j], lty = 2, col = "red")
}

# ===========================
# 5. Filtered estimates of states
# ===========================
# 'att' contains the filtered estimates of states
filtered_states <- kfs_res$att # matrix T×4

# Compute the 95% confidence intervals using the error covariance matrices (kfs_res$V)
# kfs_res$V is an array with dimensions n x m x T (m = number of states)
n <- nrow(filtered_states)
m <- ncol(filtered_states)
filter_lower_ci <- matrix(NA, n, m)
filter_upper_ci <- matrix(NA, n, m)
for (t in 1:n) {
  # Diagonal elements contain the variances of the filtered states at time t
  variances <- diag(kfs_res$Ptt[, , t])
  filter_lower_ci[t, ] <- filtered_states[t, ] - 1.96 * sqrt(variances)
  filter_upper_ci[t, ] <- filtered_states[t, ] + 1.96 * sqrt(variances)
}

# Plot each coefficient over time with its 95% CI using polygon shading
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
state_names <- colnames(filtered_states)

for (j in seq_along(state_names)) {
  # Plot filtered state line using actual time values
  # without default x-axis labels
  plot(time_values, filtered_states[, j],
    type = "l", lwd = 2, col = "blue",
    ylim = range(
      filtered_states[, j],
      filter_lower_ci[, j],
      filter_upper_ci[, j]
    ),
    xlab = "", ylab = state_names[j],
    main = paste("Filtered", state_names[j]),
    xaxt = "n"
  )
  # Add a custom x-axis with tick marks only (no year labels)
  axis(1, at = pretty(time_values), labels = TRUE)
  # Add a shaded polygon for the 95% confidence interval
  polygon(c(time_values, rev(time_values)),
    c(filter_upper_ci[, j], rev(filter_lower_ci[, j])),
    col = adjustcolor("grey", alpha.f = 0.5), border = NA
  )
  # Redraw the filtered state line on top for clarity
  lines(time_values, filtered_states[, j], lwd = 2, col = "blue")
  # Optionally, add dashed lines for CI boundaries
  lines(time_values, filter_lower_ci[, j], lty = 2, col = "red")
  lines(time_values, filter_upper_ci[, j], lty = 2, col = "red")
}

# ===========================
# 6. Outlier detection
# ===========================
# Create a time vector corresponding to the observations
time_points <- time(policy_ts)

# Initialize an empty data frame to store outliers
outliers <- data.frame(
  Time = numeric(0),
  Parameter = character(0),
  Value = numeric(0),
  CI_Lower = numeric(0),
  CI_Upper = numeric(0),
  stringsAsFactors = FALSE
)

# Loop over each state/parameter
param_names <- colnames(filtered_states)
for (j in seq_along(param_names)) {
  # Identify indices where the filtered state is outside the confidence intervals
  idx <- which(filtered_states[, j] < ci_states[[j]][, "lwr"] |
    filtered_states[, j] > ci_states[[j]][, "upr"])

  if (length(idx) > 0) {
    temp <- data.frame(
      Time      = time_points[idx],
      Parameter = param_names[j],
      Value     = filtered_states[idx, j],
      CI_Lower  = ci_states[[j]][idx, "lwr"],
      CI_Upper  = ci_states[[j]][idx, "upr"]
    )
    outliers <- rbind(outliers, temp)
  }
}

# Display the table of outliers
print(outliers)

# Create an xtable for the outliers table
outliers_xtable <- xtable(
  outliers,
  caption = "Table of detected outliers with confidence intervals",
  label = "tab:outliers",
  digits = c(0, 2, 2, 2, 2, 2)
)

# Print the Latex code to the console
print(outliers_xtable,
  include.rownames = FALSE,
  caption.placement = "top"
)


# Economic and Historical Context (UK, 1998–2000)
# While the UK did not join the euro, several significant economic and policy events occurred around 1997–1999 that could plausibly cause structural breaks or increased volatility in macroeconomic relationships:

# Operational Independence of the Bank of England: In May 1997, the Bank of England was granted operational independence to set interest rates. This was a major policy shift affecting monetary transmission and expectations.

# Tightening and Subsequent Easing of Policy: The Bank of England raised interest rates by 150 basis points between May 1997 and June 1998 to counter overheating, then cut rates by 200 basis points between October 1998 and February 1999 as the economy slowed.

# ========================
# 7. Time-constant coefficients with Taylor rule
# ========================
# Combine the transformed series into one multivariate time series.
# Intersect the ts objects on their common time window
aligned_ts <- ts.intersect(
  R     = policy_ts, # R_t
  R_lag = stats::lag(policy_ts, -1), # R_{t-1} one‐period lag of policy rate
  y     = dgdp_ts, # y_t
  pi    = inflation_ts, # π_t
  time  = time(policy_ts) # for reference
)
# Drop any remaining NAs (if lag at the very first obs introduced NA)
df_ols <- na.omit(as.data.frame(aligned_ts))

# Inspect the first few rows
head(df_ols)

# Estimate the constant‐coefficient Taylor rule
# NOTE: The function lm() already includes an intercept term by default.
# The model is: R_t = β_0 + β_R * R_{t-1} + β_y * y_t + β_π * π_t + ε_t
model_const <- lm(R ~ R_lag + y + pi, data = df_ols)

# 4. Summarize results
summary(model_const)

summary_coef <- coef(summary(model_const))
coefs_ols <- data.frame(
  Parameter = rownames(summary_coef),
  Estimate = summary_coef[, "Estimate"],
  Std_Error = summary_coef[, "Std. Error"],
  p_value = summary_coef[, "Pr(>|t|)"],
  t_value = summary_coef[, "t value"],
  row.names = NULL
)
library(xtable)
options(scipen = -999) # Force output in scientific/exponential notation

coef_table <- xtable(
  coefs_ols,
  caption = "Constant-coefficient Taylor Rule Model Coefficients",
  label = "tab:const_coeff",
  digits = c(0, -2, -2, -2, -2, -2)
)

print(coef_table, include.rownames = FALSE)
