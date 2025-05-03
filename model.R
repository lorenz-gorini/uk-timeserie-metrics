#-----------------------#
# 1. Load libraries and data
#-----------------------#
rm(list = ls()) # Clear workspace
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

# Create time series
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
# Policy rate
policy_non_stationary_ts <- ts(
  data$WeightedPolicyRate,
  start = c(start_year, round(start_quarter)),
  frequency = 4
)

# Plot the three time series
par(mfrow = c(3, 1))
plot(dgdp_ts,
  main = "GDP Growth Rate (log)",
  ylab = "GDP Growth Rate (log)",
  xlab = "Time"
)
plot(inflation_ts,
  main = "Inflation Rate",
  ylab = "Inflation Rate",
  xlab = "Time"
)
plot(policy_ts,
  main = "Policy Rate (differenced)",
  ylab = "Policy Rate (differenced)",
  xlab = "Time"
)
plot(policy_non_stationary_ts,
  main = "Policy Rate",
  ylab = "Policy Rate",
  xlab = "Time"
)

#-----------------------#
# 1.1 Unit Root Tests
#-----------------------#
# Check for unit roots; here using Augmented Dickey-Fuller Test
# Null hypothesis: series has a unit root (non-stationary)
adf_gdp <- adf.test(dgdp_ts, alternative = "stationary")
adf_price <- adf.test(inflation_ts, alternative = "stationary")
adf_policy_nonstat <- adf.test(policy_non_stationary_ts, alternative = "stationary")
adf_policy <- adf.test(policy_ts, alternative = "stationary")
print(adf_gdp) # p-value < 0.01
print(adf_price) # p-value = 0.036
print(adf_policy_nonstat) # p-value = 0.237
print(adf_policy) # p-value < 0.01

#------------------------#
# 1.2 Check for stationarity
#------------------------#
# Check for stationarity using the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
# Null hypothesis: series is stationary
kpss_gdp <- kpss.test(dgdp_ts, null = "Level")
kpss_price <- kpss.test(inflation_ts, null = "Level")
kpss_policy <- kpss.test(policy_ts, null = "Level")
kpss_policy_nonstat <- kpss.test(policy_non_stationary_ts, null = "Level")
print(kpss_gdp) # p-value > 0.1
print(kpss_price) # p-value > 0.091
print(kpss_policy) # p-value > 0.1
print(kpss_policy_nonstat) # p-value < 0.01
# The ADF and KPSS tests indicates that the null hypothesis of a unit root is
# rejected for all series, but the non-differenced policy rate, so we need to
# consider its differenced version in order to proceed.

# Combine the transformed series into one multivariate time series.
var_data <- ts(cbind(dgdp_ts, inflation_ts, policy_ts), frequency = 4)
colnames(var_data) <- c("GDP_growth", "Inflation", "PolicyRate_diff")

#-----------------------#
# 3. Lag Order Selection and VAR Estimation
#-----------------------#
# Use VARselect to choose the optimal lag length based on AIC and BIC.
lag_selection <- VARselect(var_data, lag.max = 8, type = "const")
print(lag_selection$selection)

# Select the optimal lag order p based on AIC/BIC
p_opt <- lag_selection$selection["AIC(n)"]

# Estimate the VAR(p)
var_model <- VAR(var_data, p = p_opt, type = "const")
summary(var_model)

# Extract estimated coefficient matrices and covariance matrix
coef_matrices <- Bcoef(var_model) # Coefficients for each equation
cov_matrix <- summary(var_model)$covres # Covariance matrix of residuals
print(coef_matrices)
print(cov_matrix)

#-------------------------#
# 3.2 Plot the Impulse Response Function
#-------------------------#
# Compute IRF functions via Cholesky identification for a PolicyRate shock
# and plot the responses for all three series including 95% confidence intervals
irf_res <- irf(
  var_model,
  impulse = "PolicyRate_diff",
  response = c("GDP_growth", "Inflation", "PolicyRate_diff"),
  n.ahead = 12,
  boot = TRUE,
  ci = 0.95
)
# Plot impulse responses with individual y-axis limits for each variable
responses <- c("GDP_growth", "Inflation", "PolicyRate_diff")
par(mfrow = c(length(responses), 1))
for (resp in responses) {
  irf_vals <- irf_res$irf$PolicyRate_diff[, resp]
  lower_vals <- irf_res$Lower$PolicyRate_diff[, resp]
  upper_vals <- irf_res$Upper$PolicyRate_diff[, resp]
  ylim_range <- c(min(lower_vals), max(upper_vals))

  plot(
    irf_vals,
    type = "l",
    col = "black",
    main = paste("IRF for", resp),
    xlab = "Horizon (quarters)",
    ylab = "Response",
    ylim = ylim_range
  )
  lines(lower_vals, col = "blue", lty = 2)
  lines(upper_vals, col = "blue", lty = 2)
}
par(mfrow = c(1, 1))


#-----------------------#
# 3.1 LaTeX Table for VAR Coefficients and Covariance Matrix
#-----------------------#
library(xtable)

# Transpose the coefficient matrix to have 3 columns only (one for each
# equation)
coef_matrices_transposed <- t(coef_matrices)

# Create and print LaTeX table for the transposed VAR coefficient matrices
latex_coef <- xtable(
  coef_matrices_transposed,
  caption = "Estimated VAR Coefficients (Transposed)", digits = 7
)
print(latex_coef, include.rownames = TRUE)

# Create and print LaTeX table for the residual covariance matrix with
# increased precision (4 decimal places)
latex_cov <- xtable(
  cov_matrix,
  caption = "Residual Covariance Matrix", digits = 7
)
print(latex_cov, include.rownames = TRUE)

# Check the stability condition of the VAR model
stability <- stability(var_model)
print(stability)
# Plot the stability condition
plot(stability, main = "Stability Condition of VAR Model")


# TODO: Check residuals for autocorrelation and normality
serial_test <- serial.test(var_model, lags.pt = 12, type = "PT.asymptotic")
print(serial_test)
normality_test <- normality.test(var_model)
print(normality_test)

#-----------------------#
# 4. Monte Carlo Experiment for Lag Order Selection
#-----------------------#
# Use the estimated VAR as your Data Generating Process (DGP).
n_obs <- nrow(var_data)
n_rep <- 1000 # number of Monte Carlo replications
selected_lags_aic <- numeric(n_rep)
selected_lags_bic <- numeric(n_rep)

# Extract estimated coefficients and constant terms from your VAR
# (We assume a constant VAR model here)
const_vec <- sapply(var_model$varresult, function(x) {
  coef(x)["const"]
})
# Note: Coefficient extraction might differ if you have more complex model
# specifications.

# Function to simulate a VAR process using estimated parameters:
simulate_VAR <- function(coefs, const, Sigma, n, p) {
  k <- ncol(coefs[[1]]) # number of variables
  # Initialize with zeros (or you could use the initial observations from data)
  sim_data <- matrix(0, nrow = n + p, ncol = k)
  # Generate errors (innovations)
  errors <- mvrnorm(
    n = n + p,
    mu = rep(0, k),
    Sigma = Sigma
  )

  for (t in (p + 1):(n + p)) {
    # Start with the constant term
    current_val <- const
    # Add contributions from each lag
    for (lag in 1:p) {
      current_val <- current_val + coefs[[lag]] %*% sim_data[t - lag, ]
    }
    sim_data[t, ] <- current_val + errors[t, ]
  }
  return(ts(sim_data[(p + 1):(n + p), ], frequency = 4))
}

# Prepare coefficient matrices for simulation.
# Bcoef(var_model) returns a matrix organized by equation; reshape them into a
# list for each lag.
coef_list <- list()
for (lag in 1:p_opt) {
  coef_list[[lag]] <- sapply(var_model$varresult, function(x) {
    # Extract coefficients corresponding to lag 'lag'
    # The names of coefficients are like "l1.GDP_growth", "l2.GDP_growth", ...
    var_name <- names(var_model$varresult)
    coef_name <- paste0(var_name, ".", "l", lag)
    # Build a vector for the given equation
    # If a coefficient is missing or NA, set it to 0
    sapply(coef_name, function(nm) {
      coef_val <- coef(x)[nm]
      if (is.na(coef_val)) {
        0
      } else {
        coef_val
      }
    })
  })
  # Coerce to a matrix if needed (each column corresponds to an equation)
  coef_list[[lag]] <- t(coef_list[[lag]])
}

# Run the Monte Carlo experiment
for (i in 1:n_rep) {
  sim_sample <- simulate_VAR(coef_list, const_vec, cov_matrix, n_obs, p_opt)

  # Use the same lag selection procedure on the simulated data
  lag_sel_sim <- VARselect(sim_sample, lag.max = 8, type = "const")
  selected_lags_aic[i] <- lag_sel_sim$selection["AIC(n)"]
  selected_lags_bic[i] <- lag_sel_sim$selection["SC(n)"]
}

# Summarize the frequency of each selected lag order.
table_aic <- table(selected_lags_aic)
table_bic <- table(selected_lags_bic)
print(table_aic)
print(table_bic)

#-----------------------#
# 4.1 LaTeX Table for Lag Selection Frequencies
#-----------------------#
library(xtable)

# Create a data frame for the AIC-based lag selection frequencies
df_aic <- as.data.frame(table(selected_lags_aic))
names(df_aic) <- c("Lag_Order", "Frequency")
latex_table_aic <- xtable(df_aic, caption = "Frequency of Lag Orders Selected by AIC")

# Create a data frame for the BIC-based lag selection frequencies
df_bic <- as.data.frame(table(selected_lags_bic))
names(df_bic) <- c("Lag_Order", "Frequency")
latex_table_bic <- xtable(df_bic, caption = "Frequency of Lag Orders Selected by BIC")

# Print LaTeX code for the AIC table
print(latex_table_aic, include.rownames = FALSE)

# Print LaTeX code for the BIC table
print(latex_table_bic, include.rownames = FALSE)

#-----------------------#
# 5. Impulse Response Functions and Local Projections
#-----------------------#
# (a) Using the VAR model and Cholesky identification.
# Ordering: GDP_growth, Inflation, PolicyRate
# Compute IRFs via the vars package.
irf_var <- irf(var_model,
  impulse = "PolicyRate", response = "GDP_growth",
  n.ahead = 12, boot = TRUE, ci = 0.95
)
plot(irf_var)

# (b) Local Projections Approach (Jordà, 2005)
# For each horizon h, regress GDP_growth at time (t+h) on a shock at time t and
# a set of control variables.
# We assume that the shock is the residual from the VAR equation for PolicyRate

# Extract policy residuals from the VAR (shock series)
policy_shock <- residuals(var_model)[, "PolicyRate"]

# Create a function to run a local projection regression for horizon h.
library(lmtest)
horizon_IRF <- function(h, y, shock) {
  # h: forecast horizon (0 to 12)
  # y: dependent variable (GDP_growth)
  # shock: policy shock series
  # For horizon h, the dependent variable is lagged by -h (i.e., future value).
  n <- length(y)
  # Only use observations that allow forecasting h periods ahead.
  y_h <- y[(h + 1):n]
  shock_h <- shock[1:(n - h)]

  # controls = NULL
  # if (!is.null(controls)) {
  #   controls_h <- controls[1:(n - h), ]
  #   data_reg <- data.frame(y = y_h, shock = shock_h, controls_h)
  # } else {
  #   data_reg <- data.frame(y = y_h, shock = shock_h)
  # }
  data_reg <- data.frame(y = unname(y_h), shock = unname(shock_h))
  # Run regression:
  model <- lm(y ~ shock, data = data_reg)
  return(coef(model)["shock"])
}

# Run local projections for horizons 0 to 12.
horizons <- 0:12
lp_irfs <- sapply(horizons, function(h) {
  horizon_IRF(h,
    y = var_data[, "GDP_growth"],
    shock = policy_shock
  )
})

# Plot the local projection IRFs
plot(horizons, lp_irfs,
  type = "b", col = "red", pch = 19,
  xlab = "Horizon (quarters)", ylab = "IRF",
  main = "Impulse Response: Local Projections vs. VAR"
)
lines(0:12, irf_var$irf$PolicyRate[, "GDP_growth"], type = "b", col = "blue", pch = 17)
legend("topleft",
  legend = c("Local Projections", "VAR IRF"),
  col = c("red", "blue"), pch = c(19, 17)
)
# Save the plot as PNG
png("irf_plot.png", width = 800, height = 600)
plot(horizons, lp_irfs,
  type = "b", col = "red", pch = 19,
  xlab = "Horizon (quarters)", ylab = "IRF",
  main = "Impulse Response: Local Projections vs. VAR"
)
lines(0:12, irf_var$irf$PolicyRate[, "GDP_growth"], type = "b", col = "blue", pch = 17)
legend("topleft",
  legend = c("Local Projections", "VAR IRF"),
  col = c("red", "blue"), pch = c(19, 17)
)
dev.off()
