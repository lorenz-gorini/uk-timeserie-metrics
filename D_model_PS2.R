rm(list = ls())
# Install packages if needed:
# install.packages(c("forecast","tseries"))

library(forecast) # For rwf(), Arima(), dm.test()
library(tseries)

data_path <- c(
  "/Users/lorenzogorini/Library/CloudStorage/",
  "OneDrive-UniversitàCommercialeLuigiBocconi/PhD/Courses/Metrics/",
  "Metrics 3-Time Series/uk-timeserie-metrics/data/merged_data_rates.csv"
)
data <- read.csv(paste(data_path, collapse = ""), stringsAsFactors = FALSE)

# Convert the Quarter column from "2022Q1" format to Date type by replacing the
# quarter with the corresponding start month.
data$Quarter <- as.Date(paste0(
  substr(data$Quarter, 1, 4), "-",
  (as.numeric(substr(data$Quarter, 6, 6)) - 1) * 3 + 1, "-01"
))
# Order data by Quarter
data <- data[order(data$Quarter), ]

# Convert to a time series object at quarterly frequency
start_year <- as.numeric(format(min(data$Quarter), "%Y"))
start_quarter <- (as.numeric(format(min(data$Quarter), "%m")) - 1) / 3 + 1
inflation_ts <- ts(data$Inflation,
  start = c(start_year, start_quarter),
  frequency = 4
)
# ===========================================================
# EXERCISE 1 - Rolling Forecast Evaluation (2-step-ahead forecasts)
# ===========================================================

h <- 2
n_total <- length(inflation_ts)
n_train <- floor(0.75 * n_total)
n_fcsts <- n_total - n_train - (h - 1)

# Storage
rw_fc <- numeric(n_fcsts)
ar_it_fc <- numeric(n_fcsts)
ar_di_fc <- numeric(n_fcsts)
actual <- numeric(n_fcsts)

for (i in seq_len(n_fcsts)) {
  origin <- n_train + i - 1
  train_ts <- window(inflation_ts, end = time(inflation_ts)[origin])
  test_val <- inflation_ts[origin + h]

  # 1. Random Walk
  rwf_res <- rwf(train_ts, h = h, drift = FALSE)
  rw_fc[i] <- rwf_res$mean[h]

  # 2. Iterated AR(p)
  ar_mod <- ar(train_ts, aic = TRUE, order.max = 8)
  p_opt <- ar_mod$order
  arma_fit <- Arima(train_ts, order = c(p_opt, 0, 0))
  fc_it <- forecast(arma_fit, h = h)
  ar_it_fc[i] <- fc_it$mean[h]

  # 3. Direct AR(p)
  y_all <- as.numeric(train_ts)
  emb <- embed(y_all, p_opt + h)
  df_dir <- data.frame(resp = emb[, 1], emb[, -1])
  mod_dir <- lm(resp ~ ., data = df_dir)
  last_vals <- tail(y_all, p_opt + h - 1)
  newdata <- as.data.frame(t(last_vals))
  names(newdata) <- paste0("X", 1:(p_opt + h - 1))
  ar_di_fc[i] <- predict(mod_dir, newdata = newdata)

  actual[i] <- test_val
}
# ----------------------------------------------------------------
# 2. Compute overall RMSFE for each method
# ----------------------------------------------------------------
# Compute RMSFE
rmsfe <- function(a, f) sqrt(mean((a - f)^2))
rmsfe_result <- rbind(
  RW        = rmsfe(actual, rw_fc),
  AR_Iter   = rmsfe(actual, ar_it_fc),
  AR_Direct = rmsfe(actual, ar_di_fc)
)

# Convert the results into a data.frame for xtable
results_df <- data.frame(
  Method = rownames(rmsfe_result),
  RMSE = rmsfe_result[, 1],
  row.names = NULL
)
# ----------------------------------------------------------------
# 3. Test for Significant Forecast Improvement (Diebold-Mariano Tests)
# ----------------------------------------------------------------

# To test if AR forecasts significantly outperform RW, use the Diebold–Mariano
# test on the series of the residuals e_t^2 differences
e_rw <- actual - rw_fc
e_it <- actual - ar_it_fc

# Run DM test and extract p-value:
dm_result <- dm.test(e_it, e_rw, h = h, power = 2, alternative = "less")
p_val <- dm_result$p.value
# p-value = 0.2681. so we cannot reject the null hypothesis and AR forecasts
# do not significantly outperform RW


# Add DM test p-value to the results data frame.
# Note: Only AR_Iter row gets the DM test p-value.
results_df$p_value <- c(NA, p_val, NA)

library(xtable)
xtab <- xtable(
  results_df,
  caption = "Root Mean Square Forecast Error (RMSE) and DM test p-value for each forecasting method",
  label = "tab:rmse", digits = 5
)
print(xtab, include.rownames = FALSE)


# ===========================================================
# EXERCISE 3 - Nonstationary time series model and cointegration
# ===========================================================
library(tseries) # adf.test(), pp.test()
library(vars) # VARselect()
library(urca) # ca.jo()


# Create time series
cpi_ts <- ts(data$CPI_index_2015,
  start = c(start_year, start_quarter),
  frequency = 4
)
gdp_ts <- ts(data$GDP_m,
  start = c(start_year, round(start_quarter)),
  frequency = 4
)
# For GDP, we consider logs and differences, assuming nonstationarity.
dgdp_ts <- ts(data$GDP_log_change,
  start = c(start_year, round(start_quarter)),
  frequency = 4
)
# Policy rate (assume already stationary)
policy_ts <- ts(
  data$PolicyRate,
  start = c(start_year, round(start_quarter)),
  frequency = 4
)

# 1. ADF on levels:
adf_gdp_lvl <- adf.test(gdp_ts, alternative = "stationary")
adf_inf_lvl <- adf.test(cpi_ts, alternative = "stationary")
adf_rt_lvl <- adf.test(policy_ts, alternative = "stationary")

# ADF on first differences:
adf_gdp_d1 <- adf.test(diff(gdp_ts), alternative = "stationary")
adf_inf_d1 <- adf.test(diff(cpi_ts), alternative = "stationary")
adf_rt_d1 <- adf.test(diff(policy_ts), alternative = "stationary")

# Phillips–Perron tests on first differences:
pp_gdp_d1 <- pp.test(diff(gdp_ts))
pp_inf_d1 <- pp.test(diff(inflation_ts))
pp_rt_d1 <- pp.test(diff(policy_ts))

# Combine p-values into a table:
test_results <- data.frame(
  Series = c("GDP", "Inflation", "Policy Rate"),
  ADF_Level = c(adf_gdp_lvl$p.value, adf_inf_lvl$p.value, adf_rt_lvl$p.value),
  ADF_Diff = c(adf_gdp_d1$p.value, adf_inf_d1$p.value, adf_rt_d1$p.value), # Test on differenced series
  PP_Diff = c(pp_gdp_d1$p.value, pp_inf_d1$p.value, pp_rt_d1$p.value) # Test on differenced series
)

library(xtable)
xtab_tests <- xtable(
  test_results,
  caption = "Unit Root Test P-Values for Time Series (levels and first differences)",
  label = "tab:unit_root",
  digits = 3
)
print(xtab_tests, include.rownames = FALSE)


# Combine series in levels
data_lvl <- cbind(GDP = dgdp_ts, INF = inflation_ts, RATE = policy_ts)

# Select lags up to, say, 8
lag_sel <- VARselect(data_lvl, lag.max = 8, type = "const")
lag_sel$selection
# Suppose AIC suggests p = lag_sel$selection["AIC(n)"]
p_opt <- lag_sel$selection["AIC(n)"]


# Johansen test: include constant in cointegration space (ecdet="const")
joh_trace <- ca.jo(data_lvl,
  type   = "trace",
  ecdet  = "const",
  K      = p_opt,
  spec   = "longrun"
)
joh_eigen <- ca.jo(data_lvl,
  type   = "eigen",
  ecdet  = "const",
  K      = p_opt,
  spec   = "longrun"
)

summary(joh_trace) # Trace test statistics
summary(joh_eigen) # Max-eigenvalue test

# Extract the critical values from the Johansen test summary
cval <- summary(joh_trace)@cval

# Create an xtable of the critical values
library(xtable)
cval_xtab <- xtable(cval,
  caption = "Critical Values for Johansen Trace Test",
  label = "tab:cval", digits = 3
)

# Print the table
print(cval_xtab, include.rownames = TRUE)


r <- 2 # Number of cointegrating vectors

# Estimate VECM parameters via ca.jo result
# Extract β (cointegration vector) and α (adjustment coefficients)
beta_mat <- joh_trace@V[, 1:r] # cointegrating vectors
alpha_mat <- joh_trace@W[, 1:r] # adjustment speeds

# Form the normalized cointegrating relationship
# E.g. β1*GDP + β2*INF + β3*RATE = 0
beta_norm <- beta_mat / beta_mat[1] # normalize on GDP coefficient

# Display
# Create an xtable for the normalized cointegrating vectors (beta_norm)
beta_xtab <- xtable(beta_norm,
  caption = "Normalized Cointegrating Vectors",
  label = "tab:beta_norm",
  digits = 5
)
print(beta_xtab, include.rownames = TRUE)

# Create an xtable for the adjustment coefficients (alpha_mat)
alpha_xtab <- xtable(alpha_mat,
  caption = "Adjustment Coefficients",
  label = "tab:alpha_mat",
  digits = 5
)
print(alpha_xtab, include.rownames = TRUE)

# Convert to a VECM object
vecm_mod <- cajorls(joh_trace, r = r) # OLS estimates of VECM
# Convert the VECM regression results into a LaTeX table using xtable
library(xtable)
latex_coef <- xtable(vecm_mod$rlm$coefficients,
  caption = "Regression Coefficients for VECM Equations",
  label = "tab:vecm_coef",
  digits = 5
)
print(latex_coef, include.rownames = TRUE)

# ===========================================================
# EXERCISE 4 - Spectral Analysis
# ===========================================================

# Compute growth rates
inflation_gr <- diff(cpi_ts)
policy_rate_gr <- diff(policy_ts)
gdp_gr <- diff(gdp_ts)

# Set up plotting area
par(mfrow = c(3, 2), mar = c(4, 4, 2, 1))

# Function to plot spectrum
plot_spectrum <- function(ts_data, title) {
  spec <- spectrum(ts_data, log = "no", spans = c(3, 3), taper = 0.1, plot = FALSE)
  plot(spec$freq, spec$spec,
    type = "l",
    xlab = "Frequency", ylab = "Spectral Density",
    main = title
  )
}

plot_spectrum(cpi_ts, "Spectrum: CPI - Levels")
plot_spectrum(inflation_gr, "Spectrum: CPI - Growth Rates (Inflation)")

plot_spectrum(gdp_ts, "Spectrum: GDP - Levels")
plot_spectrum(gdp_gr, "Spectrum: GDP - Growth Rates")

plot_spectrum(policy_ts, "Spectrum: Policy Rate - Levels")
plot_spectrum(policy_rate_gr, "Spectrum: Policy Rate - Growth Rates")
