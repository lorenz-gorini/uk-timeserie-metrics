# This script estimates a VAR model with step and pulse dummies for Brexit and
# Covid shocks.

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

#------------------------#
# 2. Check for structural breaks
#------------------------#
# Check for structural breaks using the Bai-Perron test
bp_test <- breakpoints(inflation_ts ~ 1, h = 0.15)
plot(bp_test)
print(breakdates(bp_test)) # 1993.5 2019.5

#-----------------------#
# 3. Detrend and Stationarity Test for policy_non_stationary_ts
#-----------------------#
# Subset the series to the window [1993Q2, 2008Q3]
policy_subset <- window(policy_non_stationary_ts, start = c(1993, 2), end = c(2008, 3))

# Plot the subset series to visually inspect trend
plot(policy_subset,
  main = "Policy Rate (Non-Stationary) [1993Q2 - 2008Q3]",
  xlab = "Time", ylab = "Policy Rate"
)

# Create a time index corresponding to the subset
policy_time_subset <- time(policy_subset)

# Fit a linear trend model for the subset
policy_trend_model <- lm(as.numeric(policy_subset) ~ policy_time_subset)
print(summary(policy_trend_model))

# Remove the trend component by extracting residuals
policy_detrended <- ts(residuals(policy_trend_model),
  start = start(policy_subset),
  frequency = frequency(policy_subset)
)

# Plot the detrended policy rate
plot(policy_detrended,
  main = "Detrended Policy Rate [1993Q2 - 2008Q3]",
  xlab = "Time", ylab = "Detrended Values"
)

# Check stationarity of the detrended series
adf_policy_detrended <- adf.test(policy_detrended, alternative = "stationary")
kpss_policy_detrended <- kpss.test(policy_detrended, null = "Level")

print(adf_policy_detrended)
print(kpss_policy_detrended)

# COMMENT: Even after detrending, we cannot reject the null hypothesis of
# non-stationarity (unit root) in ADF test, but at least now we also cannot
# reject the null hypothesis of stationarity in KPSS test. This suggests that
# the series may be a bit more stationary after removing the trend component,
# but still we cannot reject the non-stationarity in ADF test

#------------------------#
# 4. Pulse and Step Dummies for Brexit and Covid shocks
#------------------------#
# 1. Pulse dummy at referendum (2016 Q2)
pulse_brexit_ref <- as.numeric(time_idx == 2016 + (2 - 1) / 4)

# 6. Step from referendum onward (2016 Q2)
step_brex_ref <- as.numeric(time_idx >= 2016 + (2 - 1) / 4)

# 2. Step dummy from 2020 Q1 onward (post-withdrawal)
step_brex_withdrawal <- as.numeric(time_idx >= 2020 + (1 - 1) / 4)

# 3. Step dummy from 2021 Q1 onward (post-transition)
step_brex_transition <- as.numeric(time_idx >= 2021 + (1 - 1) / 4)

# 4. Pulse dummy at Covid shock (2020 Q2)
pulse_covid <- as.numeric(time_idx == 2020 + (2 - 1) / 4)

# 5. Step dummy at Covid shock (2020 Q2)
step_covid <- as.numeric(time_idx >= 2020 + (2 - 1) / 4)

#------------------------#
# 4.1 Compare pulse and step dummy for Covid structural break
#------------------------#
# Fit ARIMA models with and without the pulse dummy
library(forecast)
# Choose the AR order for the pure AR model for inflation
p_opt <- ar(inflation_ts, aic = TRUE, order.max = 8)$order
# Model with pulse dummy
fit_pulse <- Arima(inflation_ts, order = c(p_opt, 0, 0), xreg = pulse_covid)
# Model with step dummy
fit_step <- Arima(inflation_ts, order = c(p_opt, 0, 0), xreg = step_covid)
# Compare models based on AIC and BIC
cat(
  "Pulse dummy model:   AIC =",
  AIC(fit_pulse),
  " BIC =",
  BIC(fit_pulse),
  "\n"
)
cat(
  "Step dummy model:    AIC =",
  AIC(fit_step),
  " BIC =",
  BIC(fit_step),
  "\n"
)
# Likelihood-Ratio test comparing nested models: (step vs. pulse)
ll_step <- as.numeric(logLik(fit_step))
ll_pulse <- as.numeric(logLik(fit_pulse))
lr_stat <- 2 * (ll_pulse - ll_step)
p_val <- pchisq(lr_stat, df = 1, lower.tail = FALSE)
cat(
  "Likelihood Ratio Test: LR stat =",
  round(lr_stat, 2),
  " p-value =",
  round(p_val, 4),
  "\n"
)

# Plot the inflation series and the fitted values for the Covid structural
# break analysis
plot(
  window(inflation_ts, start = 2015),
  main = "Inflation with Pulse vs. Step Dummy Fits (Covid Q2 2020)",
  ylab = "Inflation (log-diff)",
  xlab = "Time"
)
lines(
  window(fitted(fit_pulse), start = 2015),
  col = "blue",
  lwd = 2,
  lty = 2
)
lines(
  window(fitted(fit_step), start = 2015),
  col = "red",
  lwd = 2,
  lty = 3
)
abline(
  v = 2020 + (2 - 1) / 4,
  col = "darkgrey",
  lty = 4
)
legend(
  "topleft",
  legend = c("Actual", "Pulse fit", "Step fit", "Covid Break"),
  col    = c("black", "blue", "red", "darkgrey"),
  lty    = c(1, 2, 3, 4),
  lwd    = c(1, 2, 2, 1)
)

#------------------------#
# 4.2 Test whether the other dummies are significant
#------------------------#
# Test whether the other dummies are significant by comparing models with Covid
# dummy only versus models that include an additional dummy along with Covid.
library(forecast)
# Choose the AR order for the pure AR model for inflation
p_opt <- ar(inflation_ts, aic = TRUE, order.max = 8)$order

# Model with Covid (step) dummy only
fit_covid_only <- Arima(inflation_ts, order = c(p_opt, 0, 0), xreg = step_covid)

# Models with one extra dummy (each) together with Covid
fit_brexit_refer <- Arima(
  inflation_ts,
  order = c(p_opt, 0, 0),
  xreg = cbind(pulse_brexit_ref, step_covid)
)
fit_step_withdrawal_brexit <- Arima(
  inflation_ts,
  order = c(p_opt, 0, 0),
  xreg = cbind(step_brex_withdrawal, step_covid)
)
fit_step_transition_brexit <- Arima(
  inflation_ts,
  order = c(p_opt, 0, 0),
  xreg = cbind(step_brex_transition, step_covid)
)

# Compare models based on AIC and BIC
cat(
  "Covid only model:      AIC =",
  AIC(fit_covid_only),
  " BIC =",
  BIC(fit_covid_only),
  "\n"
)
cat(
  "Pulse Brexit + Covid model:   AIC =",
  AIC(fit_brexit_refer),
  " BIC =",
  BIC(fit_brexit_refer),
  "\n"
)
cat(
  "Step Post-Withdrawal from EU (1/1/2020) + Covid: AIC =",
  AIC(fit_step_withdrawal_brexit),
  " BIC =",
  BIC(fit_step_withdrawal_brexit),
  "\n"
)
cat(
  "Step Post-Transition (31/12/2020) + Covid: AIC =",
  AIC(fit_step_transition_brexit),
  " BIC =",
  BIC(fit_step_transition_brexit),
  "\n"
)

# Likelihood-Ratio tests comparing nested models: (Covid only vs. additional dummy + Covid)
ll_covid_only <- as.numeric(logLik(fit_covid_only))

# Function for LR test
lr_test <- function(fit_restricted, fit_full, df) {
  ll_restricted <- as.numeric(logLik(fit_restricted))
  ll_full <- as.numeric(logLik(fit_full))
  lr_stat <- 2 * (ll_full - ll_restricted)
  p_val <- pchisq(lr_stat, df = df, lower.tail = FALSE)
  list(lr_stat = lr_stat, p_value = p_val)
}

lr_pulse <- lr_test(fit_covid_only, fit_brexit_refer, df = 1)
lr_withdr <- lr_test(fit_covid_only, fit_step_withdrawal_brexit, df = 1)
lr_trans <- lr_test(fit_covid_only, fit_step_transition_brexit, df = 1)

cat(
  "Pulse dummy + Covid vs. Covid only:         LR stat =",
  round(lr_pulse$lr_stat, 2),
  " p-value =",
  round(lr_pulse$p_value, 4),
  "\n"
)
cat(
  "Step Post-Withdrawal from EU (1/1/2020) dummy + Covid vs. Covid only: LR stat =",
  round(lr_withdr$lr_stat, 2),
  " p-value =",
  round(lr_withdr$p_value, 4),
  "\n"
)
cat(
  "Step Post-Transition (31/12/2020) dummy + Covid vs. Covid only:   LR stat =",
  round(lr_trans$lr_stat, 2),
  " p-value =",
  round(lr_trans$p_value, 4),
  "\n"
)

# Combine the transformed series into one multivariate time series.
var_data <- ts(
  cbind(
    dgdp_ts,
    inflation_ts,
    policy_ts,
    # step_brex_transition,
    pulse_covid
  ),
  frequency = 4
)
colnames(var_data) <- c(
  "GDP_growth",
  "Inflation",
  "PolicyRate",
  # "StepBrexitTransition",
  "PulseCovid"
)
