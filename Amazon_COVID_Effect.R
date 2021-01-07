
########################################
## HarvardX PH125.9x
## Data Science: Capstone
## Project: Amazon COVID Effect
## Name: Mootaz Abdel-Dayem
########################################

# Loading Required Libraries

if(!require(quantmod)) install.packages("quantmod", repos = "http://cran.us.r-project.org")
if(!require(forecast)) install.packages("forecast", repos = "http://cran.us.r-project.org")
if(!require(tseries)) install.packages("tseries", repos = "http://cran.us.r-project.org")
if(!require(timeSeries)) install.packages("timeSeries", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(DT)) install.packages("DT", repos = "http://cran.us.r-project.org")
if(!require(tsfknn)) install.packages("tsfknn", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(quantmod)
library(forecast)
library(tseries)
library(timeSeries)
library(dplyr)
library(readxl)
library(kableExtra)
library(data.table)
library(DT)
library(tsfknn)
library(ggplot2)


## Data Preparation
# Importing the data

# Set 1: Before COVID-19 Crisis
getSymbols("AMZN", src = "yahoo", from = "2010-01-01", to = "2019-02-28")
AMZN_data_before_covid <- as.data.frame(AMZN)
tsData_before_covid <- ts(AMZN_data_before_covid$AMZN.Close)

# Set 2: During COVID-19 Crisis
getSymbols("AMZN", src = "yahoo", from = "2010-01-01")
AMZN_data_after_covid <- as.data.frame(AMZN)
tsData_after_covid <- ts(AMZN_data_after_covid$AMZN.Close)

#Visualizing the data:

par(mfrow = c(1,2))
plot.ts(tsData_before_covid, ylab = "Closing Price", main = "Before COVID-19")
plot.ts(tsData_after_covid, ylab = "Closing Price", main = "During COVID-19")


#Dataset Preview
# The $Before$ dataset is from "2010-01-01" to "2019-12-31"

summary(AMZN_data_before_covid)


## Models Building

#######################
## # 1) ARIMA Model
#######################

par(mfrow = c(2,2))
acf(tsData_before_covid, main = "Before COVID-19")
pacf(tsData_before_covid, main = "Before COVID-19")

acf(tsData_after_covid, main = "After COVID-19")
pacf(tsData_after_covid, main = "After COVID-19")


# Model Fitting
#The $auto.arima$ function is used to determine the time series model for each of the datasets

modelfit_before_covid <- auto.arima(tsData_before_covid, lambda = "auto")
summary(modelfit_before_covid)

modelfit_after_covid <- auto.arima(tsData_after_covid, lambda = "auto")
summary(modelfit_after_covid)

# residual diagnostics for each of the fitted model
# Let's check the residual diagnostics for each of the fitted models

par(mfrow = c(2,3))

plot(modelfit_before_covid$residuals, ylab = 'Residuals', main = "Before COVID-19")
acf(modelfit_before_covid$residuals,ylim = c(-1,1), main = "Before COVID-19")
pacf(modelfit_before_covid$residuals,ylim = c(-1,1), main = "Before COVID-19")

plot(modelfit_after_covid$residuals, ylab = 'Residuals', main = "After COVID-19")
acf(modelfit_after_covid$residuals,ylim = c(-1,1), main = "After COVID-19")
pacf(modelfit_after_covid$residuals,ylim = c(-1,1), main = "After COVID-19")

###################################################
## 2) KNN Regression Time Series Forecasting Model
###################################################


par(mfrow = c(2,1))
predknn_before_covid <- knn_forecasting(AMZN_data_before_covid$AMZN.Close,
                                        h = 61, lags = 1:30, k = 32, msas = "MIMO")
predknn_after_covid <- knn_forecasting(AMZN_data_before_covid$AMZN.Close,
                                       h = 65, lags = 1:30, k = 36, msas = "MIMO")

plot(predknn_before_covid, main = "Before COVID-19")
plot(predknn_after_covid, main = "After COVID-19")



# KNN model evaluation to forecast the time series


knn_ro_before_covid <- rolling_origin(predknn_before_covid)
knn_ro_after_covid <- rolling_origin(predknn_after_covid)



#####################################################
## 3. Feed Forward Neural Network Model
#####################################################


#Creating Hidden layers
alpha <- 1.5^(-10)
hn_before_covid <- length(AMZN_data_before_covid$AMZN.Close)/
  (alpha*(length(AMZN_data_before_covid$AMZN.Close) + 61))
hn_after_covid <- length(AMZN_data_after_covid$AMZN.Close)/
  (alpha*(length(AMZN_data_after_covid$AMZN.Close) + 65))

#Fitting nnetar
lambda_before_covid <- BoxCox.lambda(AMZN_data_before_covid$AMZN.Close)
lambda_after_covid <- BoxCox.lambda(AMZN_data_after_covid$AMZN.Close)
dnn_pred_before_covid <- nnetar(AMZN_data_before_covid$AMZN.Close,
                                size = hn_before_covid, lambda = lambda_before_covid)
dnn_pred_after_covid <- nnetar(AMZN_data_after_covid$AMZN.Close,
                               size = hn_after_covid, lambda = lambda_after_covid)

# Forecasting with nnetar
dnn_forecast_before_covid <- forecast(dnn_pred_before_covid, h = 61, PI = TRUE)
dnn_forecast_after_covid <- forecast(dnn_pred_after_covid, h = 65, PI = TRUE)

plot(dnn_forecast_before_covid, title = "Before COVID-19")

# The performance of the neural network model using the following parameters:

accuracy(dnn_forecast_before_covid)
accuracy(dnn_forecast_after_covid)

#####################################
## All Models Comparaison
#####################################

summary_table_before_covid <- data.frame(Model =
                                           character(), RMSE = numeric(), MAE = numeric(),
                                         MAPE = numeric(), stringsAsFactors = FALSE)

summary_table_after_covid <- data.frame(Model =
                                          character(), RMSE = numeric(), MAE = numeric(),
                                        MAPE = numeric(), stringsAsFactors = FALSE)

summary_table_before_covid[1,] <- list("ARIMA", 13.08, 8.81, 1.02)
summary_table_before_covid[2,] <- list("KNN", 44.04, 33.78, 3.17)
summary_table_before_covid[3,] <- list("Neural Network", 13.01, 8.77, 1.02)

summary_table_after_covid[1,] <- list("ARIMA", 16.64, 10.44, 1.09)
summary_table_after_covid[2,] <- list("KNN", 45.97, 35.78, 3.36)
summary_table_after_covid[3,] <- list("Neural Network", 14.71, 9.82, 1.03)

kable(summary_table_before_covid, caption =
        "Summary of Models for data before COVID-19") %>%
  kable_styling(bootstrap_options =
                  c("striped", "hover", "condensed", "responsive"), full_width = F, fixed_thead = T )


kable(summary_table_after_covid, caption =
        "Summary of Models for data after COVID-19") %>%
  kable_styling(bootstrap_options =
                  c("striped", "hover", "condensed", "responsive"), full_width = F, fixed_thead = T )

## Conclusion

# Final Model : After COVID-19

forecast_during_covid <-
  data.frame("Date" = row.names(tail(AMZN_data_after_covid, n = 40)),
             "Actual Values" = tail(AMZN_data_after_covid$AMZN.Close, n = 40),
             "Forecasted Values" = dnn_forecast_before_covid$mean[
               c(-1,-7,-8,-14,-15,-21,-22,-28,-29,-35,-36,-41,-42,-43,-49,-50,-56,-57,-59,-60,-61)])

summary(forecast_during_covid)

# Based on the table above, we conclude that the actual values of Amazon Stock are much higher than the forecasted values.
# This means that there was a reason to make the stock prices go much higher almost close to doubling.
# This effect of course is the COVID-19 effect on the Amazon daily price.
