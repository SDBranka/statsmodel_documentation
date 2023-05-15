# Info on Data
# Sunspots dataset is used. It contains yearly 
# (1700-2008) data on sunspots from the National 
# Geophysical Data Center.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss             # For Dickey-Fuller and KPSS Tests


sunspots = sm.datasets.sunspots.load_pandas().data
# print(type(sunspots))
# <class 'pandas.core.frame.DataFrame'>
# print(sunspots.head())
#      YEAR  SUNACTIVITY
# 0  1700.0          5.0
# 1  1701.0         11.0
# 2  1702.0         16.0
# 3  1703.0         23.0
# 4  1704.0         36.0

# sunspots.to_csv("Data/sunspots.csv", index=False)


# Some preprocessing is carried out on the data. 
# The “YEAR” column is used in creating index.
sunspots.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
del sunspots["YEAR"]
# print(sunspots.shape)
# (309, 1)
# print(sunspots.head())
#             SUNACTIVITY
# 1700-12-31          5.0
# 1701-12-31         11.0
# 1702-12-31         16.0
# 1703-12-31         23.0
# 1704-12-31         36.0


# visualize the data
# chart1
# sunspots.plot(figsize=(12, 8))
# plt.title("Chart 1 Sunspots Data")
# plt.show()


# Check for Stationarity

# Dickey-Fuller Test
# ADF test is used to determine the presence of unit 
# root in the series, and hence helps in understand 
# if the series is stationary or not. The null and 
# alternate hypothesis of this test are:
# - Null Hypothesis: The series has a unit root.
# - Alternate Hypothesis: The series has no unit root.
# If the null hypothesis in failed to be rejected, 
# this test may provide evidence that the series 
# is non-stationary.
# A function is created to carry out the ADF test 
# on a time series.
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

# KPSS test                      
# KPSS is another test for checking the stationarity 
# of a time series. The null and alternate 
# hypothesis for the KPSS test are opposite that of 
# the ADF test.
# - Null Hypothesis: The process is trend stationary.
# - Alternate Hypothesis: The series has a unit root 
#   (series is not stationary).
# A function is created to carry out the KPSS test 
# on a time series.
def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

# The ADF tests gives the following results – test 
# statistic, p value and the critical value at 1%, 
# 5% , and 10% confidence intervals.
# ADF test is now applied on the data.
adf_test(sunspots["SUNACTIVITY"])
# Results of Dickey-Fuller Test:
# Test Statistic                  -2.837781
# p-value                          0.053076
# #Lags Used                       8.000000
# Number of Observations Used    300.000000
# Critical Value (1%)             -3.452337
# Critical Value (5%)             -2.871223
# Critical Value (10%)            -2.571929
# dtype: float64

# Based upon the significance level of 0.05 and the 
# p-value of ADF test, the null hypothesis can not 
# be rejected. Hence, the series is non-stationary.


# The KPSS tests gives the following results – 
# test statistic, p value and the critical value 
# at 1%, 5% , and 10% confidence intervals.
# KPSS test is now applied on the data.
kpss_test(sunspots["SUNACTIVITY"])
# Results of KPSS Test:
# Test Statistic           0.669866
# p-value                  0.016285
# Lags Used                7.000000
# Critical Value (10%)     0.347000
# Critical Value (5%)      0.463000
# Critical Value (2.5%)    0.574000
# Critical Value (1%)      0.739000
# dtype: float64

# Based upon the significance level of 0.05 and 
# the p-value of KPSS test, there is evidence for 
# rejecting the null hypothesis in favor of the 
# alternative. Hence, the series is non-stationary 
# as per the KPSS test.


# It is always better to apply both the tests, so 
# that it can be ensured that the series is truly 
# stationary. Possible outcomes of applying these 
# stationary tests are as follows:
# - Case 1: Both tests conclude that the series is 
#   not stationary - The series is not stationary
# - Case 2: Both tests conclude that the series is 
#   stationary - The series is stationary
# - Case 3: KPSS indicates stationarity and ADF 
#   indicates non-stationarity - The series is 
#   trend stationary. Trend needs to be removed to 
#   make series strict stationary. The detrended 
#   series is checked for stationarity.
# - Case 4: KPSS indicates non-stationarity and ADF 
#   indicates stationarity - The series is difference 
#   stationary. Differencing is to be used to make 
#   series stationary. The differenced series is 
#   checked for stationarity.

# Here, due to the difference in the results from 
# ADF test and KPSS test, it can be inferred that 
# the series is trend stationary and not strict 
# stationary. The series can be detrended by 
# differencing or by model fitting.


# Detrending by Differencing
# It is one of the simplest methods for detrending a 
# time series. A new series is constructed where 
# the value at the current time step is calculated 
# as the difference between the original observation 
# and the observation at the previous time step.
# Differencing is applied on the data and the result 
# is plotted.
sunspots["SUNACTIVITY_diff"] = sunspots["SUNACTIVITY"] - sunspots["SUNACTIVITY"].shift(1)
# chart2
# sunspots["SUNACTIVITY_diff"].dropna().plot(figsize=(12, 8))
# plt.title("Chart 2 SUNACTIVITY_diff")
# plt.show()


# ADF test is now applied on these detrended values 
# and stationarity is checked.
adf_test(sunspots["SUNACTIVITY_diff"].dropna())
# Results of Dickey-Fuller Test:
# Test Statistic                -1.486166e+01
# p-value                        1.715552e-27
# #Lags Used                     7.000000e+00
# Number of Observations Used    3.000000e+02
# Critical Value (1%)           -3.452337e+00
# Critical Value (5%)           -2.871223e+00
# Critical Value (10%)          -2.571929e+00
# dtype: float64


# KPSS test is now applied on these detrended values 
# and stationarity is checked.
kpss_test(sunspots["SUNACTIVITY_diff"].dropna())
#  InterpolationWarning: The test statisti
# c is outside of the range of p-values available in the
# look-up table. The actual p-value is greater than the p-value returned.
#   kpsstest = kpss(timeseries, regression="c", nlags="auto")
# Test Statistic           0.021193
# p-value                  0.100000
# Lags Used                0.000000
# Critical Value (10%)     0.347000
# Critical Value (5%)      0.463000
# Critical Value (2.5%)    0.574000
# Critical Value (1%)      0.739000
# dtype: float64

# Based upon the p-value of KPSS test, the null 
# hypothesis can not be rejected. Hence, the series 
# is stationary.



















