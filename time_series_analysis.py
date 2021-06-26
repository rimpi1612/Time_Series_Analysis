#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Importing Packages
import numpy as np
import pandas as pd
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')


# # Return Forecasting: Read Historical Daily Yen Futures Data
# 
# In this notebook, we are loading historical Dollar-Yen exchange rate futures data and applying time series analysis and modeling to determine whether there is any predictable behavior.

# In[32]:


# Futures contract on the Yen-dollar exchange rate:
# This is the continuous chain of the futures contracts that are 1 month to expiration
yen_futures = pd.read_csv(
    Path("../Data/yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()


# In[33]:


# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures.head()


#  # Return Forecasting: Initial Time-Series Plotting

# Plotting the "Settle" price. Do you see any patterns, long-term and/or short?

# In[34]:


# Plot just the "Settle" column from the dataframe:
yen_futures['Settle'].plot(title="Yen Futures Settle Prices")

# We do not see any long term patern. but there is a pattern significant drop after several years. 


# ---

# # Decomposition Using a Hodrick-Prescott Filter

#  Using a Hodrick-Prescott Filter, decompose the Settle price into a trend and noise.

# In[35]:


import statsmodels.api as sm
# Class Notes 10.1
# Applying the Hodrick-Prescott Filter by decomposing the "Settle" price into two separate series:
yen_noise, yen_trend = sm.tsa.filters.hpfilter(yen_futures['Settle'])


# In[36]:


yen_trend.plot()


# In[37]:


yen_noise.plot()


# In[38]:


# A dataframe of just the settle price, and add columns for "noise" and "trend" series from above:
df_hpfilter = pd.DataFrame(columns=['Settle', 'Noise', 'Trend'])
df_hpfilter['Settle'] = yen_futures['Settle']
df_hpfilter['Noise'] = yen_noise
df_hpfilter['Trend'] = yen_trend
df_hpfilter.head()


# In[39]:


# The Settle Price vs. the Trend for 2015 to the present
df_hpfilter_new = df_hpfilter.loc["2015-01-01":, : ]
# Line Plot
df_hpfilter_new.loc[:,['Settle','Trend']].plot(figsize=(20,10), title="Settle vs Trend")


# In[40]:


# Scatter plot.
df_hpfilter_new.plot(kind='scatter', x='Trend', y='Settle')


# In[41]:


# Plot the Settle Noise
df_hpfilter['Noise'].plot(title="Noise", figsize=(20,10))


# ---

# # Forecasting Returns using an ARMA Model

# Using futures Settle *Returns*, estimate an ARMA model
# 
# 1. ARMA: Create an ARMA model and fit it to the returns data. Note: Set the AR and MA ("p" and "q") parameters to p=2 and q=1: order=(2, 1).
# 2. Output the ARMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
# 3. Plot the 5-day forecast of the forecasted returns (the results forecast from ARMA model)

# In[59]:


# Create a series using "Settle" price percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
returns = (yen_futures[["Settle"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()


# In[60]:


from statsmodels.tsa.arima_model import ARMA
# Classnotes 10.2
# Estimate and ARMA model using statsmodels (use order=(2, 1))
arma_model = ARMA(returns["Settle"], order=(2,1))
# Fit the model and assign it to a variable called results
results = arma_model.fit()


# In[61]:


# Output model summary results:
results.summary()


# In[62]:


# Plot the 5 Day Returns Forecast
pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5 Day Returns Forecast")


# ---

# # Forecasting the Settle Price using an ARIMA Model

#  1. Using the *raw* Yen **Settle Price**, estimate an ARIMA model.
#      1. Set P=5, D=1, and Q=1 in the model (e.g., ARIMA(df, order=(5,1,1))
#      2. P= # of Auto-Regressive Lags, D= # of Differences (this is usually =1), Q= # of Moving Average Lags
#  2. Output the ARIMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
#  3. Construct a 5 day forecast for the Settle Price. What does the model forecast will happen to the Japanese Yen in the near term?

# In[63]:


from statsmodels.tsa.arima_model import ARIMA

# Estimate and ARIMA Model:
# Hint: ARIMA(df, order=(p, d, q))
arima_model = ARIMA(yen_futures['Settle'], order=(5,1,1))

# Fit the model
arima_results = arima_model.fit()


# In[64]:


# Output model summary results:
arima_results.summary()


# In[65]:


# Plot the 5 Day Price Forecast
pd.DataFrame(arima_results.forecast(steps=3)[0]).plot(title="5 Day Futures Price Forecast")


# ---

# # Volatility Forecasting with GARCH
# 
# Rather than predicting returns, let's forecast near-term **volatility** of Japanese Yen futures returns. Being able to accurately predict volatility will be extremely useful if we want to trade in derivatives or quantify our maximum loss.
#  
# Using futures Settle *Returns*, estimate an GARCH model
# 
# 1. GARCH: Create an GARCH model and fit it to the returns data. Note: Set the parameters to p=2 and q=1: order=(2, 1).
# 2. Output the GARCH summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
# 3. Plot the 5-day forecast of the volatility.

# In[66]:


# Class Notes 10.2
from arch import arch_model # Autoregressive Conditional Heteroskedasticity (ARCH)


# In[67]:


# Estimate a GARCH model:
garch_model = arch_model(returns, mean='Zero', vol="GARCH", p=2, q=1)

# Fit the model
garch_result = garch_model.fit(disp='off')


# In[69]:


# Summarize the model results
garch_result.summary()


# In[71]:


garch_result.plot(annualize='D');


# In[72]:


# Find the last day of the dataset
last_day = returns.index.max().strftime('%Y-%m-%d')
last_day


# In[73]:


# Create a 5 day forecast of volatility
forecast_horizon = 5
# Start the forecast using the last_day calculated above
garch_forecast = garch_result.forecast(start=last_day, horizon=forecast_horizon)
garch_forecast


# In[75]:


# Annualize the forecast
intermediate = np.sqrt(garch_forecast.variance.dropna() * 252)
intermediate.head()


# In[76]:


# Each row represents the forecast of volatility for the following days.
# Transpose the forecast so that it is easier to plot
final = intermediate.dropna().T
final.head()


# In[77]:


# Plot the final forecast
final.plot()
# This chart shows our estimate of the volatility of the YEN Price for the next 5 days.
# The chart shows that volatility (i.e., risk) in the market is expected to rise.
# Therefore, with GARCH, we have developed a cool way to forecast risk in the market.


# ---

# # Conclusions

# Based on your time series analysis, would you buy the yen now?
# 
# Is the risk of the yen expected to increase or decrease?
# 
# Based on the model evaluation, would you feel confident in using these models for trading?
# 
# As we can see the forecast for the settle price for the next 5 days is going up. We would not buy because form the GARCH forecast for volatility we can clearly see that the risk is going above.
# Since the p-value for ARMA and ARIMA are > 0.5, the model is not very accurate for trading.

# In[ ]:




