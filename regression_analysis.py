#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')


# # Regression Analysis: Seasonal Effects with Sklearn Linear Regression
# In this notebook, Build a SKLearn linear regression model to predict Yen futures ("settle") returns with *lagged* Yen futures returns. 

# In[2]:


# Futures contract on the Yen-dollar exchange rate:
# This is the continuous chain of the futures contracts that are 1 month to expiration
yen_futures = pd.read_csv(
    Path("../Data/yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()


# In[3]:


# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures.head()


# # Data Preparation

# ### Returns

# In[5]:


# Create a series using "Settle" price percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
yen_futures['Return'] = yen_futures["Settle"].pct_change() * 100
yen_futures = yen_futures.replace(-np.inf, np.nan).dropna()
yen_futures.tail() 


# ### Lagged Returns 

# In[6]:


# Create a lagged return using the shift function
yen_futures['Lagged_Return'] = yen_futures['Return'].shift()
yen_futures.dropna(inplace=True)
yen_futures.tail()


# ### Train Test Split

# In[7]:


# Create a train/test split for the data using 2018-2019 for testing and the rest for training
train = yen_futures[:'2017']
test = yen_futures['2018':]


# In[8]:


# Create four dataframes:
# X_train (training set using just the independent variables), X_test (test set of of just the independent variables)
# Y_train (training set using just the "y" variable, i.e., "Futures Return"), Y_test (test set of just the "y" variable):
X_train = train["Lagged_Return"].to_frame()
X_test = test["Lagged_Return"].to_frame()
y_train = train["Return"]
y_test = test["Return"]


# In[10]:


X_train.head()


# # Linear Regression Model

# In[11]:


# Create a Linear Regression model and fit it to the training data
from sklearn.linear_model import LinearRegression

# Fit a SKLearn linear regression using just the training set (X_train, Y_train):
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# # Make predictions using the Testing Data
# 
# Note: We want to evaluate the model using data that it has never seen before, in this case: X_test.

# In[13]:


# Make a prediction of "y" values using just the test dataset
predicted_y_values = regressor.predict(X_test)


# In[14]:


# Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
Results = y_test.to_frame() #  transforms actual test values into a DataFrame.
Results["Predicted Return"] = predicted_y_values


# In[15]:


# Plot the first 20 predictions vs the true values
Results[:20].plot(subplots=True)


# # Out-of-Sample Performance
# 
# Evaluate the model using "out-of-sample" data (X_test and y_test)

# In[18]:


from sklearn.metrics import mean_squared_error
# Calculate the mean_squared_error (MSE) on actual versus predicted test "y" 
mse = mean_squared_error(
    Results["Return"],
    Results["Predicted Return"]
)

# Using that mean-squared-error, calculate the root-mean-squared error (RMSE):
rmse = np.sqrt(mse)
print(f"Out-of-Sample Root Mean Squared Error (RMSE): {rmse}")


# # In-Sample Performance
# 
# Evaluate the model using in-sample data (X_train and y_train)

# In[19]:


# Construct a dataframe using just the "y" training data:
in_sample_results = y_train.to_frame()

# Add a column of "in-sample" predictions to that dataframe:  
in_sample_results["In-sample Predictions"] = regressor.predict(X_train)

# Calculate in-sample mean_squared_error (for comparison to out-of-sample)
in_sample_mse = mean_squared_error(
    in_sample_results["Return"],
    in_sample_results["In-sample Predictions"]
)

# Calculate in-sample root mean_squared_error (for comparison to out-of-sample)
in_sample_rmse = np.sqrt(in_sample_mse)
print(f"In-sample Root Mean Squared Error (RMSE): {in_sample_rmse}")


# # Conclusions

# In[ ]:


# The out-of-sample RMSE is lower than the in-sample RMSE. 
# RMSE is typically lower for training data, but is higher in this case.

