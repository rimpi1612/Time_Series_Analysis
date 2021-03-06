# A Yen for the Future

![Yen Photo](Images/unit-10-readme-photo.png)

## Background

The financial departments of large companies often deal with foreign currency transactions while doing international business. As a result, they are always looking for anything that can help them better understand the future direction and risk of various currencies. Hedge funds, too, are keenly interested in anything that will give them a consistent edge in predicting currency movements.

In this assignment, we test the many time-series tools that we have learned in order to predict future movements in the value of the Japanese yen versus the U.S. dollar.

We  gain proficiency in the following tasks:

1. Time Series Forecasting
2. Linear Regression Modeling


- - -

### Files

Use the following  code to complete this assignment. 

[Time-Series Starter Notebook](Notebook/time_series_analysis.ipynb)

[Linear Regression Starter Notebook](Notebook/regression_analysis.ipynb)

[Yen Data CSV File](Data/yen.csv)

- - -

#### Time-Series Forecasting

In this notebook, we  load historical Dollar-Yen exchange rate futures data and apply time series analysis and modeling to determine whether there is any predictable behavior.

Follow the steps outlined in the time-series notebook file to reflect the following:

1. Decomposition using a Hodrick-Prescott Filter (Decompose the Settle price into trend and noise).
2. Forecasting Returns using an ARMA Model.
3. Forecasting the Settle Price using an ARIMA Model.
4. Forecasting Volatility with GARCH.

Using the results of the time series analysis and modeling we try to answer the following questions:

1. Based on our time series analysis, would we buy the yen now?
2. Is the risk of the yen expected to increase or decrease?
3. Based on the model evaluation, would we feel confident in using these models for trading?

It's difficult to conclusively say buy or sell, but given the results of these models, I would recommend avoid buying.

![Settle](Images/settle_vs_trend.png)


The Hodrick-Prescott filter shows both trend and noise of the price changes. The noise mostly sits between +$100 and -$100, sometimes reaching out at +/-$400.  However, it is not specifically point out a consistent upward trend based on price history from 2015 to the present.

![Noise](Images/noise.png)

The volatility measurement via the GARCH model projects the annualized 5-day forecast shows volatility increasing
from 7.44 to 7.60.  

![Garch](Images/garch.png)

![Volatility](Images/volatility.png)


#### Linear Regression Forecasting

In this notebook, we build a Scikit-Learn linear regression model to predict Yen futures ("settle") returns with *lagged* Yen futures returns and categorical calendar seasonal effects (e.g., day-of-week or week-of-year seasonal effects).

Following the steps for the regression_analysis, the notebook file reflects the following:

1. Data Preparation (Creating Returns and Lagged Returns and splitting the data into training and testing data)
2. Fitting a Linear Regression Model.
3. Making predictions using the testing data.
4. Out-of-sample performance.
5. In-sample performance.

Using the results of the linear regression analysis and modeling we try to answer the following question:

* Does this model perform better or worse on out-of-sample data compared to in-sample data?

The regression model based on X and y training data and applying it to X testing data, there does not appear to be much correlation between predicted and actual returns over the next 20 trading days.  Each major peak appears to go in opposite directions.  

![Regressor](Images/regressor.png)

The model appears to perform better with out-of-sample data compared to in-sample data.  Out-of-sample root mean squared error was lower than in-sample readings. By this measure the model may do a better job of predicting out-of-sample prices comparatively, but it will be risky to relay on this model to predict future prices.

![InOut](Images/put_in_sample.png)


- - -

### Hints and Considerations

* Out-of-sample data is data that the model hasn't seen before (Testing data).
* In-sample data is data that the model was trained on (Training data).