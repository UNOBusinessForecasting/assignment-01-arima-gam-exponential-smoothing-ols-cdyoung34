
import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf

## read in the data
data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')

## print the first five rows of the data
data.head()

## only use two columns of the data frame
data_p = data[['Timestamp', 'trips']]
data_p.columns = ['ds', 'y']

## create the model and fit the model
## I added the is_fitted line because of my comment in the assignment 1 channel
model = Prophet()
modelFit = model.fit(data_p)
modelFit._is_fitted = True

## make the future data frame
future = model.make_future_dataframe(periods=744, freq='H')

# make predicitons
forecast = model.predict(future)

# turn predictions into vector
pred = forecast.loc[forecast.index[-744:], 'yhat'].values


