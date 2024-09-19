
import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf

data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')


data.head()


data_p = data[['Timestamp', 'trips']]
data_p.columns = ['ds', 'y']


model = Prophet()
modelFit = model.fit(data_p)


future = model.make_future_dataframe(periods=744, freq='H')
forecast = model.predict(future)


pred = forecast.loc[forecast.index[-744:], 'yhat'].values

