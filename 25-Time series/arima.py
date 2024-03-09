# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:28:37 2024

@author: kajal
"""
import pandas as pd
import numpy as np
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

from matplotlib import pyplot

walmart=pd.read_csv("C:/Data Science/datasets/Walmart Footfalls Raw.csv")

#Data partition
Train=walmart.head(147)
Test=walmart.tail(12)

#In order to use this model we need first finfd out p,d and q
#p represents no of autoregressive terms-lags of dependent variable
#q represents no of moving average terms- lagged forecasting errors in prediction equation
#d represent no of non seasonal differences
#to find the values of p,d and q we use autocorrelation forcasting
#and partial autocorrelation plots(PACF)
#p values is the values of x axis of PACF
#where the plot crossess
#the upper confidence interval for the first time
#the first line which crosses the confidence interval
#q value is the value of x axis
#the plot crossess
#the upper confidence interval for the first time
tsa_plots.plot_acf(walmart.Footfalls,lags=12) #q for MA is 5

tsa_plots.plot_pacf(walmart.Footfalls,lags=12) #p for AR


#ARIMA with AR=3, MA=5
model1=ARIMA(Train.Footfalls,order=(3,1,5))
res1=model1.fit()
print(res1.summary())

#Forecast for next 12 months
start_index=len(Train)
end_index=start_index+11
forecast_test=res1.predict(start=start_index,end=end_index)

print(forecast_test)

#Evalute forecast
rmse_test=sqrt(mean_squared_error(Test.Footfalls,forecast_test))
print('Test EMSE: %.3f' % rmse_test)

#plot the forecast against  actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_test,color='red')
pyplot.show()

#Auto ARiima -automatically discover th eoptimal order for an arima
#pip install pmdarima --user

import pmdarima as pm
ar_model=pm.auto_arima(Train.Footfalls,start_p=0,start_q=0,max_p=12,max_q=12,m=1,d=None,seasonal=False,start_p=0,trace=True,error_action='warn',stepwise=True)
