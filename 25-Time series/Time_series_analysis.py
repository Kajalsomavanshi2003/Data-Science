# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:48:06 2024

@author: Admin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')

df=pd.read_csv("C:/Data Science/datasets/AirPassengers.csv")
df.columns
df=df.rename({'#Passengers': 'Passengers'},axis=1)
print(df.dtypes)

#Month is text and passenger is init
#Now let us convert it into date and time
df['Month']=pd.to_datetime(df['Month'])
print(df.dtypes)
df.set_index('Month',inplace=True)
plt.plot(df.Passengers)

#There is increasing trend and it has got seasonality

#is tha data stationary?
#Dickly Fuller test
from statsmodels.tsa.stattools import adfuller
adf,pvalue,usedlag_,critical_values_,icbest_=adfuller(df)
print("pvalue = ",pvalue,"if above 0.05, data is not stationary")
#Since the data is not stationary we may need SARIMA and not just ARIMA
#Now let us extract the year and month from the data and time column
df['year']=[d.year for d in df.index]
df['month']=[d.strftime('%b') for d in df.index]
years=df['year'].unique()

#Plot yearly and monthly valueas as boxplot
sns.boxplot(x='year',y='Passengers',data=df)
#No of passengers are going up year by year
sns.boxplot(x='month',y='Passengers',data=df)
#Over all there is higher trend in july and august compared to rest
#Extract and plot trend seasonal and residuals
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed=seasonal_decompose(df['Passengers'],model='additive')

#Additive time series;
#Value=Base level +Trend +Seasonality+Error
#Multiplicative time seriees;
#value=Base Level*trend*Seasonality*error

trend=decomposed.trend
seasonal=decomposed.seasonal #Cyclic behaviour may not be seasonal
residual=decomposed.resid
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Passenger'],label='Original',color='yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend,label='Trend',color='yellow')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal',color='yellow')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual,label='Residual',color='yellow')
plt.legend(loc='upper left')
plt.show()
'''
Trend is going up from 1950s to 60s
IT is highly seasonal showing peaks at particular interval
this help to specify the prediction model
'''
from statsmodels.tsa.stattools import acf
acf_144=acf(df.Passengers,nlags=144)
plt.plot(acf_144)
#Auto correlation above zero means positive correlation and below as negative 
#obtain same but with single line and more info...
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)
#Any lag before 40 has positive correlation 
#Horizantal bands indicate 95% and 99% (dashed ) confidence based