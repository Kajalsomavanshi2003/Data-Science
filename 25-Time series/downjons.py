# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:11:13 2024

@author: Kajal
"""
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme()
#using the available dowjones data in seaborn
dowjones=sns.load_dataset("dowjones")
dowjones.head()
sns.lineplot(data=dowjones,x="Date",y="Price")
"""
A simple moving average (SMA) calculates the avg of selected
eange of values
by  the no of period in that raange
The most typical moving averages are 30-day 50-day, 100-day and 365 day
moving averages. Moving averages aare nice cause they can determine treds
while ignoring short term fluctuations
once can calculate the sma by simply using
"""
dowjones['sma_30']=dowjones['Price'].rolling(window=30, min_periods=1).mean()
dowjones['sma_50']=dowjones['Price'].rolling(window=50, min_periods=1).mean()
dowjones['sma_100']=dowjones['Price'].rolling(window=100, min_periods=1).mean()
dowjones['sma_365']=dowjones['Price'].rolling(window=365, min_periods=1).mean()

sns.lineplot(x="Date",y="value",legend='auto',hue='variable',data=dowjones.melt('Date'))

'''
As u can see the higher value of the window,
the lesser it is affected by short term fluctuations
and it captures long terms trends in the data
Simple Moving Averages are often used by treders
in the stock market for technical analysis
'''
#Exponential moving average
'''
Simple moving averages are nice but
they give equal weigtage to each of the data points,
whta if u wanted an average that will give higher weight
to more recent points and lesser to points in the past. in that case,
what you want is to compute the exponential moving average(EMA)
'''
dowjones['ema_50']=dowjones['Price'].ewm(spam=50,adjust=False).mean()
dowjones['ema_100']=dowjones['Price'].ewm(spam=100,adjust=False).mean()

sns.linearplot(x="Date",y="value",legend='auto',hue='variable',data=dowjones[['Date','Price','ema_50','sma_50']].melt('Date'))
'''
As you can see the ema_50 follows the price chart more closely
that the sma_50 and is more sensitive to the recent data points

'''