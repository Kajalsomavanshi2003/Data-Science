'''
A quick overview of timeseries 

Dataset from :''



'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

#load the dataset
df=pd.read_csv('')
df.columns
df=df.rename('')

print(df.dtypes)

df.set_index('',inplace=True)


plt.plot(df.Passangers)
#there is interesting trend and it has got seasonality

#is the data stationary
# 





















