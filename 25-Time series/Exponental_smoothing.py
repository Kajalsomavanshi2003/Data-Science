# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:10:00 2024

@author: Kajal
"""
import pandas as pd
import numpy as np
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
#
cocacola=pd.read_excel("C:/Data Science/datasets/CocaCola_Sales_Rawdata.xlsx")
#Let us plot the data and ts nature
cocacola.Sales.plot()

#Splitting the data into train and test set data
#Since we are working on quarterly datasets and year 
#Test data=4 quarters
#train data=38

Train=cocacola.head(38)
Test=cocacola.tail(4)
#Here we are considering performance parameter as mean absolute
#rather than mean square error
#custom function is written to calculate MPSE
def MAPE(pred,org):
    temp=np.abs((pred-org)/org)*100
    return np.mean(temp)

#EDA which compramise identification of level trendes and seasonal
#In order to seperate trend and seasonality moving avarage
mv_pred=cocacola["Sales"].rolling(4).mean()
mv_pred.tail(4)
#now let us calculate mean absolute percentage of these
#values
MAPE(mv_pred.tail(4),Test.Sales)
#moving average is predictiong complete values out of which last
#are considered as predicted values and last four values of test
#basic purpose of moving average is deseasonalizing
cocacola.Sales.plot(label='org')
#This is originaal plot
#now let us seperate out trend and sesonality
for i in range(2,9,2):
    #it will take window size 2,4,6,8
    cocacola["Sales"].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=3)
    
#u can see i=4 and 8 are deseasonable plots
#Time series decomposition is the another technique of seperating 
#seasonality
decompose_ts_add=seasonal_decompose(cocacola.Sales,model="addictive",period=4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

#similar plot can be decomposed using multiplicative
decompose_ts_mul=seasonal_decompose(cocacola.Sales,model="multiplicative",period=4)
print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()
#you can observe the difference between these plots
#Now let us plot ACF plot to check the auto correlation
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales,lags=4)
#we can observe the output in which r1,r2,r3 and r4 has higher
####This is all about EDA
#Let us apply data to data driven models
#Simple exponential method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model=SimpleExpSmoothing(Train['Sales']).fit()

pred_ses=ses_model.predict(start=Test.index[0],end=Test.index[-1])

#now  calculate MAPE
MAPE(pred_ses,Test.Sales)
#we are getting 8.369866
#Holts exponential smoothing #here only trend is captured
hw_model=Holt(Train["Sales"]).fit()
pred_hw=hw_model.predict(start=Test.index[0],end=Test.index[-1])
MAPE(pred_hw,Test.Sales)
#10.4857
#Holts winter exponential smoothing with additive seasonality
hwe_model_add_add=ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_model_add_add=hwe_model_add_add.predict(start=Test.index[0],end=Test.index[-1])
MAPE(pred_hwe_model_add_add,Test.Sales)

#Holts winter exponential smoothing with multiplicative seasonal
hwe_model_mul_add=ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_hwe_model_mul_add=hwe_model_mul_add.predict(start=Test.index[0],end=Test.index[-1])
MAPE(pred_hwe_model_mul_add,Test.Sales)