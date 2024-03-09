# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:11:05 2024

@author: Kajal
"""
# -- coding: utf-8 --


import pandas as pd 
import numpy as np 

wcat = pd.read_csv("C:/2-dataset/wc-at.csv")
wcat.head()

#EDA
#1 Measure the central tendancy 
#2 Measures of dispersion 
#3 Third moment business decision 
#4 Foutth moment bussiness decision 
wcat.info()
wcat.describe()
#Graphical Representation 
import matplotlib.pyplot as plt 
plt.bar(height=wcat.AT, x=np.arange(1,110,1))
plt.hist(wcat.AT)
#data is right skewed 
#scatter plot 
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')
#direction :postive, linearity : Moderate, Strength : poor 
#Now let us calculate correlation coeficient 
np.corrcoef(wcat.Waist, wcat.AT)
#let us check the direction using covar factor 
cov_output = np.cov(wcat.Waist, wcat.AT)[0,1]
cov_output 
#Now let us apply to linear regression model 
import statsmodels.formula.api as smf 
#all machine learning algo are implemented using sklearn;
#but for this statsmodel 
#Packae is being used because it gives you
#backend calculation of bita-0 and bita-1
model =smf.ols('AT~Waist', data = wcat).fit()
model.summary()
#OLS helps to find best fit model, which causes 
#least square error.
#first you check R squared valued =0.670, if R square = 0.8 means that model is strong corelation 
#fit, if R-square = 0.8 to 0.6 modetate fit. 
#Next you chack P>|t|=0, it means less than alpha, 
#alpha is 0.0, Hence the model is accepted 



#Regression line 
pred1 = model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist, pred1, "r")
plt.show()


## error calculations
res1=wcat.AT-pred1
np.mean(res1)
## it must be zero and here it -9.386986600867379e-14
res_sqrl=res1*res1
mse1=np.mean(res_sqrl)
rmse1=np.sqrt(mse1)
rmse1
## 32.76 lesser the value better the model
# how to improve this model, transformation  of 
plt.scatter(x=np.log(wcat['Waist']),y=wcat['AT'],color='brown')
np.corrcoef(np.log(wcat.Waist,wcat.AT))
# r value is 0.82 < 0.85 hence moderate linerarity

model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()

#again check the R-suqare value =0.67 which is less than 0.8
#p value is 0 less than 0.05
pred2= model2.predict(pd.DataFrame(wcat['Waist']))
#check wcat pred2 fromn variable exoplorer
#scatter diagram
plt.scatter(np.log(wcat.Waist), wcat.AT)
plt.plot(np.log(wcat.Waist), pred2, "r")
plt.legend(['predicted line', 'observed data'])
#error calculation
res2=wcat.AT-pred2
res_sqr2=res2*res2
mse=np.mean(res_sqr2)
rmse2=np.sqrt(mse)
rmse2
# there are no considerable changes
# p value is 0.02 less than 0.05
pred3=model3.predict









