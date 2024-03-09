# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 08:25:27 2024

@author: Kajal
"""


import pandas as pd
import numpy as np
wcat = pd.read_csv('C:/2-dataset/wc-at.csv')
#Exploratory data analysis
#1. Measure  the central tendancy
#2. Measure of dispersion 
#3. Third moment business decision
#4. Fourth moment business decision
wcat.info()
wcat.describe()

#Graphical Representation
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)
# data is right skewed

# scatter plot
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')
# direction:positive, linearity:moderate, strength: poor
# Now let us calculate correlation coeficient
np.corrcoef(wcat.Waist,wcat.AT)
#let us check the direction using the cover factor
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
  
## let us apply to linear regression model
import statsmodels.formula.api as smf
# all machine learning algorithms are implement using sklearn
# but for this statmodel used because it gives you
# backend calculation of bita-0 and bita-1

model=smf.ols('AT~Waist',data=wcat).fit()
model.summary()

# OLS helps to find best fit model, which causes
# least square error.
# first you check R squared value=0.670, if square = 0.8 means that model is best fit
# fit, if R-Square =0.8 to 0.6 moderate correlation
# Next you check P>|t}=0, it means less than alpha,
# alpha is 0.05, Hence the model is accepted

# Regression line
pred1=model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,'r')
plt.show()

# Error calculation
res1=wcat.AT-pred1
np.mean(res1)
## it must be zero and here it Out[39]: -9.386986600867379e-14
res_sqrl=res1*res1
mse1=np.mean(res_sqrl)
rmse1=np.sqrt(mse1)
rmse1 # This is for simple linear regression
# Now to improve this model log transformation is used
plt.scatter(x=np.log(wcat['Waist']),y=wcat['AT'],color='brown')
np.corrcoef(wcat.Waist,wcat.AT)
# 0.81855781 this value we got so it is moderately linearity

######################################################
# Model 2

model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()
# We got an r_squared value is 0.675 which is not as much good.It is moderatly fit
pred2=model2.predict(pd.DataFrame(wcat['Waist']))
# scatter diagram
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend('Predicted line','Observed data')
plt.show()
# Error calculation
res2=wcat.AT-pred1
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

# Now lwt change the y value instead of x
plt.scatter(x=wcat['Waist'],y=np.log(wcat['AT']),color='brown')
np.corrcoef(wcat.Waist,np.log(wcat.AT))
# Now it become 0.84090

model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.summary()
# the R-Squared value is 0.707
pred3=model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at=np.exp(pred3)
pred3_at

plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,'r')
plt.legend('Predicted line','Observed data')
plt.show()

# error calculation
res3=wcat.AT-pred3_at
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
# 38.529001758071416