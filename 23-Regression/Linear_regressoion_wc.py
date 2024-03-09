# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:49:41 2024

@author: Kajal
"""
import pandas as pd
import numpy as np
wcat=pd.read_csv("C:/Data Science/datasets/wc-at.csv")
#EDa
#1. measure the central tendency
#2. measure of dispersion
#3. Third moment business desicion
#4. Fourth moment business desicion
wcat.info()
wcat.describe()
#Graphical representetion
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)
#data is right skwed
#scatter plot
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')
#direction: positive,linearility:moderate, Strength:poor
#Now let us calculate correlation coefficient
np.corrcoef(wcat.Waist,wcat.AT)
#Let us check the direction using covar factor
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
##now let us apply the linear regrssion model
import statsmodels.formula.api as smf
#All machne learning algorithms implemented using sklearn:
    #but for this statsmodels
    #package is used because it is gives you
    #backed calculations of bits-0 and bita-1
model=smf.ols('AT~Waist',data=wcat).fit()
model.summary()

#OLS helps to find best fit model which causes
#least square error
#first you checked R squared value-0.670 ,if R square =0.8  means model is best fit
#fit if R square=0.8 to 0.6 moderate fit
#Next u check P>|t|=0 it means less than alpha
#alpha is 0.05 here the model is accepted

#Regression line
pred1=model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,"r")
plt.legend(['Actual data','Predicted data'])
plt.show()

#error calculations
res1=wcat.AT-pred1
np.mean(res1)

#it must be zero and here it 10^-14=~0
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1

#32.76.76 lesser value better the model
#how to improve this model transformation of
plt.scatter(x=np.log(wcat['Waist']),y=wcat['AT'],color='brown')
#data is linearliy scattered, direction positive, strength : poor
np.corrcoef(np.log(wcat.Waist),wcat.AT)
#r value is 0.82<0.85 hence moderate linearity
model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary
#Again check the r square value=0.67 which is less than 0.8
#p value is 0 less than 0.05
pred2=model2.predict(pd.DataFrame(wcat['Waist']))
pred2
#check wcat and pred2 from variable explorer
##scatter diagram
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,"r")
plt.legend(['Actual data','predicted data'])
#Error calculations
res2=wcat.AT-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#32.76
##there is no considerable changes
#Now let us change y value instead of x
plt.scatter(x=wcat['Waist'],y=np.log(wcat['AT']),color='orange')
np.corrcoef(wcat.Waist,np.log(wcat.AT))
#r value is 0.84 <0.85 hence moderate linearlity
model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.summary()
##Again check the R-square value=0.707 which is less than 0.8
#p value is 0.02 less than 0.05
#bita=0=.7410
#
pred3=model3.predict(pd.DataFrame(wcat['Waist']))
pred3=np.exp(pred3)
pred3
#check wcat and pred3_ at from variable explorer
##scatter diagram
#############regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,"r")
plt.legend(['predicted line','observed data'])
plt.show()
###################
#error calculations

res3=wcat.AT-pred3
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#38.52
#There are no significant change as r=0.8409 R-squared=0.707
model4=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat)
#Y is log (AT) X=Waist
model4.symmary()
#R-suarred=0.779<0.85,there is scope of improvement
#p=0.000<0.05 hence acceptable'
#bita-0=7.8241
#bita-1=0.2289
pred4=model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at=np.exp(pred4)
pred4_at
#########################################
#Regression Line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicted Line','Observed data_model3'])
plt.show()
########################################
##error calculations

res4=wcat.AT-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#32.24
#among all the model model4 is the best
###################################################
data={"model":pd.Series(["SLR","log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse

##################################################
##We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(wcat,test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#Y is log(AT) and X=Waist

final_model.summary()
#Rsquared=0.779<0.8 there is scope of improvement
#p=0.000<0.5 hence acceptable
#bita0=-7.82
#bita-1=0.2289
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at

###################################
train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at

##############################################
#Evalution on test data
test_err=test.AT-test_pred_at
test_sqr=test_err*test_err
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
##############################################
#Evalutoion on train data
train_res=train.AT-train_pred_at
train_sqr=train_res*train_res
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#########################################
#test_rmse>train_rmse#The model is overfit
