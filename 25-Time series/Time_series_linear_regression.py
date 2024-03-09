# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:25:08 2024

@author: kajal
"""
import pandas as pd
walmart = pd.read_csv("C:/Data Science/datasets/Walmart Footfalls Raw.csv")

#pre processing
import numpy as np

walmart["t"]=np.arange(1,160)

walmart["t_square"]=walmart["t"]  * walmart["t"]
walmart["log_footfalls"]=np.log(walmart["Footfalls"])
walmart.columns

#month=jan,feb,march, april.......dec
#in walmart data we have jan-1991 in 0th column,we need only first
#example jan from each cell
p=walmart["Month"][0]
#before we will extract let us create new column called
#month to stored extracted value
p[0:3]

walmart['months']=0
#you can check the dataframe with months name with all values 0
#the total records  are 159 in walmart

for i in range(159):
    p=walmart["Month"][i]
    walmart['months'][i]=p[0:3]
    
month_dummies=pd.DataFrame(pd.get_dummies(walmart['months']))
#now let us concatenate these dummy values to dataframe
walmart1=pd.concat([walmart,month_dummies],axis=1)
#u can check the dataframe walmart1

#visualization -Time plot
walmart1.Footfalls.plot()
Train=walmart1.head(147)
Test=walmart1.tail(12)

#to change the index value in pandas data frame
#Test.seet_index(np.arange(1,13))


#################L I N E A R####################
import statsmodels.formula.api as smf
linear_model=smf.ols('Footfalls ~ t',data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_linear)) **2 ))
rmse_linear

#########        Exponential  #################


Exp=smf.ols('log_footfalls ~ t',data=Train).fit()
pred_Exp=pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_Exp =np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_linear))**2))

rmse_Exp

################## Quadratic #################

Quad=smf.ols('Footfalls ~ t + t_square' ,data=Train).fit()
pred_Quad=pd.Series(Quad.predict(Test[['t','t_square']]))
rmse_Quad =np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_linear))**2))

rmse_Quad

############
##################Additive seasonality##############
add_sea=smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea=pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_add_sea))**2))
rmse_add_sea

#################3Multiplicative Seasonality###############
Mul_sea=smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_Mul_sea=pd.Series(Mul_sea.predict(Test))
rmse_Mul_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_Mul_sea)))**2))
rmse_Mul_sea

##################Addiitive Seasonality Quadratic tred##########3
add_sea_quad=smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad=pd.Series(add_sea_quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
####################Multiplicative seasonality linear tred###############
Mul_add_sea=smf.ols('log_footfalls ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_Mul_add_sea=pd.Series(Mul_add_sea.predict(Test))
rmse_Mul_add_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_Mul_add_sea)))**2))
rmse_Mul_add_sea

###############Testing######################

data={"MODEL":pd.Series(["rmse_linear","rmse_quad","rmse_Exp","rmse_add_sea","rmse_add_sea_quad","rmse_Mul_sea","rmse_Mul_add_sea"])}
table_rmse=pd.DataFrame(data)
table_rmse

#rmse_add_sea has the lowest value

###################Testing######################
predict_data=pd.read_excel("C:/Data Science/datasets/Predict_new.xlsx")
model_full=smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_new=pd.Series(model_full.predict(predict_data))
pred_new=pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Footfalls"]=pd.Series(pred_new)

#Autoregression model(AM)
#Calculating residuals from best model applied on full data
#AV-FV
full_res=walmart1.Footfalls-model_full.predict(walmart1)

#ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res,lags=12)

#ACF is an (complete) auto correlation function gives values
#of auto correlation of any time series with its lagged values

#PACF is a partial auto-correlation function
#It finds correlation present with lags of the residualas
tsa_plots.plot_pacf(full_res,lags=12)

#Alternative approach of ACF plot
#from pandas.plotting import autocorrelation plot
#autocorrelation_ppyplot.show()

#AR model

from statsmodels.tsa.ar_model import AutoReg
model_ar=AutoReg(full_res,lags=[1])
#model_arAutoReg(Train_res,lags=12)
model_fit=model_ar.fit

print('Coefficients; %s'% model_fit.params)

pred_res=model_fit.predict(start=len(full_res),end=len(full_res)+len(predict_data)-1,dynamic=False)
pred_res.reset_index(drop=True,inplace=True)

#The Final Predications using ASQT and AR(1) model
final_pred=pred_new+pred_res
final_pred

