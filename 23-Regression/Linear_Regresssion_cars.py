# -- coding: utf-8 --
"""
Created on Thu Feb 22 14:59:03 2024

@author: Kajal
"""

#multiple correlation regression analysis
import pandas as pd
import numpy as np
import seaborn as sns
cars=pd.read_csv("C:/Data Science/datasets/Cars.csv")

cars.info()
# exploratory data analysis
#1 meausure the central tendancy
#2 measure the dispersion 
#3 third moment business decision-skewness
#4 forth moment business decision -kortosis
#5 probability distribution
#6 graphical representation like histogram and boxplot
cars.describe()
#graphical representation

import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,82,1))
#data is right skewed
sns.displot(cars.HP)

#there are severaloutliers in HP columns
#similar operations expectedd for other three column
sns.distplot(cars.MPG)
#data slight left distributed
sns.distplot(cars.MPG)
#there is no outlier
sns.distplot(cars.VOL)
#data is slightly left distributed
sns.distplot(cars.SP)
#data slight right distributed

plt.distplot(cars.WT)
plt.boxplot(cars.WT)
#there is several outlier
#now let us plot joint plot joint plot is to show scatter
#histogram

import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])


#now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times each value occured
#92 HP value occured 7 times


##QQ Plot 
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist='norm',plot=pylab)
plt.show()
#MPG data is normally distributed
#There are 10 scatter plot needs to be plotted one by one
#to plot so we can use pair plot

import seaborn as sns
sns.pairplot(cars.iloc[:,:])
#Write here
#Linearity=
#direction=
#strength=
#you can check collinearity problem between the


#now check the  r value between variables
cars.corr()
#you can check SP  and HP ,r value is 0.97 and same way
#you can check  WT and VOL ,it has got 0.999 which is greater 


#now although we observed strongly corrleted pairs still
#linear Regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
#R squared value observed is 0.771<0.85
#p-values of WT and VOL is 0.814 and 0.556 which very high
#it means it is greater than 0.05 ,WT and VOL columns
#we need to ignore
#or deleted.Instead deleting 81 entries
#let us check row wise outliers
#identifyu is there any influential values
#to check you can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers
#go to data frame and check 76 th entry
#let us delete that entry

cars_new=cars.drop(cars.index[[76]])

#agian apply regression to cars_new
ml_new=smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new.summary()
#R square value is 0.819 but p values are same
#now next option to delte the column 
#but que is that whichj column is to be delted
#we already checked correlation factor r
#Wt is less so it can be deleted

#another approach is to check the collinearity 
# r square is given that value
# we will have to apply the regression w r to x1 and input 
# as x1 x2 and x3 and so on forth



rsp_hp=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsuared
vif_hp=1/(1-rsp_hp)
vif_hp

#calculatin varience INfluential factor,calculating VIF Helps 
#to X1,X2,X3,X4

rsp_wt=smf.ols('WT~HP+VOL+SP',data=cars).fit().rsuared
vif_hp=1/(1-rsp_wt)


rsp_vol=smf.ols('VOL~HP+VOL+SP',data=cars).fit().rsuared
vif_hp=1/(1-rsp_vol)


rsp_sp=smf.ols('SP~HP+VOL+SP',data=cars).fit().rsuared
vif_hp=1/(1-rsp_sp)

##vif_Wt=639.53 and vif_vol=638.80 hence vif_wt 
#is greater ,thumb rule its vif should be  not be greater than 1

#######################################################3
#Optional part
#Storing the values in dataframe
d1={'variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt]}
vif_frame=pd.DataFrame(d1)
vif_frame

##################################################333333333
#let us try WT and apply correleation to remaining three
final_ml=smf.ols('MOG~VOL+SP+HP',data=cars).fit()
final_ml.summary()
#R squared is 0.770 and p values 0.00,0.012<0.05

#prediction
pred=final_ml.predict(cars)
pred=final_m1.predict(cars)

res=final_m1.resid
sm.qqplot(res)
plt.show()

stats.probplot(res,dist='norm',plot=pylab)
plt.show()

#let us plot the residual plot ,which takes the residuals

sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()