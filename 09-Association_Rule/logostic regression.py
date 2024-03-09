# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:55:16 2024

@author: Kajal
"""

import pandas as pd
import numpy as np
import seaborn as sns
cars=pd.read_csv('c:/2-dataset/Cars.csv')
cars.describe()
#graphical representation
import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arrange(1,82,1))
sns.distplot(cars.HP)
#data is right skewed
plt.boxplot(cars.HP)
##there are several outliers in HP  columns
#similar oreantation are ecpected for other three columns 
sns.distplot(cars.MGP)
#data is slighty left distributed 
plt.boxplot(cars.MGP)
#there are no outliers 

plt.boxplot(cars.VOL)
sns.distplot(cars.SP)
#data is slightly right distributed
plt.boxplot(cars.WT)
plt.boxplot(cars.WT)
##There are several outliers 
#Now let us plot joint plot, joint plot is to show scat
#histogram
import seaborn as sns
sns.jointplot(x = cars['HP'], y=cars['MPG'])

#now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars["HP"])
#count plot shows how many times the each value occures
#92 hp value occured 7 times

##QQ plot 
from scipy import stats 
import pylab
stats.probplot(cars.MPG, dist='norm', plot=pylab)
plt.show()
#MPG data is normally distributed
#There are 10 scatter plots need to be plotted one by 
#to plot so we can use pair plots
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
#you can check the collinearity problem between the
#you can check plot between SP and HP they are strong
#same way you can  check the collinearity problem between the inputs 
# you can check plot between SP and HP , r values is 0.97 and same way
#you can check WT and VOl , it has got 0.999
#which is grater
# Now although we observed strongly 4correlated pairs
#still we go for linear regression 
import statsmodels.formula.api as smf
ml1=smf.ols('MGP~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
#R square values observed is 0.771<0.85
#p~values of WT and VOL is 0.814 and 0.556 which is  very high 
#it means it is greater than 0.05 , WT and vol columns
#we need to ignore 
#or delete .instead deletiing 81 entries 
# let u check the row wise outlier 
#identifying is there any influential value
#to check you can use imflunential index  
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers 
#go to data frame and check 76 th entry 
#let us delete that entry '
cars_new=cars.drop(cars.index())
#again apply regression to cars new


















#question is which column is to be deleted '
#we  have alredy
#another approach is to check the collinearity 
#rsquare is giving that 

rsq_hp=smf.ols()





rsq_sp=smf.ols('SP~HP+WT+VOL',data=cars).fit().rsquared
vif_sp=1/(1-rsq_sp)
#vif_WT=639.53 adn vif_vol=638.80 hence vif_wt

pred=final_ml.resid
sm.qqplot(res)
plt.show()

#this QQ plot  is on residule which is obtained on training 
stats.probplot(res,dis)







































































































import seaborn as sns
sns.pairplot(cars.iloc[:,:])
#leniarity for each 
#directtion and strength







































