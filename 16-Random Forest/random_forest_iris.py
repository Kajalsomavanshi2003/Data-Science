# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:46:22 2024

@author: Kajal
"""
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
df['target'] = iris.target
df.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),iris.target,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
#n_estimators:number of trees in the forest
model.fit(X_train,y_train)

model.score(X_test,y_test)
y_predicted=model.predict(X_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm
                
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')