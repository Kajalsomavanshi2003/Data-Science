# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:31:25 2024

@author: Kajal
"""

import pandas as pd
df=pd.read_csv("D:/Data Science/6-Datasets/salaries.csv.xls")
df.head()
inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
input_n=inputs.drop(['company','job','degree'],axis='columns')
target
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(input_n,target)
#Is salary of google,computer engineer,bachelors degree>100?
model.predict([[2,1,0]])
#Is salary of google,computer enginerr,master degree?100?
model.predict([[2,1,1]])