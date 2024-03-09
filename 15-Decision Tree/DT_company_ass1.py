# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:34:04 2024

@author: Kajal
"""
'''
1.A cloth manufacturing company is interested to know about
 the different attributes contributing to high sales. 
 Build a decision tree  model with Sales as target variable 
 (first convert it into categorical 
variable).
 Business Contraints
 Interpretability: The Decision Tree model needs to be interpretable so that stakeholders can understand the factors contributing to classifying individuals as risky or good.

 Scalability: The model should be scalable to handle a large volume of data as the company expands its operations or encounters more data.

 Accuracy: While interpretability is important, the model should also be accurate in predicting whether an individual is risky or good based on their taxable income.

 Cost of Misclassification: Minimize the misclassification of individuals, especially those who are actually risky but classified as good. False negatives (misclassifying risky individuals as good) can be costly for the company in terms of fraud.

 Maximize:
 Maximize the accuracy of the Decision Tree model. 
 Higher accuracy ensures that the model can effectively
  distinguish between risky and good individuals based on 
  their taxable income.
  
 Minimize:
  Minimize the cost associated with misclassifications,
  particularly false negatives. Misclassifying risky individuals
  as good may lead to potential fraud-related losses for the
  company. Thus, minimizing false negatives is crucial to 
  mitigate such risks and associated costs.
  '''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("Company_Data.csv")

# Check for missing values
data.isnull().sum()

# Drop rows with missing values
data.dropna(inplace=True)

# Get column names
data.columns

# Drop the "Eduation" column
data = data.drop(["Eduation"], axis=1)

# Convert categorical variables into binary using LabelEncoder
lb = LabelEncoder()
data["Sales"] = lb.fit_transform(data["Sales"])
data["CompPrice"] = lb.fit_transform(data["CompPrice"])
data["Income"] = lb.fit_transform(data["Income"])
data["Advertising"] = lb.fit_transform(data["Advertising"])
data["Population"] = lb.fit_transform(data["Population"])
data["Price"] = lb.fit_transform(data["Price"])
data["ShelveLoc"] = lb.fit_transform(data["ShelveLoc"])
data["Age"] = lb.fit_transform(data["Age"])
data['default'].unique()
data['default'].value_counts()

# Get column names
colnames = list(data.columns)

# Separate predictors and target variable
predictors = colnames[:15]
target = colnames[15]

# Split data into training and testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3)

# Build Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion='entropy')
model.fit(train[predictors], train[target])

# Make predictions on the test data
preds = model.predict(test[predictors])

# Display the predictions
preds

# Create confusion matrix to evaluate the model
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

# Calculate the accuracy on the test dataset
np.mean(preds == test[target])

# Check accuracy on the training dataset
preds_train = model.predict(train[predictors])
pd.crosstab(train[target], preds_train, rownames=['Actual'], colnames=['Predictions'])
np.mean(preds_train == train[target])
