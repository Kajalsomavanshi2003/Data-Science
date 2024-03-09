# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:53:05 2024

@author: Kajal
"""
'''
2. Divide the diabetes data into train and test datasets
 and build a  Decision Tree model with 
 Outcome as the output variable. 
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

data = pd.read_csv("Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns


#Converting into binary
lb = LabelEncoder()
data[" Number of times pregnant"] = lb.fit_transform(data[" Number of times pregnant"])
data[" Plasma glucose concentration"] = lb.fit_transform(data[" Plasma glucose concentration"])
data["Diastolic blood pressure"] = lb.fit_transform(data["Diastolic blood pressure"])
data[" Triceps skin fold thickness"] = lb.fit_transform(data[" Triceps skin fold thickness"])
data[" 2-Hour serum insulin"] = lb.fit_transform(data[" 2-Hour serum insulin"])
data[" Body mass index"]=lb.fit_transform(data[" Body mass index"])
data[" Diabetes pedigree function"]=lb.fit_transform(data[" Diabetes pedigree function"])
data[" Age (years)"]=lb.fit_transform(data[" Age (years)"])
data['default'].unique()
data['default'].value_counts()
# Create a list of column names from the dataset
colnames = list(data.columns)

# Select predictors (features) and target variable
predictors = colnames[:15]  # Predictors include the first 15 columns
target = colnames[15]  # Target variable is the 16th column

# Split the data into training and testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3)

# Import the Decision Tree Classifier from scikit-learn
from sklearn.tree import DecisionTreeClassifier as DT

# Create an instance of the Decision Tree Classifier with entropy as the criterion for splitting
model = DT(criterion='entropy')

# Fit the model on the training data using predictors as input features and target as the output variable
model.fit(train[predictors], train[target])

# Make predictions on the testing data using the trained model
preds = model.predict(test[predictors])

# Display the predictions
preds

# Create a confusion matrix to evaluate the model's performance on the test dataset
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

# Calculate the accuracy of the predictions on the test dataset
np.mean(preds == test[target])

# Now, evaluate the accuracy of the model on the training dataset
preds_train = model.predict(train[predictors])

# Create a confusion matrix for the training dataset predictions
pd.crosstab(train[target], preds_train, rownames=['Actual'], colnames=['Predictions'])

# Calculate the accuracy of the predictions on the training dataset
np.mean(preds_train == train[target])
