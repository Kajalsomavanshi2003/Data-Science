# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:51:45 2024

@author: Kajal
"""

'''Build a Decision Tree  on the fraud data. Treat those who have taxable_income <= 30000 as Risky
 and others as Good (discretize the taxable income column).
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

# Load the dataset
hr_data = pd.read_csv('C:/Data Science/datasets/HR_DT (1).csv.xls')

# Create a dataframe from the dataset
df = pd.DataFrame(hr_data)

# Rename the target column
df.rename(columns={'Sales': 'Target'}, inplace=True)

# Separate features (X) and target variable (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Decision Tree Classifier model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
model_score = model.score(X_test, y_test)
print("Model Accuracy:", model_score)

# Predict using the trained model
y_predicted = model.predict(X_test)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)

# Visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()