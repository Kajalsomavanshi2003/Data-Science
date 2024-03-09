# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:50:27 2024

@author: Kajal
"""
'''In the recruitment domain, HR faces the challenge of
 predicting if the candidate is faking their salary or not.
 For example, a candidate claims to have 5 years of 
 experience and earns 70,000 per month working as a regional
 manager. The candidate expects more money than his previous
 CTC. We need a way to verify their claims 
 (is 70,000 a month working as a regional manager with an 
  experience of 5 years a genuine claim or does he/she make
  less than that?) Build a Decision Tree and Random Forest 
 model with monthly income as the target variable. 

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
import pandas as pd
'''
# Load the dataset
import pandas as pd
fraud_data = pd.read_csv('C:/2-dataset/Fraud_check.csv.xls')

# Display the first few rows of the dataframe
print(fraud_data.head())

# Create a dataframe from the dataset
df = pd.DataFrame(fraud_data)

# Rename columns for clarity
df.rename(columns={'Undergrad': 'Education', 
                   'Marital.Status': 'MaritalStatus', 
                   'Taxable.Income': 'TaxableIncome', 
                   'City.Population': 'CityPopulation',
                   'Work.Experience': 'WorkExperience', 
                   'Urban': 'IsUrban'}, inplace=True)

# Display the first 12 rows of the dataframe
print(df[:12])

# One-hot encode the categorical columns
df = pd.get_dummies(df, columns=['Education', 'MaritalStatus', 'IsUrban'], drop_first=True)

# Separate features (X) and target variable (y)
X = df.drop('TaxableIncome', axis=1)
y = df['TaxableIncome']

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