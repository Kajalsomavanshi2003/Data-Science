# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:23:56 2024

@author: Kajal
"""



import pandas as pd

# Load the dataset
hr_data = pd.read_csv('C:/Data Science/datasets/HR_DT (1).csv.xls')

# Display the first few rows of the dataframe
print(hr_data.head())

# Create a dataframe from the dataset
df = pd.DataFrame(hr_data)

# Rename columns for clarity
df.rename(columns={'Position of the employee': 'Position', 
                   'no of Years of Experience of employee': 'Experience', 
                   ' monthly income of employee': 'Income'}, inplace=True)

# Display the first 12 rows of the dataframe
print(df[:12])

# One-hot encode the 'Position' column
df = pd.get_dummies(df, columns=['Position'], drop_first=True)

# Separate features (X) and target variable (y)
X = df.drop('Income', axis=1)
y = df['Income']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
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