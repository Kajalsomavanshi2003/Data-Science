# -- coding: utf-8 --
"""
Created on Mon Feb  5 15:38:48 2024

@author: Kajal
"""
# Import pandas library for data manipulation
import pandas as pd

# Load the dataset from the given path
fraud_data = pd.read_csv('path/to/Fraud_check.csv.xls')

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
