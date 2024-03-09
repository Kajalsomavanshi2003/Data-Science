# -- coding: utf-8 --
"""
Created on Mon Feb  5 15:07:35 2024

@author: Kajal
"""
# Import pandas library for data manipulation
import pandas as pd

# Read the CSV file into a DataFrame named company
company = pd.read_csv('Company_Data.csv')

# Retrieve the attributes and methods of the DataFrame object
dir(company)

# Create a new DataFrame named df using the company DataFrame
df = pd.DataFrame(company)

# Display the first few rows of the DataFrame df
df.head()

# Assign the 'Sales' column of the company DataFrame to a new column named 'Sales' in the df DataFrame
df['Sales'] = company.Sales

# Display the first 12 rows of the DataFrame df
df[0:12]

# Drop the 'Sales' column from the DataFrame X and assign it to X
X = df.drop('Sales', axis='columns')

# Assign the 'Sales' column to y
y = df.Sales

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Import the RandomForestClassifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier model with 20 decision trees
model = RandomForestClassifier(n_estimators=20)

# Train the RandomForestClassifier model on the training data
model.fit(X_train, y_train)

# Calculate the accuracy score of the model on the testing data
model.score(X_test, y_test)

# Make predictions on the testing data
y_predicted = model.predict(X_test)

# Calculate the confusion matrix to evaluate the model's performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

# Import the necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sn

# Set the figure size for the heatmap
plt.figure(figsize=(10, 7))

# Create a heatmap of the confusion matrix with annotations
sn.heatmap(cm, annot=True)

# Set the labels for the heatmap
plt.xlabel('Predicted')
plt.ylabel('True')

# Display the plot
plt.show()
