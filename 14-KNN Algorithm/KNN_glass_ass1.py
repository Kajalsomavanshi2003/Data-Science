# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:53:46 2024

@author: Kajal
"""

'''
problem ststement no : 1
1.	A glass manufacturing plant uses different earth elements to 
design new glass materials based on customer requirements. 
For that, they would like to automate the process of classification
as it’s a tedious job to manually classify them. Help the company 
achieve its objective by correctly classifying the glass type based 
on the other features using KNN algorithm.
'''


import pandas as pd  # Importing pandas library for data manipulation
import numpy as np  # Importing numpy library for numerical operations

# Read the dataset into a DataFrame
glass = pd.read_csv("C:/Data Science/datasets/glass (1).csv")
glass  # Display the DataFrame

# Display summary statistics of the DataFrame
glass.describe()

# Display information about the DataFrame including the data types of columns
glass.info()

# Count the occurrences of each unique value in the 'Type' column
glass['Type'].value_counts()

# Define a normalization function
def norm(i):
    x = (i - i.min()) / (i.max() - i.min())  # Normalizing the data
    return x

# Apply normalization function to the dataset (excluding the 'Type' column)
glass_norm = norm(glass.iloc[:, 0:9])

# Assign input (X) and output (Y) variables
X = np.array(glass_norm.iloc[:, :])  # Input features
Y = np.array(glass['Type'])  # Output target

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# Stratified sampling is used to avoid data imbalance during splitting

# Build the KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)  # Define the KNN model with k=21
knn.fit(X_train, Y_train)  # Train the model
pred = knn.predict(X_test)  # Make predictions on the test set

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, Y_test))  # Print the accuracy score
pd.crosstab(pred, Y_test)  # Create a cross-tabulation of predicted vs. actual values

# Select the optimal value of k
acc = []
for i in range(3, 50, 2):  # Iterate through odd values of k from 3 to 49
    n = KNeighborsClassifier(n_neighbors=i)  # Define the KNN model
    n.fit(X_train, Y_train)  # Train the model
    train_acc = np.mean(n.predict(X_train) == Y_train)  # Calculate training accuracy
    test_acc = np.mean(n.predict(X_test) == Y_test)  # Calculate testing accuracy
    acc.append([train_acc, test_acc])  # Append the accuracies to the list

# Plot the graph of training and testing accuracies
import matplotlib.pyplot as plt
plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], 'ro-')  # Plot training accuracy
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], 'bo-')  # Plot testing accuracy

# Try for k=5
knn = KNeighborsClassifier(n_neighbors=5)  # Define the KNN model with k=5
knn.fit(X_train, Y_train)  # Train the model
pred = knn.predict(X_test)  # Make predictions on the test set
accuracy_score(pred, Y_test)  # Calculate and print the accuracy score
pd.crosstab(pred, Y_test)  # Create a cross-tabulation of predicted vs. actual values














