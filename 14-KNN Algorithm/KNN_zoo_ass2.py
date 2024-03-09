# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:24:26 2024

@author: Kajal
"""
'''
PROBLEM STATEMENT 2 
A National Zoopark in India is dealing with the problem of 
segregation of the animals based on the different attributes
 they have. Build a KNN model to automatically classify the 
 animals. Explain any inferences you draw in the documentation.
'''

import pandas as pd
import numpy as np

# Load the Zoo dataset
zoo = pd.read_csv("C:/Data Science/datasets/Zoo.csv")

# Display the dataset
zoo

# Describe the dataset to understand its distribution
zoo.describe()

# Get information about the dataset
zoo.info()

# Check the distribution of values in the 'type' column
zoo['type'].value_counts()

# Remove the 'animal name' column as it is not important
zoo = zoo.drop('animal name', axis=1)

# Normalization function
def norm(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Apply normalization function to the dataset
zoo_norm = norm(zoo.iloc[:, 0:16])

# Define X as input and Y as output
X = np.array(zoo_norm.iloc[:, :])
Y = np.array(zoo['type'])

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Build and train the KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, Y_test))
pd.crosstab(pred, Y_test)

# Selecting the correct value of k
acc = []
for i in range(3, 50, 2):
    n = KNeighborsClassifier(n_neighbors=i)
    n.fit(X_train, Y_train)
    train_acc = np.mean(n.predict(X_train) == Y_train)
    test_acc = np.mean(n.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

# Plotting the graph of train_acc and test_acc
import matplotlib.pyplot as plt

# Plot the graph for train accuracy (red circles) and test accuracy (blue circles)
plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], 'ro-')  # Train accuracy
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], 'bo-')  # Test accuracy

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the KNN classifier on the training data
knn.fit(X_train, Y_train)

# Predict the labels for the test data using the trained KNN classifier
pred = knn.predict(X_test)

# Compute the accuracy score by comparing the predicted labels with the actual labels of the test data
accuracy_score(pred, Y_test)

# Generate a cross-tabulation (contingency table) between the predicted labels and the actual labels of the test data
pd.crosstab(pred, Y_test)


# This model seems to be perfect
