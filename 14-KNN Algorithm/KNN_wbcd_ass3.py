# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 02:14:18 2024

@author: Kajal
"""

import pandas as pd
import numpy as np

# Load the WBCD dataset
wbcd = pd.read_csv('C:/datasets/wbcd.csv')

# Display the shape of the dataset
wbcd.shape
# There are 569 rows and 32 columns

# Display the descriptive statistics of the dataset
wbcd.describe()
# In the output column, there is 'B' for benign and 'M' for malignant

# Convert 'B' to 'Benign' and 'M' to 'Malignant' in the 'diagnosis' column
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'B', 'Benign', wbcd['diagnosis'])
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'M', 'Malignant', wbcd['diagnosis'])

# Drop the first column (patient ID)
wbcd = wbcd.iloc[:, 1:32]

# Normalization function
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Apply normalization function to the dataframe
wbcd_n = norm_func(wbcd.iloc[:, 1:32])

# Define X as input and y as output
X = np.array(wbcd_n.iloc[:, :])
y = np.array(wbcd['diagnosis'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train the KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_test))
pd.crosstab(pred, y_test)

# Selecting the correct value of k
acc = []
for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    acc.append([train_acc, test_acc])

# Plotting the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], "bo-")

# The appropriate values of k with good accuracy are 3, 5, 7, and 9
# Let's check for k=7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
accuracy_score(pred, y_test)
pd.crosstab(pred, y_test)

# For k=7, there are zero false positives and good accuracy, making it an appropriate value of k
