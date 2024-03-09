# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:38:09 2023

@author: Kajal
"""
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the Excel file
uni1 = pd.read_excel("C:/datasets/University_Clustering.xlsx")

# Generate descriptive statistics for the dataset
uni1.describe()

# Display information about the dataset
uni1.info()

# Drop the "State" column from the dataset
uni = uni1.drop(["State"], axis=1)

# Import necessary libraries for PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Consider only numerical data by dropping the first column (University names)
uni_data = uni.iloc[:, 1:]

# Normalize the numerical data
uni_normal = scale(uni_data)

# Perform PCA with 6 principal components
pca = PCA(n_components=6)
pca_values = pca.fit_transform(uni_normal)

# The amount of variance that each PCA explains
var = pca.explained_variance_ratio_
var

# Cumulative variance explained by the principal components
var1 = np.cumsum(np.round(var, decimals=4) * 100)
var1

# Variance plot for PCA components obtained
plt.plot(var1, color="red")

# PCA scores
pca_values

# Create a DataFrame to store the PCA scores
pca_data = pd.DataFrame(pca_values)
pca_data.columns = ["comp0", "comp1", "comp2", "comp3", "comp4", "comp5"]

# Concatenate the University names with the first three PCA components
final = pd.concat([uni["Univ"], pca_data.iloc[:, 0:3]], axis=1)

# Scattering diagram
# Create a scatter plot of the first two PCA components
ax = final.plot(x='comp0', y='comp1', kind='scatter', figsize=(12, 8))

# Add University names as text on the plot
final[['comp0', 'comp1', 'Univ']].apply(lambda x: ax.text(*x), axis=1)
