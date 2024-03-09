# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:34:28 2023

@author: Kajal
"""
#
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from numpy.linalg import eig

#defining sample data
Marks=np.array([[3,4],[2,8],[6,9]])
print(Marks)

Marks_df=pd.DataFrame(Marks,columns=["Physics","Maths"])
Marks_df

plt.scatter(Marks_df["Physics"],Marks_df["Maths"])

#Make the data mean centric
#step 2:Normalization and standardization
    
Meanbycolumn=np.mean(Marks.T,axis=1)
print(Meanbycolumn)

scaled_data=Marks-Meanbycolumn

Marks.T
scaled_data

#Step 3:
    #find covarience matrix of aboved scaled data
Cov_mat=np.cov(scaled_data.T)
Cov_mat

#Step 4:
#Find the corressponding eigen values and eigen vector
#of above covarience matrix

# Calculate the eigenvalues (Eval) and eigenvectors (Evec) of the covariance matrix (Cov_mat)
Eval, Evec = eig(Cov_mat)

# Print the eigenvalues
print(Eval)

# Print the eigenvectors
print(Evec)


#Get the original data projected to principal components as new axis
# Project the scaled data onto the eigenvectors to obtain the projected data
# Evec.T.dot(scaled_data.T) performs the dot product of the transpose of the eigenvectors matrix (Evec.T)
# and the transpose of the scaled data matrix (scaled_data.T)
Projected_data = Evec.T.dot(scaled_data.T)

# Print the projected data
print(Projected_data)

#the result will give the Pc1 and pc2

# Import the PCA (Principal Component Analysis) module from scikit-learn
from sklearn.decomposition import PCA

# Create an instance of PCA with 2 principal components
pca = PCA(n_components=2)

# Fit the PCA model and transform the data to the new coordinate system
# Marks is the input data for PCA transformation
# The fit_transform() method calculates the principal components and applies the dimensionality reduction
# The transformed data will have 2 columns (since n_components=2)
transformed_data = pca.fit_transform(Marks)
