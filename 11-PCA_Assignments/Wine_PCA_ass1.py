# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:14:37 2023

@author: Kajal  
"""
'''
Business Objective:

maximize: A composite measure of wine quality, potentially derived from expert ratings or sensory evaluations.

minimize: production cost of the wine so it will helpful for revenue 

constrains: consistency of Brand Image to compit with the other companies

'''
'''
DataFrame : 
    
    
### Nominal Columns:

1. 'Type':
   - Description: Nominal variable representing the type or class of wine.
   - Categories: e.g., 1, 2, 3, or 'Red', 'White', 'Rose'.

### Ordinal Columns:

2. 'Alcohol':
   - Description: Ordinal variable representing the alcohol content of the wine.
   - Order: Increasing alcohol content.

3. 'Malic':
   - Description: Ordinal variable representing the malic acid content.
   - Order: Increasing malic acid content.

4. 'Ash':
   - Description: Ordinal variable representing the ash content.
   - Order: Increasing ash content.

5. 'Alcalinity':
   - Description: Ordinal variable representing the alkalinity of the wine.
   - Order: Increasing alkalinity.

6. 'Magnesium':
   - Description: Ordinal variable representing the magnesium content.
   - Order: Increasing magnesium content.

7. 'Phenols':
   - Description: Ordinal variable representing the total phenols content.
   - Order: Increasing total phenols content.

8. 'Flavanoids':
   - Description: Ordinal variable representing the flavonoids content.
   - Order: Increasing flavonoids content.

9. 'Nonflavanoids':
   - Description: Ordinal variable representing the non-flavanoids content.
   - Order: Increasing non-flavanoids content.

10. 'Proanthocyanins':
    - Description: Ordinal variable representing the proanthocyanins content.
    - Order: Increasing proanthocyanins content.

11. 'Color':
    - Description: Ordinal variable representing the color intensity.
    - Order: Increasing color intensity.

12. 'Hue':
    - Description: Ordinal variable representing the hue of the wine.
    - Order: Increasing hue.

13. 'Dilution':
    - Description: Ordinal variable representing the dilution factor.
    - Order: Increasing dilution factor.

14. 'Proline':
    - Description: Ordinal variable representing the proline content.
    - Order: Increasing proline content.

'''
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('wine.csv')
df

#######################################
df.columns
'''
Index(['Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline'],
      dtype='object')
'''
#######################################
df.dtypes
'''
Type                 int64
Alcohol            float64
Malic              float64
Ash                float64
Alcalinity         float64
Magnesium            int64
Phenols            float64
Flavanoids         float64
Nonflavanoids      float64
Proanthocyanins    float64
Color              float64
Hue                float64
Dilution           float64
Proline              int64
dtype: object

All the  datatype is of numeric type
'''

#######################################
a=df.describe()
# there scale diffrence in mean and std and the mean and median are  near approx same
# but show some variaion as the standard deviation is showing diffrence with mean
# so the datapoints are scatter from median

#####################################
# Check for the null values

n=df.isnull()
n.sum()
'''
Out[39]: 
Type               0
Alcohol            0
Malic              0
Ash                0
Alcalinity         0
Magnesium          0
Phenols            0
Flavanoids         0
Nonflavanoids      0
Proanthocyanins    0
Color              0
Hue                0
Dilution           0
Proline            0
dtype: int64

The dataframe doesn't contain any null value'
'''
#####################################
# Visualize the Data
# To undestand the corelaion between the datapoints and the columns we plot
# some plots

import seaborn as  sns

sns.pairplot(df)
# from the pairplot observe that the data is more scatter and the relation between
# the columns are quite similar

# To identify if there is any outlier in columns we plot the boxplot
sns.boxplot(df)
# There is and outlier present in the magnesium column

# to analyze whether the columns follow patern or not we draw the heatmap 

corr=df.corr()
sns.heatmap(corr)
# from the heamap i can understand that the diagonal colour of are same so the 
# columns follow some pattern 

#####################################
# So there is outlier is present and also the the column shows skewness propery
# and there is scale difference in mean and std so we use standardization technique as we are going to use 
# PCA

# Standardization
# initialize the scalar
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
df=scalar.fit_transform(a)
dataset=pd.DataFrame(df)
res=dataset.describe()
# in the resvariable we will see that the mean value is almost value 
#Standard deviation is zero

#########################

# Model Building

#For visualzing the cluster of  the above dataframe we  have to draw
# Dendodron first then we cluster the datapoints

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# linkage function give the hierarchical and Agglomotive clustering
 

z=linkage(dataset,method='complete',metric='euclidean')

plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering')
plt.xlabel('Index')
plt.ylabel('Disance')
#sch is help to draw 
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#appying agglomerative clustering choose 1 as a cluster from dendogram

# In dedrogram is not show the clustering it only shows how many clusters are there

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=2,linkage='complete',affinity='euclidean').fit(dataset)

#apply labels to the cluster
h_complete.labels_
# so these all are in the form of array we have to convert the Series
cluster_labels=pd.Series(h_complete.labels_)
# so these all are in the form of array we have to convert the Series
cluster_labels=pd.Series(h_complete.labels_)

df['clust']=cluster_labels
df

##########################################
# K-Means Clustering
# Import necessary libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read the data from 'wine.csv' file
df = pd.read_csv('wine.csv')

# Initialize an empty list to store Total Within Sum of Squares (TWSS) for different number of clusters
TWSS = []

# Define a range of cluster numbers from 2 to 7
k = list(range(2, 8))

# Iterate over each value of k
for i in k:
    # Initialize KMeans with current number of clusters
    kmeans = KMeans(n_clusters=i)
    
    # Fit the KMeans model to the dataset
    kmeans.fit(dataset)
    
    # Append the inertia (TWSS) to the TWSS list
    TWSS.append(kmeans.inertia_)
    
    '''
    KMeans inertia, also known as the sum of squares method, measures the distance of all data points from the centroids of the clusters.
    It quantifies the difference between observed values and predicted values.
    '''

# Plot an elbow curve to visualize the relationship between the number of clusters and TWSS
plt.plot(k, TWSS, 'ro-')
plt.xlabel('No of clusters')
plt.ylabel('Total within SS')

# Apply KMeans clustering with 3 clusters
model = KMeans(n_clusters=3)
model.fit(dataset)

# Get cluster labels for each data point
model_labels = pd.Series(model.labels_)

# Assign cluster labels to the DataFrame
df['clust'] = model_labels

# Display the first few rows of the DataFrame
df.head()

# Select only the columns related to the clustering results (attributes: 0, 1, and 2)
d = df.iloc[:, [0, 1, 2]]

# Display the selected columns
d

#################### PCA #####################

# Import necessary libraries
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# Normalize the numeric data using z-score normalization
uni_normal = scale(df)

# Perform PCA with 3 principal components
pca = PCA(n_components=3)
pca_values = pca.fit_transform(uni_normal)

# The amount of variance explained by each principal component
var = pca.explained_variance_ratio_
var

# Cumulative variance explained by the principal components
var1 = np.cumsum(np.round(var, decimals=4) * 100)
var1

# Variance plot for PCA components obtained
plt.plot(var1, color='red')

# PCA scores
pca_values

# Create a DataFrame to store the PCA scores
pca_data = pd.DataFrame(pca_values)
pca_data.columns = ['comp0', 'comp1', 'comp2']

# Concatenate cluster labels with PCA scores
final = pd.concat([df['clust'], pca_data], axis=1)

# Visualize the DataFrame
ax = final.plot(x='comp0', y='comp1', kind='scatter', figsize=(12, 8))

# Add cluster labels as text on the plot
final[['comp0', 'comp1', 'clust']].apply(lambda x: ax.text(*x), axis=1)
