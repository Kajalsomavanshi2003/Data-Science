# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:02:14 2023

@author: Kajal

"""

#=======================================================================
##########################DATA DICTIONARY############################
#========================================================================
'''
Name of feature    Description          Type                                Relevance

ChildBks           Child Books        (Categorical)Quantitative
YouthBks           Youth Books        (Categorical)Quantitative
CookBks            Cooking Books      (Categorical)Quantitative
DoItYBks           Do It books        (Categorical)Quantitative
RefBks             Reference Books    (Categorical)Quantitative
ArtBks             Art Books          (Categorical)Quantitative
GeogBks            Geography Books    (Categorical)Quantitative
ItalCook           Itallian Books     (Categorical)Quantitative
ItalAtlas          Ital At las Books  (Categorical)Quantitative
ItalArt            Italian Art Books  (Categorical)Quantitative
Florence           Folarance Books    (Categorical)Quantitative
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:/datasets/book.csv")
df
#############################EDA###################################
df.columns
'''
'ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
       'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence'
'''

df.describe
#From describe, mean of all features is between 0 to 1
# minimum and maximum of all feature is 0 and 1 respectively.

df.size    #22000
df.shape   #(2000,11)
df.isnull
#There is no any null value for the given dataset
df.isnull().sum()
'''
ChildBks     0
YouthBks     0
CookBks      0
DoItYBks     0
RefBks       0
ArtBks       0
GeogBks      0
ItalCook     0
ItalAtlas    0
ItalArt      0
Florence     0

There is 0 null values 
for the given dataset
'''
df.count

####HISTOGRAM###
sns.histplot(df['ChildBks'],kde=True)   # Number of Books Present are 846
sns.histplot(df['YouthBks'],kde=True)   # Number of Books Present are 494
sns.histplot(df['CookBks'],kde=True)    # Number of Books Present are 862
sns.histplot(df['DoItYBks'],kde=True)   # Number of Books Present are 569
sns.histplot(df['RefBks'],kde=True)     # Number of Books Present are 431
sns.histplot(df['ArtBks'],kde=True)     # Number of Books Present are 484
sns.histplot(df['GeogBks'],kde=True)    # Number of Books Present are 552
sns.histplot(df['ItalCook'],kde=True)   # Number of Books Present are 223
sns.histplot(df['ItalAtlas'],kde=True)  # Number of Books Present are 78
sns.histplot(df['ItalArt'],kde=True)    # Number of Books Present are 104
sns.histplot(df['Florence'],kde=True)   # Number of Books Present are 214

###Boxplot###
sns.boxplot(x=df['ChildBks'])  #no any outliers'
sns.boxplot(x=df['YouthBks'])  #Outliers are present
sns.boxplot(x=df['CookBks'])   #no any outliers'
sns.boxplot(x=df['DoItYBks'])  #no any outliers'
sns.boxplot(x=df['RefBks'])    #Outliers are present
sns.boxplot(x=df['ArtBks'])    #Outliers are present
sns.boxplot(x=df['GeogBks'])   #no any outliers'
sns.boxplot(x=df['ItalCook'])  #Outliers are present
sns.boxplot(x=df['ItalAtlas']) #Outliers are present
sns.boxplot(x=df['ItalArt'])   #Outliers are present
sns.boxplot(x=df['Florence'])  #Outliers are present

###Scatter plot####
sns.scatterplot(data=df)
#Scatter plot for all the columns

####Pairplot#####
sns.set_style("whitegrid")
sns.pairplot(data=df)
plt.show()

####mean,median,std#####
df.mean()
'''
ChildBks     0.4230
YouthBks     0.2475
CookBks      0.4310
DoItYBks     0.2820
RefBks       0.2145
ArtBks       0.2410
GeogBks      0.2760
ItalCook     0.1135
ItalAtlas    0.0370
ItalArt      0.0485
Florence     0.1085
'''

df.median()
'''
ChildBks     0.0
YouthBks     0.0
CookBks      0.0
DoItYBks     0.0
RefBks       0.0
ArtBks       0.0
GeogBks      0.0
ItalCook     0.0
ItalAtlas    0.0
ItalArt      0.0
Florence     0.0
'''

df.std()
'''
ChildBks     0.494159
YouthBks     0.431668
CookBks      0.495340
DoItYBks     0.450086
RefBks       0.410578
ArtBks       0.427797
GeogBks      0.447129
ItalCook     0.317282
ItalAtlas    0.188809
ItalArt      0.214874
Florence     0.311089
'''

###Dublicates###
dup=df.duplicated()
dup
#True-duplicated value present
#False-duplivate value are not present
sum(dup)  #1680

#========================================================================
############################DATA PEPROCESSING####################################
#there is no need to rename or remove the column
#####Normalization########
#All the data in the form of 0 and 1 so no need to normalize it


#=========================================================================
###################################Clustering##############################################
#=========================================================================
#now we apply clustering
# Import necessary libraries
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pandas as pd

# Generate linkage matrix using complete linkage method and Euclidean distance metric
df2 = linkage(df, method='complete', metric='euclidean')

# Plot dendrogram for hierarchical clustering
plt.figure(figsize=(10, 5))
plt.title("Hierarchical clustering")
sch.dendrogram(df2, leaf_rotation=0, leaf_font_size=10)
plt.show()

# Applying Agglomerative Clustering
# Initialize Agglomerative Clustering with parameters
book = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean').fit(df)

# Get cluster labels
cluster_labels = pd.Series(book.labels_)

# Assign cluster labels as a new column to DataFrame df
df['Cluster'] = cluster_labels

# Group by cluster labels and calculate means for each cluster
res = df.groupby(df.Cluster).mean()

# KMeans clustering
# Initialize an empty list to store Total Within Sum of Squares (TWSS)
TWSS = []

# Define the range of cluster numbers to explore
k_range = list(range(2, 8))

# Iterate through each number of clusters
for i in k_range:
    # Initialize KMeans with the current number of clusters
    kmeans = KMeans(n_clusters=i)
    
    # Fit KMeans to the data
    kmeans.fit(res)
    
    # Append the inertia (TWSS) to the TWSS list
    TWSS.append(kmeans.inertia_)

# Function to find the ideal cluster number using the elbow method
def find_cluster_number(TWSS):
    # Initialize an empty list to store differences in TWSS
    diff = []
    
    # Calculate differences between consecutive TWSS values
    for i in range(0, len(TWSS) - 1):
        d = TWSS[i] - TWSS[i + 1]
        diff.append(d)
    
    # Find the cluster number corresponding to the maximum difference
    max_diff = max(diff)
    k = diff.index(max_diff) + 3  # Adjust for the 0-based index and the range start
    return k

# Call the function to find the ideal cluster number
k = find_cluster_number(TWSS)
print("Cluster number is =", k)

# Initialize KMeans with the ideal cluster number
model = KMeans(n_clusters=k)

# Fit KMeans to the Agglomerative Clustering results
model.fit(book)

# Get cluster labels from the model
model_labels = pd.Series(model.labels_)

# Assign cluster labels as a new column to the DataFrame res
res['clusters'] = model_labels

# Save the results to a CSV file
res.to_csv('bookNew_kmean.csv')

# Print current working directory
import os
os.getcwd()

#===========================================================================

''' ****************************Association Rule On Book.csv**********************'''

# Import necessary libraries
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Read data from 'book.csv' file
data = pd.read_csv('book.csv')

# Display the data
data

# Count the occurrences of each item in the dataset
from collections import Counter
item_frequencies = Counter(data)

# Apply Apriori algorithm to find frequent itemsets
# min_support specifies the minimum support threshold for an itemset to be considered frequent
# use_colnames=True ensures that column names are used in the returned DataFrame
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)

# Generate association rules from the frequent itemsets
# metric="confidence" specifies the metric to evaluate the generated rules
# min_threshold=0.5 sets the minimum confidence threshold for the generated rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display the first 20 association rules
rules.head(20)

# Display the top 10 association rules based on lift in descending order
rules.sort_values('lift', ascending=False).head(10)

# Visualize the association rules as a network graph
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph from the association rules DataFrame
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')

# Draw the graph
fig, ax = plt.subplots(figsize=(14, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.title("Association Rules Network", fontsize=15)
plt.show()

###################################################

# the benefits/impact of the solution 
# By identifying books that are frequently purchased together,
# the bookstore can create curated bundles or recommendations, enhancing the overall 
# shopping experience for customers.
# By using this association rule we can stratergically placed the books together to encourage
# the customer to purchased more items which will help to increased the overall revenue

