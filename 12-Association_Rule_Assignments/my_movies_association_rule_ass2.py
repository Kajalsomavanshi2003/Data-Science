# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:44:09 2023

@author: Kajal
"""
'''
Business Objective:

Maximize: Audience engagement,profit of the movie

Minimize: Production time and production cost

ContraintsL: The business may face constraints related to market competition

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:/datasets/my_movies.csv")
df
df.columns
'''
'Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
       'LOTR2', 'Harry Potter2', 'LOTR',
       'Braveheart', 'Green Mile'
'''
#=============================================================
########################Data Dictionary#############
#===============================================================
'''
Name of feature                    Type
Sixth Sense                     Quantitative
Gladiator                       Quantitative
LOTR1                           Quantitative
Harry Potter1                   Quantitative
Patriot                         Quantitative
LOTR2                           Quantitative
Harry Potter2                   Quantitative
LOTR                            Quantitative
Braveheart                      Quantitative
Green Mile                      Quantitative
'''

#======================================================================
####################EDA###########################
#=======================================================================
df.columns
'''
'Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
       'LOTR2', 'Harry Potter2', 'LOTR',
       'Braveheart', 'Green Mile'
'''
df.size    #100
df.shape   #10,10
df.describe
df.head()  
#To display the top 5 rows
df.tail()
#to display the bottom 5 rows
df.isnull
df.isnull().sum()
'''
Sixth Sense      0
Gladiator        0
LOTR1            0
Harry Potter1    0
Patriot          0
LOTR2            0
Harry Potter2    0
LOTR             0
Braveheart       0
Green Mile       0
There is no null value from the given dataset
'''
#####Histogram
sns.histplot(df['Sixth Sense'],kde=True) #data is not normally distributed
sns.histplot(df['Gladiator'],kde=True)  ##data is not normally distributed
sns.histplot(df['LOTR1'],kde=True)     ##data is not normally distributed
sns.histplot(df['Harry Potter1'],kde=True) #data is not normally distributed
sns.histplot(df['Patriot'],kde=True)    #data is not normally distributed
sns.histplot(df['LOTR2'],kde=True)   #data is not normally distributed
sns.histplot(df['Harry Potter2'],kde=True)  #data is not normally distributed
sns.histplot(df['LOTR'],kde=True)   #data is not normally distributed
sns.histplot(df['Braveheart'],kde=True)  #data is not normally distributed
sns.histplot(df['Green Mile'],kde=True) #data is not normally distributed

####Boxplot
sns.boxplot(x=df['Sixth Sense'])   #No outliers
sns.boxplot(x=df['Gladiator'])     #No outliers
sns.boxplot(x=df['LOTR1'])         #outliers are present
sns.boxplot(x=df['Harry Potter1']) #outliers are present
sns.boxplot(x=df['Patriot'])       #No outliers
sns.boxplot(x=df['LOTR2'])         #outliers are present
sns.boxplot(x=df['Harry Potter2']) #outliers are present
sns.boxplot(x=df['LOTR'])          #Outliers are present
sns.boxplot(x=df['Braveheart'])    #outliers are present
sns.boxplot(x=df['Green Mile'])    #outliers are present

####Scatterplot
sns.scatterplot(data=df)
#Draw the scatter plot for all the columns

#####Pairplot
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# No Datapoints are corelated as the all the datapoints are in scatter form 

#######################Meean,median,std####################################
df.mean()
'''
Sixth Sense      0.6
Gladiator        0.7
LOTR1            0.2
Harry Potter1    0.2
Patriot          0.6
LOTR2            0.2
Harry Potter2    0.1
LOTR             0.1
Braveheart       0.1
Green Mile       0.2
dtype: float64
'''
df.median()
'''
Sixth Sense      1.0
Gladiator        1.0
LOTR1            0.0
Harry Potter1    0.0
Patriot          1.0
LOTR2            0.0
Harry Potter2    0.0
LOTR             0.0
Braveheart       0.0
Green Mile       0.0
dtype: float64
'''
df.std()
'''
Sixth Sense      0.516398
Gladiator        0.483046
LOTR1            0.421637
Harry Potter1    0.421637
Patriot          0.516398
LOTR2            0.421637
Harry Potter2    0.316228
LOTR             0.316228
Braveheart       0.316228
Green Mile       0.421637
dtype: float64
'''
#=======================================================================
############Data preprocessing################
#=======================================================================
#Here no need to remove or rename the columnn
#############Normalizationn
#all the data are range between 0 and 1 or standard deviation
#near to 0 so no need to normalize it

#========================================================================
##################Clustering##################
#========================================================================
# Import necessary libraries
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Generate linkage matrix using complete linkage method and Euclidean distance metric
z = linkage(df, method="complete", metric="euclidean")

# Plot dendrogram for hierarchical clustering
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show()

# Import AgglomerativeClustering from sklearn.cluster
from sklearn.cluster import AgglomerativeClustering

# Initialize Agglomerative Clustering with parameters
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean').fit(df)

# Get cluster labels from the fitted model
cluster_labels = pd.Series(h_complete.labels_)

# Display the cluster labels
cluster_labels

# Assign cluster labels as a new column to DataFrame df
df['cluster'] = cluster_labels

# Display columns of DataFrame df
df.columns

# Display shape of DataFrame df
df.shape

# Create a copy of DataFrame df
df = df.iloc[:, :]

# Display columns of DataFrame df
df.columns

# Group by cluster labels and calculate means for each cluster for columns starting from the 3rd index
df.iloc[:, 2:].groupby(df.cluster).mean()

#Normalization
#The data is numeric one so we have to perform normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(df)
df_norm

b=df_norm.describe()

sns.boxplot(df_norm)
# No Outlier is remaining
# The all the quantile points are converted in the rande of 0-1
###########################################
# Model Building
# Association Rules
# Import necessary libraries
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Read data from 'my_movies.csv' file
data = pd.read_csv('my_movies.csv')

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

# Explanation of the benefits/impact of the solution
# By using the association rules, we can suggest movies to customers to increase viewership.
# This can lead to increased revenue and popularity of the movies.







