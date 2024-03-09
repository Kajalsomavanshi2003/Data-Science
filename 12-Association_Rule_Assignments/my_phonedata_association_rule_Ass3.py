# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:29:49 2023

@author: Kajal
"""
'''
Business Objective:

Maximize: The customer Satisfaction

Minimize: The product return means the failure of the productt

Cobnstrains: Resources and availability of the devices 
'''

'''
Dataframe:

['red', 'white', 'green', 'yellow', 'orange', 'blue'] all the columns is of
nominal there is no ordinal data is present in the dataset
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:/datasets/myphonedata.csv")
df
#=============================================================
##########Data Dictionary
#=============================================================
'''
Name of Feature                  Type
Red                        Quantitative data
white                      Quantitative data
green                      Quantitative data
yellow                     Quantitative data
Orange                     Quantitative data
Blue                       Quantitative data
'''
#EDA
df.columns
'''
'red', 'white', 'green', 'yellow',
 'orange', 'blue
 '''

df.describe
df.shape    #(11,6)
df.size     #66
df.isnull
df.isnull().sum()
'''
red       0
white     0
green     0
yellow    0
orange    0
blue      0
'''
df.count
##########Histogram############
sns.histplot(df['red'],kde=True) 
#data is not normally diistributed
sns.histplot(df['white'],kde=True)
#data is not normally diistributed
sns.histplot(df['green'],kde=True)
#data is not normally diistributed
sns.histplot(df['yellow'],kde=True)
#data is not normally diistributed
sns.histplot(df['orange'],kde=True)
#data is not normally diistributed
sns.histplot(df['blue'],kde=True)
#data is not normally diistributed

###########Boxplot################
sns.boxplot(df['red'])
#No outliers present
sns.boxplot(df['white'])
#No outliers present
sns.boxplot(df['green'])
#Outliers are present
sns.boxplot(df['yellow'])
#outliers are present
sns.boxplot(df['orange'])
#outliers are present
sns.boxplot(df['blue'])
#No outliers present

############ScatterPlot##############
sns.scatterplot(data=df)    
#scatter plot for all the columns

#############PairPlot################
sns.pairplot(data=df)
plt.show()
########################Mean,mode,std############################
df.mean()
'''
red       0.545455
white     0.636364
green     0.181818
yellow    0.090909
orange    0.181818
blue      0.545455
'''
df.median()
'''
red       1.0
white     1.0
green     0.0
yellow    0.0
orange    0.0
blue      1.0
'''
df.std()
'''
red       0.522233
white     0.504525
green     0.404520
yellow    0.301511
orange    0.404520
blue      0.522233
'''
#######################################################
#to check the duplicatee
df.duplicated()
#true-duplicated data
#false-no duplicated data
'''
0     False
1     False
2     False
3     False
4     False
7     False
8     False
10    False
dtype: bool
'''
df=df.drop_duplicates()
df
#============================================================
######################Data preprocessing###################
#==============================================================
#No need to remove or rename the column

###############Normalization##############
def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=normalize(df.iloc[:,:])
df_norm
#all the data in the form of 0 and 1 so we do not need to normalization
#if we normalize the data the output will be same
a=df_norm.describe()
a
'''
 red     white    green    yellow   orange      blue
count  8.000000  8.000000  8.00000  8.000000  8.00000  8.000000
mean   0.500000  0.625000  0.25000  0.125000  0.25000  0.375000
std    0.534522  0.517549  0.46291  0.353553  0.46291  0.517549
min    0.000000  0.000000  0.00000  0.000000  0.00000  0.000000
25%    0.000000  0.000000  0.00000  0.000000  0.00000  0.000000
50%    0.500000  1.000000  0.00000  0.000000  0.00000  0.000000
75%    1.000000  1.000000  0.25000  0.000000  0.25000  1.000000
max    1.000000  1.000000  1.00000  1.000000  1.00000  1.000000
'''

#now all the data will be normally distributed 

#now we apply clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
df2=linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(10,5))
plt.title(" Hierarchical clustering")
sch.dendrogram(df2,leaf_rotation=0,leaf_font_size=10)
plt.show()


#applying agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
phone=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(df_norm)
phone.labels_
cluster_labels=pd.Series(phone.labels_)
df_norm['Cluster']=cluster_labels
res=df_norm.groupby(df_norm.Cluster).mean()
res


# Model Building
# Association Rules
# Import necessary libraries
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Read data from 'myphonedata.csv' file
data = pd.read_csv('myphonedata.csv')

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
# By using this data, we can suggest to the customer which color they should select for the mobile.
