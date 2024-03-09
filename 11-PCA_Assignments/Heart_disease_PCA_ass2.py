# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:30:46 2023

@author: Kajal
"""
'''
1. Business problem

Business Objective:

Maximize:  Reduce the number of cases where the model 
fails to identify individuals with heart disease.

Minimize:  Accuracy, sensitivity, specificity, or an appropriate 
combination of these metrics depending on the business priorities.

Constraints: Maintaining the result,Data Privacy and Ethics



DataFrame:

 Sure, I'll categorize the columns into nominal and ordinal based on their characteristics. 

### Nominal Columns:

1.'sex': 
   - Description: Nominal variable representing gender.
   - Categories: 0 = Female, 1 = Male

2.'cp' (chest pain type):
   - Description: Nominal variable representing the type of chest pain.
   - Categories: 
      - 0 = Typical angina
      - 1 = Atypical angina
      - 2 = Non-anginal pain
      - 3 = Asymptomatic

3. 'fbs' (fasting blood sugar):
   - *Description:* Nominal variable indicating whether fasting blood sugar is greater than 120 mg/dl.
   - *Categories:* 0 = False, 1 = True

4. **'restecg' (resting electrocardiographic results):**
   - Description: Nominal variable representing resting electrocardiographic results.
   -Categories:
      - 0 = Normal
      - 1 = ST-T wave abnormality
      - 2 = Left ventricular hypertrophy

5. 'exang' (exercise induced angina):
   - Description: Nominal variable indicating whether angina was induced by exercise.
   - Categories: 0 = No, 1 = Yes

6. 'slope':
   - Description: Nominal variable representing the slope of the peak exercise ST segment.
   -Categories:
      - 0 = Upsloping
      - 1 = Flat
      - 2 = Downsloping

7. 'ca' (number of major vessels colored by fluoroscopy):
   - Description: Nominal variable indicating the number of major vessels colored by fluoroscopy.
   - Categories: 0, 1, 2, 3, 4

8. 'thal':
   - Description: Nominal variable representing Thallium stress test result.
   - Categories:
      - 0 = Normal
      - 1 = Fixed defect
      - 2 = Reversible defect

9. 'target':
   - Description: Nominal variable indicating the presence or absence of heart disease.
   - Categories: 0 = No heart disease, 1 = Heart disease

### Ordinal Columns:

1. 'age':
   - Description:*Ordinal variable representing age.
   - Order: Increasing age.

2.'trestbps' (resting blood pressure):
   - Description: Ordinal variable representing resting blood pressure.
   - Order: Increasing resting blood pressure.

3. 'chol' (serum cholesterol):
   - Description: Ordinal variable representing serum cholesterol levels.
   - Order: Increasing cholesterol levels.

4. 'thalach' (maximum heart rate achieved):
   - Description: Ordinal variable representing the maximum heart rate achieved.
   - Order: Increasing maximum heart rate.

5. 'oldpeak' (ST depression induced by exercise relative to rest):
   - Description: Ordinal variable representing ST depression induced by exercise relative to rest.
   - Order: Increasing ST depression.


'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("C:/datasets/heart disease.csv")
df

#=================Data dictionary====================
'''
Name of feature                                    description
age                                          Nominal categorical data    
sex                                          Nominal categorical data    
cp                                           Nominal categorical data    
trestbps                                     Nominal categorical data    
chol                                         Nominal categorical data   
fbs                                          Nominal categorical data
restecg                                      Nominal categorical data
thalach                                      Nominal categorical data       
exang                                        Nominal categorical data    
oldpeak                                      Nominal categorical data  
slope                                        Nominal categorical data       
ca                                           Nominal categorical data    
thal                                         Nominal categorical data    
target                                       Nominal categorical data   
'''

#===================================================================
###################################EDA############################
#======================================================================
df.columns      
'''
Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
      dtype='object')
'''
df.size  # 4242
df.shape # (303, 14)
df.head  # Display top 5 rows
df.tail  #display bottom 5 rows
df.describe
'''
<bound method NDFrame.describe of      age  sex  cp  trestbps  chol  fbs  ...  exang  oldpeak  slope  ca  thal  target
0     63    1   3       145   233    1  ...      0      2.3      0   0     1       1
1     37    1   2       130   250    0  ...      0      3.5      0   0     2       1
2     41    0   1       130   204    0  ...      0      1.4      2   0     2       1
3     56    1   1       120   236    0  ...      0      0.8      2   0     2       1
4     57    0   0       120   354    0  ...      1      0.6      2   0     2       1
..   ...  ...  ..       ...   ...  ...  ...    ...      ...    ...  ..   ...     ...
298   57    0   0       140   241    0  ...      1      0.2      1   0     3       0
299   45    1   3       110   264    0  ...      0      1.2      1   0     3       0
300   68    1   0       144   193    1  ...      0      3.4      1   2     3       0
301   57    1   0       130   131    0  ...      1      1.2      1   1     3       0
302   57    0   1       130   236    0  ...      0      0.0      1   1     2       0
'''
df.dtypes
'''
ge           int64
sex           int64
cp            int64
trestbps      int64
chol          int64
fbs           int64
restecg       int64
thalach       int64
exang         int64
oldpeak     float64
slope         int64
ca            int64
thal          int64
target        int64
dtype: object
'''
df.isnull()
#True-null value
#false-no null value
df.isnull().sum()
'''
age         0
sex         0
cp          0
trestbps    0
chol        0
fbs         0
restecg     0
thalach     0
exang       0
oldpeak     0
slope       0
ca          0
thal        0
target      0
dtype: int64
#No null valuess 
'''
#############################Histogram#####################
sns.histplot(df['age'],kde=True) 
#data is not normally distributed
sns.histplot(df['sex'],kde=True) 
#total 207 males
sns.histplot(df['cp'],kde=True) 
#data is not normally distributed
sns.histplot(df['trestbps'],kde=True) 
#Right skwed data
sns.histplot(df['chol'],kde=True) 
#Right skwed data
sns.histplot(df['fbs'],kde=True) 
sns.histplot(df['thalach'],kde=True) 
#Left skwed data
sns.histplot(df['exang'],kde=True) 
sns.histplot(df['oldpeak'],kde=True) 
#Right skwed data
sns.histplot(df['slope'],kde=True) 
sns.histplot(df['ca'],kde=True) 
#Right skwed
sns.histplot(df['thal'],kde=True) 
#Left skwed data
sns.histplot(df['target'],kde=True) 

####################Boxplot#############################
sns.boxplot(x=df['age'])       # No Outliers
sns.boxplot(x=df['sex'])       # No Outliers
sns.boxplot(x=df['cp'])        # No Outliers
sns.boxplot(x=df['trestbps'])  # Outliers are present
sns.boxplot(x=df['chol'])      #outliers are present
sns.boxplot(x=df['fbs'])       #outliers are present
sns.boxplot(x=df['thalach'])   #Outliers are present
sns.boxplot(x=df['exang'])     #No outliers
sns.boxplot(x=df['oldpeak'])   #Outliers are present
sns.boxplot(x=df['slope'])     #Outliers are present
sns.boxplot(x=df['ca'])        #Outliers are present
sns.boxplot(x=df['thal'])      #Outliers are present
sns.boxplot(x=df['target'])    #No outliers

####################Pairplot############################
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

#==============================================================
#####################Data Preprocessing########################
#=============================================================
#no need to remove or rename columnn
################Normalization############################
def normalize(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=normalize(df.iloc[:,:])
df_norm
'''
t[28]: 
          age  sex        cp  trestbps  ...  slope    ca      thal  target
0    0.708333  1.0  1.000000  0.481132  ...    0.0  0.00  0.333333     1.0
1    0.166667  1.0  0.666667  0.339623  ...    0.0  0.00  0.666667     1.0
2    0.250000  0.0  0.333333  0.339623  ...    1.0  0.00  0.666667     1.0
3    0.562500  1.0  0.333333  0.245283  ...    1.0  0.00  0.666667     1.0
4    0.583333  0.0  0.000000  0.245283  ...    1.0  0.00  0.666667     1.0
..        ...  ...       ...       ...  ...    ...   ...       ...     ...
298  0.583333  0.0  0.000000  0.433962  ...    0.5  0.00  1.000000     0.0
299  0.333333  1.0  1.000000  0.150943  ...    0.5  0.00  1.000000     0.0
300  0.812500  1.0  0.000000  0.471698  ...    0.5  0.50  1.000000     0.0
301  0.583333  1.0  0.000000  0.339623  ...    0.5  0.25  1.000000     0.0
302  0.583333  0.0  0.333333  0.339623  ...    0.5  0.25  0.666667     0.0
'''
#normalizze the data. Range between 0 and 1
#And standard deviation near to 0
a=df_norm.describe()
a
'''
              age         sex          cp  ...          ca        thal      target
count  303.000000  303.000000  303.000000  ...  303.000000  303.000000  303.000000
mean     0.528465    0.683168    0.322332  ...    0.182343    0.771177    0.544554
std      0.189210    0.466011    0.344017  ...    0.255652    0.204092    0.498835
min      0.000000    0.000000    0.000000  ...    0.000000    0.000000    0.000000
25%      0.385417    0.000000    0.000000  ...    0.000000    0.666667    0.000000
50%      0.541667    1.000000    0.333333  ...    0.000000    0.666667    1.000000
75%      0.666667    1.000000    0.666667  ...    0.250000    1.000000    1.000000
max      1.000000    1.000000    1.000000  ...    1.000000    1.000000    1.000000
'''

#==================================================================
######################Clustering################################
#===================================================================
# Import necessary libraries
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate linkage matrix using complete linkage method and Euclidean distance metric
z = linkage(df, method="complete", metric="euclidean")

# Plot dendrogram for hierarchical clustering
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show()

# Initialize Agglomerative Clustering with parameters
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean').fit(df)

# Get cluster labels from the fitted model
cluster_labels = pd.Series(h_complete.labels_)

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

# Initialize KMeans with different numbers of clusters and calculate Total Within Sum of Squares (TWSS)
TWSS = []
k = list(range(2, 8))
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    TWSS.append(kmeans.inertia_)

# Plot an elbow curve to determine the optimal number of clusters
plt.plot(k, TWSS, 'ro-')
plt.xlabel('No of clusters')
plt.ylabel('Total within SS')

# Initialize KMeans with the optimal number of clusters and fit the model
model = KMeans(n_clusters=3)
model.fit(df)

# Get cluster labels from the model
model_labels = pd.Series(model.labels_)

# Assign cluster labels as a new column to DataFrame df
df['clust'] = model_labels

# Display the DataFrame with the cluster labels
df.head()

# Select only specific columns from the DataFrame
d = df.iloc[:, [0, 1, 2]]



#################### PCA #####################

# Import necessary libraries
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import pandas as pd

# Read the data from 'heart disease.csv' file
df = pd.read_csv('heart disease.csv')

# Display the data
df

# Normalize the numeric data using z-score normalization
uni_normal = scale(df)

# Print the normalized data
uni_normal

# Initialize PCA with 3 principal components
pca = PCA(n_components=3)

# Fit PCA to the normalized data and transform it to obtain the PCA scores
pca_values = pca.fit_transform(uni_normal)

# The amount of variance explained by each principal component
var = pca.explained_variance_ratio_

# Print the explained variance ratio for each principal component
var

# Cumulative variance explained by the principal components
var1 = np.cumsum(np.round(var, decimals=4) * 100)

# Print the cumulative variance
var1

# Variance plot for PCA components obtained
plt.plot(var1, color='red')

# PCA scores
# Print the PCA scores
pca_values

# Create a DataFrame to store the PCA scores
pca_data = pd.DataFrame(pca_values)
pca_data.columns = ['comp0', 'comp1', 'comp2']

# Concatenate the cluster labels with the PCA scores
final = pd.concat([df.clust, pca_data], axis=1)

# Visualize the DataFrame
# Create a scatter plot of the PCA scores for components 0 and 1
ax = final.plot(x='comp0', y='comp1', kind='scatter', figsize=(12, 8))

# Add cluster labels as text on the plot
final[['comp0', 'comp1', 'clust']].apply(lambda x: ax.text(*x), axis=1)
