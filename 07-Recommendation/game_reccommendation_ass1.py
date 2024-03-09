# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:36:31 2023

@author: Kajal
"""
'''
               ~~~Problem Statment~~~
               
This dataset is related to the video gaming industry and a survey was 
conducted to build a recommendation engine so that the store can
improve the sales of its gaming DVDs.Build a Recommendation Engine and 
suggest top selling DVDs to the store customers.


          ~~~Business Objectives~~~
Create the recommendation system to improve the sales of gaming DVDs.
   
     Maximize:   Sales the Maximum no of gaming DVDs.
     Minimize:  
     Constraint:
         
'''
'''############################DATA DICTIONARY#########################################

Name_Of_ feature      Type                     Relevance

UserId              Quantitative(Nominal)     Relevant, ID provide useful information
Gamvning              Qualitative(Categorical)  Relevant, Game provide useful information
Rating              Quantitative(Nominal)     Relevant, Rating provide useful information

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/11-recommendation_system assignment/game.csv")
df

'''########################EDA###################################'''
df.columns
#['userId', 'game', 'rating'], dtype='object'
df.shape
#(5000, 3)
df.size
#15000
df.head()
#return top 5 rows
df.tail()
#Returs bottom 5 rows
df.describe()
'''
  userId       rating
count  5000.000000  5000.000000
mean   3432.282200     3.592500
std    1992.000866     0.994933
min       1.000000     0.500000
25%    1742.500000     3.000000
50%    3395.000000     4.000000
75%    5057.750000     4.000000
max    7120.000000     5.000000

from describe mean of all features are greated than 0 and 1
standard deviation of rating is near to 1
'''
df.isnull()
#True-null value present
#false-Null values are not present

df.isnull().sum()
'''
userId    0
game      0
rating    0
dtype: int64

There is no null values from the given dataset
'''
'''#####################Checking outliers#######################'''
sns.boxplot(df['userId'])   
#There is no outliers from the given features
#All the data is normally distributed

sns.boxplot(df['game'])

sns.boxplot(df['rating'])
#Outliers are present from the given data set
IQR = df.rating.quantile(0.75)-df.rating.quantile(0.25)
upper_limit1 = df.rating.quantile(0.75)+1.5*IQR
upper_limit1
lower_limit = df.rating.quantile(0.25)-1.5*IQR

# to remove the outlioers
df2 = pd.DataFrame(np.where(df.rating > upper_limit1, upper_limit1, np.where(df.rating < lower_limit, lower_limit, df.rating)))
# df_trimmed=al.loc[~outliers_df]
sns.boxplot(x=df2[0])   #data will not distributed normally
sns.distplot(df['rating']) 
#Data will not be normally distributed 
#It will be the right skwed



'''############################pairplot##########################'''
sns.pairplot(df)


'''###########################Recommendation System####################'''


import pandas as pd
# Import pandas library

game=pd.read_csv("C:/11-recommendation_system assignment/game.csv")
# Read the CSV file into a DataFrame

game.shape
# Display the shape of the DataFrame (number of rows and columns)

game.columns
# Display the column names of the DataFrame

game.game
# Display the 'game' column of the DataFrame (This line seems to contain a typo, 'game.game' should be 'game['game']')

# Here we are considering only genre
from sklearn.feature_extraction.text import TfidfVectorizer
# Import TfidfVectorizer from scikit-learn

# This is term frequency-inverse document frequency
# Each row is treated as a document
tfidf=TfidfVectorizer(stop_words='english')
# Create a TfidfVectorizer object with English stop words

game['game'].isnull().sum()
# Count the number of null values in the 'game' column

# Now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(game.game)
# Fit and transform the 'game' column using TfidfVectorizer to create the TF-IDF matrix

tfidf_matrix.shape
# Display the shape of the TF-IDF matrix

# You will get a 5000x3068 matrix
# It has created a sparse matrix, meaning that we have 3068 games

from sklearn.metrics.pairwise import linear_kernel
# Import linear_kernel from scikit-learn

# This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
# Compute the cosine similarity matrix between all pairs of games using the TF-IDF matrix

# Each element of the TF-IDF matrix is compared with each element of the TF-IDF matrix only
# Output will be a similarity matrix of size 5000x5000
# Here in cosine_sim_matrix, there are no game names, only index is provided

# We will try to map game name with game index given
# For that purpose, a custom function is written

game_index=pd.Series(game.index,index=game['game']).drop_duplicates()
# Convert the game index into a Series format, where we want index and corresponding game names

game_id=game_index['Quake']
# Get the game ID for a specific game name (here, 'Quake')

game_id

def get_game_recommendations_with_ratings(name, topN):
    # Define a function to get game recommendations with ratings
    
    # Sort the game based on the rating of the game
    sorted_games = game.sort_values(by='rating', ascending=False)
    
    # Top N games with the highest ratings
    top_game_recommendations = sorted_games.head(topN+1)
    
    # It is taking the original index while assigning, so changing the index
    top_game_recommendations.reset_index(drop=True, inplace=True)
    
    print(top_game_recommendations)

# Enter your game and number of games to be recommended
get_game_recommendations_with_ratings('NASCAR Heat', topN=5)
