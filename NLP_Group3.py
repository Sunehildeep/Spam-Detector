# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:12:18 2022

@author: COMP 237 Group 3
"""
#Importing libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns

'''Data Exploration''' #Rincy
#Load the data into pandas dataframe
plt.style.use('seaborn-dark')

#Loading the CSV file
group3_shakira = pd.read_csv('./Youtube05-Shakira.csv')

#Initial exploration of the dataframe
pd.set_option('display.max_columns', None)

group3_shakira.head()

group3_shakira.tail()

group3_shakira.info()

# Using a heatmaps to better visualize the integrity of the dataframe.
# The information provided by posts.info() is visualy confirmed. There are no missing data in this dataframe.
plt.figure(figsize = (16,8))
sns.heatmap(group3_shakira.isnull(), cmap = 'viridis', cbar = False)

'''Data Preprocessing''' #Sunehildeep
#Removing comment_id, author and date columns
group3_shakira.drop(['COMMENT_ID','AUTHOR','DATE'],axis=1,inplace=True)

#Checking for null values
group3_shakira.isnull().sum()


'''Model Training''' #Man Kit Chan

'''Model Evaluation''' #Pak Wah Wong

'''Prediction''' #Huyen Anh
