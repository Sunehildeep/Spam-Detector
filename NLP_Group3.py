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
from nltk.corpus import stopwords

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

#Removing comment_id, author and date columns
group3_shakira.drop(['COMMENT_ID','AUTHOR','DATE'],axis=1,inplace=True)

#Checking for null values
group3_shakira.isnull().sum()

#Improving the data by removing stop words
stop_words = stopwords.words('english')
group3_shakira = group3_shakira['CONTENT'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


'''Model Training''' #Man Kit Chan
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(group3_shakira)
print("\nDimensions of training data:", train_tc.shape)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)

# Shuffle the dataset
group3_shakira_shuffled = group3_shakira.sample(frac=1)

# Compute number of rows for training
# trow = round(len(group3_shakira_shuffled) * 0.75)

# df_train = group3_shakira_shuffled.iloc[:trow,:]
# df_test = group3_shakira_shuffled.iloc[trow:,:]

# x_train, y_train = df_train.iloc[:,:-1], df_train.iloc[:,-1]
# x_test, y_test = df_test.iloc[:,:-1], df_test.iloc[:,-1]

#classifier = MultinomialNB().fit(x_train, y_train)



'''Model Evaluation''' #Pak Wah Wong

'''Prediction''' #Huyen Anh
