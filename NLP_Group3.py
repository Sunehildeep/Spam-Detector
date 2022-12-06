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
from sklearn.model_selection import cross_val_score

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

#Printing the new shape of the dataframe
print(group3_shakira.shape)

# Shuffle the dataset
group3_shakira_shuffled = group3_shakira.sample(frac=1)

# Compute number of rows for training
trow = round(len(group3_shakira_shuffled) * 0.75)

# Split the dataset into training and testing
training = group3_shakira_shuffled[:trow]
testing = group3_shakira_shuffled[trow:]

#Printing shape
training.shape
testing.shape

training_X = training['CONTENT']
training_Y = training['CLASS']

testing_X = testing['CONTENT']
testing_Y = testing['CLASS']

#Printing shape
training_X.shape
training_Y.shape
testing_X.shape
testing_Y.shape

cv = CountVectorizer(stop_words=stopwords.words('english'))
features_train = cv.fit_transform(training_X)

tfidf = TfidfTransformer()
features_train_tfidf = tfidf.fit_transform(features_train)
features_test = cv.transform(testing_X)
features_test_tfidf = tfidf.transform(features_test)

type(features_train_tfidf)

classifier = MultinomialNB(alpha=0.40, fit_prior=False, class_prior=None)
classifier.fit(features_train_tfidf, training_Y)

#Cross Validation with 5 folds
scores = cross_val_score(classifier, features_train_tfidf, training_Y, cv=5)
print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Testing the model
y_pred = classifier.predict(features_test_tfidf)

#Printing the accuracy score
print("Accuracy Score: ", accuracy_score(testing_Y, y_pred))

#Printing the confusion matrix
print("Confusion Matrix: ", confusion_matrix(testing_Y, y_pred))

input_data = [
    'That song is AWESOME',
    'Best World Cup song ever!!',
    'Love the song',
    'even 12 years goes on, this song never dissapointed us.',
    'THE GIRLS:18+ Youtube: It is fine Someone: Says "heck" Youtube: be gone',
    'CLICK https://123.com/ W3lc0m3 to our website!'
]

target_data = [0, 0, 0, 0, 1, 1]

input_data_features = cv.transform(input_data)
input_data_tfidf = tfidf.transform(input_data_features)

predictions = classifier.predict(input_data_tfidf)

for i in range(len(input_data)):
    print(input_data[i], '->', predictions[i])

#Printing the classification report
print("Classification Report: ", classification_report(target_data, predictions))

#Printing the accuracy score
print("Accuracy Score: ", accuracy_score(target_data, predictions))