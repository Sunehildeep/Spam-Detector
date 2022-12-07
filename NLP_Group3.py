# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:12:18 2022

@author: COMP 237 Group 3
"""
#Importing libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score

#Loading the CSV file
group3_shakira = pd.read_csv('D:\Study\Semester 3\AI - Python\Group 3\youtube05-Shakira.csv')

#Initial exploration of the dataframe
group3_shakira.head()

group3_shakira.tail()

group3_shakira.info()

#Removing comment_id, author and date columns
group3_shakira.drop(['COMMENT_ID','AUTHOR','DATE'],axis=1,inplace=True)

#Checking for null values
group3_shakira.isnull().sum()

#Improving the data by removing stop words
stop_words = stopwords.words('english')
group3_shakira['CONTENT'] = group3_shakira['CONTENT'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Use pandas.sample to shuffle the dataset, set frac =1 
group3_shakira_shuffled = group3_shakira.sample(frac=1)

#Using pandas split your dataset into 75% for training and 25% for testing
trow = round(len(group3_shakira_shuffled) * 0.75)
df_train = group3_shakira_shuffled.iloc[:trow,:]
df_test = group3_shakira_shuffled.iloc[trow:,:]
x_train, y_train = df_train.iloc[:,:-1], df_train.iloc[:,-1]
x_test, y_test = df_test.iloc[:,:-1], df_test.iloc[:,-1]

# Using nltk toolkit classes and methods prepare the data for model building
# Build a count vectorizer and extract term counts 
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(df_train['CONTENT'])
print("\nDimensions of training data:", train_tc.shape)
#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(train_tfidf, y_train)

# Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
###############################################
# Scoring functions

num_folds = 5
accuracy_values = cross_val_score(classifier, 
        train_tfidf, y_train, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier, 
        train_tfidf, y_train, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier, 
        train_tfidf, y_train, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier, 
        train_tfidf, y_train, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")

# Predict the output categories
predictions_train = classifier.predict(train_tfidf)
# compute accuracy of the classifier
accuracy_train = 100.0 * (y_train == predictions_train).sum() / y_train.shape[0]
print("Accuracy of the new classifier =", round(accuracy_train, 2), "%")

# Test the model on the test data
# Build a count vectorizer and extract term counts 
test_tc = count_vectorizer.transform(df_test['CONTENT'])
print("\nDimensions of testing data:", test_tc.shape)
# Transform vectorized data using tfidf transformer
test_tfidf = tfidf.transform(test_tc)
type(test_tfidf)

predictions_test = classifier.predict(test_tfidf)

# print the confusion matrix and 
confusion_matrix = confusion_matrix(y_test, predictions_test)
print (confusion_matrix)

# print the accuracy of the model
accuracy_test = 100.0 * (y_test == predictions_test).sum() / y_test.shape[0]
print("Accuracy with testing data: ", round(accuracy_test, 2), "%")

#Prediction
input_data = [
    'That song is AWESOME',
    'Best World Cup song ever!!',
    'Love the song',
    'even 12 years goes on, this song never dissapointed us, the atmosphere, vibes all make me goosebumps.',
    'THE GIRLS:18+ Youtube: It is fine Someone: Says "heck" Youtube: be gone',
    'CLICK https://123.com/ W3lc0m3 to our website!'
]
target_data = [0, 0, 0, 0, 1, 1]

# Convert data to a dataframe
input_df = pd.DataFrame({'CONTENT':input_data})

# Vectorize the input data
input_tc = count_vectorizer.transform(input_df['CONTENT'])
input_tfidf = tfidf.transform(input_tc)

# Predict the sentiment
predictions = classifier.predict(input_tfidf)

# Print the output
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', category)

#Print the accuracy of the model
accuracy = 100.0 * (target_data == predictions).sum() / len(target_data)
print("Accuracy with prediction data: ", round(accuracy, 2), "%")