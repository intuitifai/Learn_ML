# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:46:14 2020

@author: Rahul Kumar
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
nltk.download('stopwords')
corpus = []
skipwords = ['not', "don't", 'negative', 'no', 'ugly']
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    for elements in skipwords:
        if elements in all_stopwords:
            all_stopwords.remove(elements)
    review = [ps.stem(word) for word in review
              if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1566)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

merged_train_test_output = np.concatenate((
    y_pred.reshape(len(y_pred), 1),
    y_test.reshape(len(y_pred), 1)
    ), axis=1)

#print(merged_train_test_output)

#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#print(accuracy_score(y_test, y_pred))

# Ask for a review
new_review = input("Enter your thoughts here: ")
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
if 1 in new_y_pred:
    print("Thanks for your review! We will love to see you again")
else:
    print("Please let us know more. We will try to help you out as much possible")