# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:45:21 2020

@author: Rahul Kumar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


classifier = svm.SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

# Train the model
y_pred = classifier.predict(x_test)

merged_train_test_output = np.concatenate((
    y_pred.reshape(len(y_pred), 1),
    y_test.reshape(len(y_pred), 1)
    ), axis=1)

print(merged_train_test_output)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
