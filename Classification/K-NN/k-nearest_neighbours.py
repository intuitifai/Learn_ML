# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:56:33 2020

@author: Rahul Kumar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

neigh = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
neigh.fit(x_train, y_train)

y_pred = neigh.predict(x_test)

print(np.concatenate((
                        y_pred.reshape(len(y_pred), 1),
                        y_test.reshape(len(y_test), 1)
                      ), axis=1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
