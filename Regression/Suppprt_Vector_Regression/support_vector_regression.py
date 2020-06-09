# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:48:54 2020

@author: Rahul Kumar
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training SVR model on the Training Data set
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

# Predict the new result
print(sc_y.inverse_transform(regressor.predict(sc_x.fit_transform([[6.5]]))))

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(
                                        regressor.predict(x)), color='blue')
plt.title('Support Vector Regression on Salary VS Position Level')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()