# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:06:59 2020

@author: Rahul
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

# Training the simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(x_test)

# Visualizing the Training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('SALARY VS EXPERIENCE GRAPH - TRAINING SET')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('SALARY VS EXPERIENCE GRAPH - TEST SET')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
