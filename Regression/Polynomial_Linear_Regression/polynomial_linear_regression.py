# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:17:31 2020

@author: Rahul Kumar
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

polynomial_regressor = PolynomialFeatures(degree=5)
x_polynomial = polynomial_regressor.fit_transform(x)

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_polynomial, y)

plt.scatter(x, y, color='red')
plt.plot(x, linear_regressor.predict(x), color='blue')
plt.title('Linear Regression on Salary VS Position Level')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

plt.scatter(x, y, color='red')
plt.plot(x, linear_regressor_2.predict(x_polynomial), color='blue')
plt.title('Polynomial Regression on Salary VS Position Level')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

print("Why Linear Model doesn't predict correctly?")
print("Answer: The plotted graph shows the salary to be way above correct one")
print("Which is = ", linear_regressor.predict([[6.5]]))
print("But the polynomial graph actually predicts the correct salary")
print("Which is = ", linear_regressor_2.predict(
    polynomial_regressor.fit_transform([[6.5]])))
