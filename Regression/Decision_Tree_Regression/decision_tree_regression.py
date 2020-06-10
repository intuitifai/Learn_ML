# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:56:30 2020

@author: Rahul Kumar
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Creating regression model
regressor = DecisionTreeRegressor(random_state=1)
# Training the model
regressor.fit(x, y)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('DECISION TREE REGRESSION')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()