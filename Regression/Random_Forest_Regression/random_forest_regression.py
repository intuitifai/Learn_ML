# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:26:42 2020

@author: Rahul Kumar
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('RANDOM FOREST REGRESSION')
plt.xlabel('Position Values')
plt.ylabel('Salaries')
plt.show()