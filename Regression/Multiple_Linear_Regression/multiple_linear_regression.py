# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:31:32 2020

@author: Rahul Kumar
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Separate the Train and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

# Train the model on Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the Test set Result
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((
                        y_pred.reshape(len(y_pred), 1),
                        y_test.reshape(len(y_test), 1)
                      ), axis=1))
