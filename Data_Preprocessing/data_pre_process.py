# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:01:40 2020

@author: Rahul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('sample_data/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

# Taking care of missing data
"""
Missing data or NAN can be replaced by these values:

    If “mean”, then replace missing values using the mean along each column
    Can only be used with numeric data.

    If “median”, then replace missing values using the median along each column
    Can only be used with numeric data.

    If “most_frequent”, then replace missing using the most frequent value 
    along each column. Can be used with strings or numeric data.

    If “constant”, then replace missing values with fill_value. 
    Can be used with strings or numeric data.

"""
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)


# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=1)

# Feature Scaling
"""
Feature Scaling: It is a step of Data Pre Processing which is applied to
independent variables or features of data. It basically helps to normalise the
data within a particular range. Sometimes, it also helps in speeding up the
calculations in an algorithm.
"""
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
