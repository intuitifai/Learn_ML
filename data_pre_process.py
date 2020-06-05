# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:01:40 2020

@author: rahul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('sample_data/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)
