# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:41:14 2020

@author: Rahul Kumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
wcss = []

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

hc = AgglomerativeClustering(n_clusters=5,
                             affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(x)
print(y_hc)

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, c='red',
            label='Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, c='blue',
            label='Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, c='green',
            label='Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, c='cyan',
            label='Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=100, c='magenta',
            label='Cluster 5')
plt.legend()
plt.title('Cluster of Customers')
plt.xlabel('Annual Income in k$')
plt.ylabel('Spending Scores [1-100]')
plt.show()