# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:32:11 2020

@author: Rahul Kumar
"""
import pandas as pd
import matplotlib.pyplot as plt
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
ads_selected = []
number_rewards_1 = [0] * d
number_rewards_0 = [0] * d
total_rewards = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_rewards_1[i]+1,
                                         number_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_rewards_1[ad] += 1
    else:
        number_rewards_0[ad] += 1
    total_rewards += reward


# Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of Ads Selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
