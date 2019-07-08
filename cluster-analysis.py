#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:19:16 2019

@author: ibm
"""

import pandas as pd
import numpy as np
import math
#Demo: Calculation of distance
Credit_Score_Expenses = pd.read_csv("./data/Credit_Score_Expenses.csv")
Credit_Score_Expenses.columns.values
Credit_Score_Expenses.describe()
Credit_Score_Expenses.shape

# Euclidean Distance Caculator
def distance_matrix(data_frame):
    import numpy as np
    result_distance=np.zeros((data_frame.shape[0],data_frame.shape[0]))
    for i in range(0 , data_frame.shape[0]):
        for j in range(0 , data_frame.shape[0]):
            result_distance[i,j]=round(math.sqrt(sum((data_frame.iloc[i] - data_frame.iloc[j])**2)),0)
    print(result_distance)

distance_matrix(Credit_Score_Expenses)

#Cluster Analysis
sup_market = pd.read_csv("./data/Super_market_Coupons.csv")
print(sup_market.shape)
print(sup_market.columns.values)
print(sup_market.head())

#Building Clusters here
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,  random_state=333) # Mention the Number of clusters
X=sup_market.drop(["cust_id"],axis=1) # Custid is not needed
kmeans = kmeans.fit(X) #Model building
print (kmeans)

#The Results
centers= kmeans.cluster_centers_
#Format and print  
np.set_printoptions(suppress=True)
print(np.around(centers))

# Getting the cluster labels and attaching them to the original data
labels = kmeans.predict(X)
print(labels)
sup_market["Cluster_id"]=labels
sup_market.head()

#Final Results
print(sup_market.groupby(['Cluster_id']).mean())
print(sup_market.groupby(['Cluster_id']).count())
#The Final Target population
target_data=sup_market[(sup_market["Cluster_id"]==1) | (sup_market["Cluster_id"]==3)]
print(target_data.shape)
target_data.sample(100)