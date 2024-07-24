# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:21:45 2024

@author: julio
"""

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import numpy as np

# Holdout
from sklearn.model_selection import train_test_split

#K-fold or leaveoneout
from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score
from sklearn import metrics

# Label encoding for classes and used in target variables
from sklearn.preprocessing import LabelEncoder
# Ordinal encoding for classes and used in features variables
from sklearn.preprocessing import OrdinalEncoder
# One hot encoding for classes creating new columns and setting 1 and 0 to the rest
# and used in features variables
from sklearn.preprocessing import OneHotEncoder
#mean enconding - enconding by mean of categories [auto,price -> mean_auto_price]
from sklearn.preprocessing import TargetEncoder


data = pd.read_excel('census.xlsx')

X = data.iloc[:,:-1]

y = data.iloc[:,-1:]

label_encoder = True
ordinal_encoder = True

    
    
if label_encoder:
    le = LabelEncoder()
    y = le.fit_transform(y)
    
if ordinal_encoder:
    cols=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    oe = OrdinalEncoder()
    X[cols] = oe.fit_transform(X[cols])
    
    
# Distance between centroids, initialize at random but the best is k-means++ that get the furthest point in the initialization. 
# Classify by calculating the distance of point and centroid, then refresh centroid with the points in each group until convergence -  
# minimizing a criterion known as the inertia  - sum min ||(x-mean)||^2
# uses the elbow curve to determine the K parameter
# Affect by outliers and needs normalization/ data prep (all of them need)
# hard clustering

# Gaussian mixture model is very similar to kmeans, but are soft-clustering technique that uses probability
# Needs initializaiont, usually made by AIC,BIC or silhouete model
# Affect by outliers and needs normalization/ data prep 
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

kmeans.cluster_centers_


# Grouping by density. Use circle with determined radius to choose points that are core (if x points are inside the circle then it is a core point).
# Point close to core points are associated to the class of the core points
# points not close to the core are not selected 
# Robust to outliers given the previous text
# Doesnt need to inform number os clusters
# high eps classify more points but might turn everything the same class (bias), less eps classify less points increase variance, but too low difficult the cluster creation
# Something similar with the min samples inside the circle
# hard clustering]
# needs normalization/ data prep 

clustering = DBSCAN(eps=3, min_samples=2).fit(X)

clustering.labels_


# Group by similarity, usually a distance
# has many type of linkage (similarity metric between cluster): single, complete, average, centroid/ward
# single group the closest points first
# complete group by maximum distance
# average group with the points between max and min dist
# ward group by centroids just like kmeans
# It does not need initialization
# Usually single linkage can make outlier  observations in a singleton branch of the dendogram
# Complete tends to group the outliers more
# hard clustering
# needs normalization/ data prep 

clustering2 = AgglomerativeClustering().fit(X)

clustering2.labels_