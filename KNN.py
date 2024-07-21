# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:56:18 2024

@author: julio
"""


from sklearn.neighbors import KNeighborsClassifier
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

split_train_test = True
cross_validation = True
label_encoder = True
ordinal_encoder = True

print('Naive_Bayes \n')
def cross_validation_function(model,X,y,cv):
    scores = cross_validate(model, X, y, cv=5) #, scoring=scoring
    print('Model mean score Test - cross validation')
    print(scores['test_score'].mean())
    print('\n')
    
    
if label_encoder:
    le = LabelEncoder()
    y = le.fit_transform(y)
    
if ordinal_encoder:
    cols=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    oe = OrdinalEncoder()
    X[cols] = oe.fit_transform(X[cols])
    
# instance algorithm
# BASED ON DISTANCE!!
# algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
# Non parametric
# outliers affect and benefits from normalization
# Curse of dimensionality affects as dimensions grow points became more distant and harder to classify
# Can be used to image classification: best use cases include for KD-tree 
# are Image processing,Computer graphics, Geographic Information Systems(GIS) etc.
# It doesn't handle categorical features, if they are not numbers

# weights{‘uniform’, ‘distance’},
# uniform’ : uniform weights. All points in each neighborhood are weighted equally.
# ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

# source https://medium.com/@geethasreemattaparthi/ball-tree-and-kd-tree-algorithms-a03cdc9f0af9
# Ball Tree:
# It is a tree-based data structure designed for organizing and querying points in 
# multidimensional spaces. It is also a binary tree with a hierarchical (binary) structure.
# The ball tree data structure is particularly effective when there are a lot of dimensions.
# Unlike KD-trees, Ball trees use hyperspheres to represent partitions,
# grouping nearby points within each hypersphere.

#KD Tree:
# A hierarchical data structure called a KD-Tree (K-Dimensional Tree) is used for 
# multidimensional space partitioning.
# It creates a binary tree with each node representing a part of the space
# by iteratively dividing the space along predefined dimensions.
# Often used for effective closest neighbor searches.


# Explanations
# https://scikit-learn.org/stable/modules/neighbors.html

# Differents settings 
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


if split_train_test:
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model =KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)
    print('\n')
    print('Model score Training - holdout')
    print(model.score(X_train, y_train))
    print('\n')
    print('Model score Test - holdout')
    print(model.score(X_test, y_test))
    print('\n')

if cross_validation:
    # scoring = ['precision_macro', 'recall_macro']
    cross_validation_function(KNeighborsClassifier(n_neighbors=3), X, y, 5)

