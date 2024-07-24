# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:30:19 2024

@author: julio
"""

from sklearn import svm
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
    
    
# SVM 

# Low effect of outliers
# non parametric model
# training complexity O(n^2) and O(n^3)
# The traditional svm is affect by unbalanced data. Many SVM implementations address this by assigning different weights to positive and negative instances.
# Initially a classifier algorithm but used in regression


# Find the support vectos -> the ones that minimizes the margin (the plane with minimal distance between all points). 
# If has mixed classes, the optimization problem can add slacks variables
# Use kernel trick (It allows us to operate in the original feature space without computing the coordinates of the data in a higher dimensional space. Now, a basic concept in ML is the dot product. You often do dot products of the features of a data sample with some weights w, the parameters of your model. Instead of doing explicitly this projection of the data in 3D and then evaluating the dot product, you can find a kernel function that simplifies this job by simply doing the dot product in the projected space for you, without the need to actually compute projections and then the dot product. This allows you to find a complex non linear boundary that is able to separate the classes in the dataset. This is a very intuitive explaination. )

# the minimization equation -> l1-svm - ||w||^2 + C sum slack or l2-svm ||w||^2 + C/2 sum slack^2 

print('Decision Tree Regression \n')
if split_train_test:
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model =svm.SVC().fit(X_train,y_train,sample_weight=None)
    print('\n')
    print('Model score Training - holdout')
    print(model.score(X_train, y_train))
    print('\n')
    print('Model score Test - holdout')
    print(model.score(X_test, y_test))
    print('\n')

if cross_validation:
    # scoring = ['precision_macro', 'recall_macro']
    cross_validation_function(svm.SVR(), X, y, 5)