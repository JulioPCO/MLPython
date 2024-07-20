# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:05:40 2024

@author: julio
"""

from sklearn.linear_model import LinearRegression
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
    
# Model Regression y=alfa*X + beta
# Assumptions Gaussian, homocedastic
# Affected by outliers and Benefits from normalization
# Parametric model -A parametric model is one that can be parametrized by a finite number of parameters. 
# Optimization by minimizing the MSE (y-f(x))^2
#O(n) - não tenho certeza
#.score = R² -> 

print('Linear regression \n')
if split_train_test:
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model =LinearRegression().fit(X_train,y_train,sample_weight=None)
    print('\n')
    print('Linear intercept')
    print(model.intercept_)
    print('\n')
    print('Linear coefficients')
    print(model.coef_)
    print('\n')
    print('Model score Training - holdout')
    print(model.score(X_train, y_train))
    print('\n')
    print('Model score Test - holdout')
    print(model.score(X_test, y_test))
    print('\n')
    
if cross_validation:
    # scoring = ['precision_macro', 'recall_macro']
    cross_validation_function(LinearRegression(), X, y, 5)