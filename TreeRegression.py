# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:07:14 2024

@author: julio
"""

from sklearn.tree import DecisionTreeRegressor
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
    
    
# Trees Non parametric model for regression or classification.
# tree can be seen as a piecewise constant approximation.
# Simple to understand and to interpret. Trees can be visualized.
# Requires little data preparation. Generally doesnt need normalization or dummy variables (Like transforming red,green, blue into 1,2,3)
# O(log(n))
# Without prunning, bagging, boosting can overfit easily
# Susceptible to class unbalancing (desbalanceamento de classe)
# Robust to outliers and missing values. But of course can somewhat affect the decision.
# criterion{“gini”, “entropy”, “log_loss”} for classification
# criterion {'absolute_error', 'squared_error', 'poisson', 'friedman_mse'} for regression

# More generally, ensemble models can be applied to any base learner beyond trees, in averaging methods such as Bagging methods, model stacking, or Voting, or in boosting, as AdaBoost.

# gradient boosting tree: basically update the y values by using a weak learner and a step parameter in which
# y = Fm(xi) --> Fm(x) = Fm-1(xi) + alfa*weaklearn(x), F1 = F0 +  alfa*weaklearn(x), Fo for mse is the average of y

# adaboosting -> attribute initially equal weight for every label, and use a weak learner to predict the labels. If the predictor gets the label correct, 
# the associated weight to that label decreases, otherwise the weight increase.

# aggregating bootstrap/bagging tree: Full trees classified using bootstrap over data to reduce variance of the model


# Random forrest: Create many trees using aggregating bootstrap/bagging tree with only a small but consistent number of unique features considered 
# and to predict use metric to get the average result for the n tree created.
# example: Averages the label/output of 20 trees for a certain features. 
  
    
# Tree algorithms: ID3, C4.5, C5.0 and CART

# https://scikit-learn.org/stable/modules/tree.html#tree
# Differents settings 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

print('Decision Tree Regression \n')
if split_train_test:
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model =DecisionTreeRegressor(criterion='squared_error').fit(X_train,y_train,sample_weight=None)
    print('\n')
    print('Tree feature importances')
    print(model.feature_importances_)
    print('\n')
    print('Model score Training - holdout')
    print(model.score(X_train, y_train))
    print('\n')
    print('Model score Test - holdout')
    print(model.score(X_test, y_test))
    print('\n')

if cross_validation:
    # scoring = ['precision_macro', 'recall_macro']
    cross_validation_function(DecisionTreeRegressor(), X, y, 5)