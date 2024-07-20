# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:26:53 2024

@author: julio
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
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
    
    
# “naive” assumption of conditional independence between every pair of features 
# given the value of the class variable. No covariance/correlation between variables
# The QDA with no correlation equals the gaussian naive bayes
# For gaussian naive bayes the mean and variance are estimated by maximum likelihood
# Can be executed in batch
# Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods
# Problem is that make assumption of distribution 

# naive Bayes can be either parametric or nonparametric, although in practice the former is more common

# Outliers affect naive bayes 
#explanation: outliers will affect the shape of the Gaussian distribution and have the usual


# Normalization shouldn't be necessary since the features are only compared to each other.
# p(class|thing) = p(class)p(thing|class) =
# = p(class)p(feature_1|class)p(feature_2|class)...p(feature_N|class)
# So when fitting the parameters for the distribution feature_i|class it will just rescale the parameters (for the new "scale") in this case (mu, sigma2), but the probabilities will remain the same.


# Bernoulli Naive Bayes used when features are binary
# Under the hood, BernoulliNB binarizes the features based on a numeric threshold

# Explanations
# https://scikit-learn.org/stable/modules/naive_bayes.html

# Differents settings 
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB


if split_train_test:
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model =GaussianNB().fit(X_train,y_train)
    print('\n')
    print('Model score Training - holdout')
    print(model.score(X_train, y_train))
    print('\n')
    print('Model score Test - holdout')
    print(model.score(X_test, y_test))
    print('\n')

if cross_validation:
    # scoring = ['precision_macro', 'recall_macro']
    cross_validation_function(BernoulliNB(), X, y, 5)

