# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:16:51 2019

@author: Joseph
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")
comp_test_dataset = pd.read_csv("test.csv")

X = dataset.iloc[:, [2,4,5,6,7,9]].values # [Pclass(c), Sex(c), Age, SibSp, Parch, Fare]
y = dataset.iloc[:, 1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
missingvalues = missingvalues.fit(X[:, 2:3])
X[:, 2:3] = missingvalues.transform(X[:, 2:3])

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('Pclass', OneHotEncoder(), [0]),
                        ('Sex', OneHotEncoder(), [1]),], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Split data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Training the classifiers
classifiers = []
from sklearn.linear_model import LogisticRegression
classifiers.append({
    'name': 'logsitic_regression', 
    'classifier': LogisticRegression()
})
from sklearn.ensemble import RandomForestClassifier
classifiers.append({
    'name': 'random_forest',
    'classifier': RandomForestClassifier(n_estimators=50, criterion="entropy")
})
from sklearn.naive_bayes import GaussianNB
classifiers.append({
    'name': 'naive_bayes',
    'classifier': GaussianNB()        
})
# Throwing Deep Learning in here just for fun (need a learning curve?)
from sklearn.neural_network import MLPClassifier
classifiers.append({
    'name': 'nn',
    'classifier': MLPClassifier(max_iter=500) 
})


for classifier in classifiers:
    classifier['classifier'].fit(X_train, y_train)
    classifier['y_pred'] = classifier['classifier'].predict(X_test)


# Metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
for classifier in classifiers:
    print("\nClassification report for classifier %s:\n\n%s\n" % (
            classifier['classifier'], classification_report(y_test, classifier['y_pred'])))
    print("Area under curve (ROC): %s\n" % roc_auc_score(y_test, classifier['y_pred']))
    print("Confusion Matrix:\n%s)" % confusion_matrix(y_test, classifier['y_pred']))