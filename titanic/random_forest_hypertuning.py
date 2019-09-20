# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:43:13 2019

@author: Joseph
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pre_process(dataset, dependant=None):
    X = dataset[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].values
    #X = dataset.iloc[:, [2,4,5,6,7,9]].values # [Pclass(c), Sex(c), Age, SibSp, Parch, Fare]
    if dependant is not None:
        y = dataset.iloc[:, dependant].values
    else:
        y = None
    
    # Taking care of missing data
    from sklearn.impute import SimpleImputer
    missingvalues = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
    missingvalues = missingvalues.fit(X[:, 2:3])
    X[:, 2:3] = missingvalues.transform(X[:, 2:3])
    missingvalues = missingvalues.fit(X[:, 5:6])
    X[:, 5:6] = missingvalues.transform(X[:, 5:6])
    
    # Encoding categorical data
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    ct = ColumnTransformer([('Pclass', OneHotEncoder(), [0]),
                            ('Sex', OneHotEncoder(), [1]),], remainder='passthrough')
    X = np.array(ct.fit_transform(X), dtype=np.float)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    return (X, y)


dataset = pd.read_csv("train.csv")
comp_test_dataset = pd.read_csv("test.csv")

X, y = pre_process(dataset, dependant=1)
# Split data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the classifiers
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50, criterion="entropy")

"""
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# Metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
print("\nClassification report for classifier %s:\n\n%s\n" % (
        classifier, classification_report(y_test, y_pred)))
print("Area under curve (ROC): %s\n" % roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n%s)" % confusion_matrix(y_test, y_pred))
"""

# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

rf_param_grid = {
    "max_depth": [None],
    "max_features": [1, 3, 6],
    "min_samples_split": [2, 3, 10],
    "min_samples_leaf": [1, 3, 10],
    "bootstrap": [False],
    "n_estimators": [50, 100, 300, 500],
    "criterion": ["gini", "entropy"]
}

from sklearn.model_selection import GridSearchCV
gsRFC = GridSearchCV(classifier, param_grid=rf_param_grid, cv=kfold, 
                      scoring="accuracy", n_jobs=-1, verbose=1)

gsRFC.fit(X_train, y_train)

RFC_best = gsRFC.best_estimator_

gsRFC.best_score_


# Generate competition submission
X_comp, y_none = pre_process(comp_test_dataset)

comp_test_pred = gsRFC.predict(X_comp)

comp_output = pd.DataFrame(
    data=np.append(comp_test_dataset[["PassengerId"]].values, comp_test_pred.reshape((418, 1)), axis=1),
    columns=["PassengerId", "Survived"]
)

comp_output.to_csv("submission.csv", index=False)