# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:18:42 2019

@author: Joseph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ========== Import Data ==========
from data_preprocessing import scaler, X_train, X_test, y_train, y_test
dependent_column = 'AgeBin_code'


from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# ========== Tune Model with Hyper Parameters ==========
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate
from sklearn.metrics import classification_report
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6)

#from sklearn.preprocessing import label_binarize
#y = label_binarize(data_raw[dependent_column], classes=[0,1,2,3,4])

# base model
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 100, 500, 10))
nn.fit(X_train, y_train)
g = plot_learning_curve(nn,"NN learning curves", X_train, y_train, cv=cv_split)

y_pred = nn.predict(scaler.transform(X_test))
print(classification_report(y_test, y_pred))
print('-'*20)


# Tune Hyper parameters
param_grid = {
    'solver': ['lbfgs'], 
    'max_iter': [1000,1200,1400,1600,1800,2000], 
    'alpha': [0.001, 0.03, 0.01, 0.03, 0.1, 0.3, 1], 
    'hidden_layer_sizes': [(20, 100), (10, 100, 10), (10, 100, 250, 20), (20, 250, 500, 20)]
}

tune_model = GridSearchCV(
    MLPClassifier(), param_grid=param_grid, cv=cv_split, n_jobs=-1, verbose=1
)
tune_model.fit(X_train, y_train)
y_pred = tune_model.predict(scaler.transform(X_test))
print(classification_report(y_test, y_pred))
print('\nParams:', tune_model.get_params())
print('-'*20)

print('AFTER DT Parameters: ', tune_model.best_params_)
