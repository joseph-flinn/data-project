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
from data_preprocessing import data_raw, data_dummy, X_train, X_test, y_train, y_test
dependent_column = 'Age'

# ========== Tune Model with Hyper Parameters ==========
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6)

#from sklearn.preprocessing import label_binarize
#y = label_binarize(data_raw[dependent_column], classes=[0,1,2,3,4])

# base model
dtree = ensemble.RandomForestRegressor(random_state=0)
base_results = cross_validate(
        dtree, data_dummy, data_raw[dependent_column], cv=cv_split,
        return_train_score=True
)
dtree.fit(data_dummy, data_raw[dependent_column])

print('BEFORE DT Parameters: ', dtree.get_params())
print('BEFORE DT Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('BEFORE DT Test w/bin score mean {:.2f}'.format(base_results['test_score'].mean()*100))
print('BEFORE DT Test w/bin score 3*STD: +/- {:.2f}'.format(base_results['train_score'].mean()*100*3))
print('-'*20)

# Tune Hyper parameters
param_grid = {
    'n_estimators': [10, 50, 100, 300],
    'max_depth': [2, 4, 6, 8, 10, None],
    'oob_score': [True],
}

tune_model = GridSearchCV(
    ensemble.RandomForestRegressor(), param_grid=param_grid, 
    scoring='r2', cv=cv_split, return_train_score=True
)

tune_model.fit(data_dummy, data_raw[dependent_column])

print('AFTER DT Parameters: ', tune_model.get_params())
print('AFTER DT Training w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('AFTER DT Test w/bin score mean {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('AFTER DT Test w/bin score 3*STD: +/- {:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100))
print('-'*20)


# ========== Tune Model with Feature Selection ==========
from sklearn import feature_selection

# base model
print('BEFORE DT RFE Training Shape Old: ', data_dummy.shape)
print('BEFORE DT RFE Training Columns Old: ', data_dummy.columns.values)
print('BEFORE DT RFE Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('BEFORE DT RFE Test w/bin score mean {:.2f}'.format(base_results['test_score'].mean()*100))
print('BEFORE DT RFE Test w/bin score 3*STD: +/- {:.2f}'.format(base_results['train_score'].mean()*100*3))
print('-'*20)

dtree_rfe = feature_selection.RFECV(
    dtree, step=1, scoring='accuracy', cv=cv_split
)
dtree_rfe.fit(data_dummy, data_raw[dependent_column])


indices = np.argsort(dtree.feature_importances_)[::-1][:40]
g = sns.barplot(y=X_train.columns[indices][:40], 
                x=dtree.feature_importances_[indices][:40], 
                orient='h')
g.set_xlabel('Relative importance', fontsize=12)
g.set_ylabel('Features', fontsize=12)
g.tick_params(labelsize=9)
g.set_title('DT feature importance')


X_rfe = data_dummy.columns.values[dtree_rfe.get_support()]
rfe_results = cross_validate(
        dtree, data_dummy[X_rfe], data_raw[dependent_column], cv=cv_split, 
        return_train_score=True
)

print('AFTER DT RFE Training Shape New: ', data_dummy[X_rfe].shape)
print('AFTER DT RFE Training Columns New: ', X_rfe)
print('AFTER DT RFE Training w/bin score mean: {:.2f}'.format(rfe_results['train_score'].mean()*100))
print('AFTER DT RFE Test w/bin score mean: {:.2f}'.format(rfe_results['test_score'].mean()*100))
print('AFTER DT RFE Test w/bin score 3*S: {:.2f}'.format(rfe_results['test_score'].mean()*100*3))
print('-'*20)

rfe_tune_model = GridSearchCV(
    ensemble.RandomForestRegressor(), param_grid=param_grid, scoring='roc_auc',
    cv=cv_split, return_train_score=True
)
rfe_tune_model.fit(data_dummy[X_rfe],  y)

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.get_params())
print('AFTER DT RFE Tuned Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('AFTER DT RFE Tuned Test w/bin score mean {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('AFTER DT RFE Tuned Test w/bin score 3*STD: +/- {:.2f}'.format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100))
print('-'*20)



# Visually Exploring
g = sns.barplot(x='Pclass', y='Age', data=data_raw)

g = sns.boxplot(x='Fare', hue='Age', data=data_raw)

g = sns.distplot(data_raw['Age'])
g = sns.distplot(data_raw['SibSp'])
g = sns.distplot(data_raw['Parch'])
g = sns.distplot(data_raw['Fare'])

g = sns.distplot(data_raw['AgeBin'])