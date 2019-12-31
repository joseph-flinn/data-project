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
from data_preprocessing import data_raw, dependent_column, data_dummy, X_train, X_test, y_train, y_test


# ========== Tune Model with Hyper Parameters ==========
from sklearn import tree
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6)

from sklearn.preprocessing import label_binarize
y = label_binarize(data_raw[dependent_column], classes=[0,1,2,3,4])

# base model
dtree = tree.DecisionTreeClassifier(random_state=0)
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
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10, None],
    'random_state': [0]
}

tune_model = GridSearchCV(
    tree.DecisionTreeClassifier(), param_grid=param_grid, 
    scoring='roc_auc', cv=cv_split, return_train_score=True
)

tune_model.fit(data_dummy, y)

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
    tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc',
    cv=cv_split, return_train_score=True
)
rfe_tune_model.fit(data_dummy[X_rfe],  y)

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.get_params())
print('AFTER DT RFE Tuned Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('AFTER DT RFE Tuned Test w/bin score mean {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('AFTER DT RFE Tuned Test w/bin score 3*STD: +/- {:.2f}'.format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100))
print('-'*20)





# ========= Picking best algorithm =========
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn import svm

classifiers = [
    # Ensemble Methods
    #('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    #('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    
    # Gaussian Process
    ('gpc', gaussian_process.GaussianProcessClassifier())
    
    # GLM
    #('lr', linear_model.LogisticRegressionCV()),
    
    # Naive Bayes
    #('bnb', naive_bayes.BernoulliNB()),
    #('gnb', naive_bayes.GaussianNB()),
    
    # Nearest Neighbor
    #('knn', neighbors.KNeighborsClassifier()),
    
    # SVM
    #('svc', svm.SVC(probability=True))
]


classifier_res = []
for classifier in classifiers:
    classifier_res.append(classifier[1].__class__.__name__)
    cv_results = cross_validate(classifier[1], data_dummy, 
                                data_raw[dependent_column], cv=cv_split,
                                return_train_score=True)
    classifier[1].fit(data_dummy, data_raw[dependent_column])
    classifier_res.append('Training score mean: {:.2f}'.format(cv_results['train_score'].mean()*100))
    classifier_res.append('Test score mean {:.2f}'.format(cv_results['test_score'].mean()*100))
    classifier_res.append('Test score 3*STD: +/- {:.2f}'.format(cv_results['test_score'].mean()*100*3))
    classifier_res.append('-'*20)
    
for line in classifier_res:
    print(line)


# Hyperparameter Tune with GridSearchCV
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [0.1, 0.25, 0.5, 0.75, 1.0]
grid_learn = [0.01, 0.03, 0.05, 0.1, 0.25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, 0.03, 0.05, 0.10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed= [0]

grid_params = [
    [{  # BaggingClassifier
        'estimator__n_estimators': grid_n_estimator,
        'estimator__max_samples': grid_ratio,
        'estimator__random_state': grid_seed
    }],
    [{  # ExtraTreesClassifier 
        'estimator__n_estimators': grid_n_estimator,
        'estimator__criterion': grid_criterion,
        'estimator__max_depth': grid_max_depth,
        'estimator__random_state': grid_seed
    }],
    [{  # RandomForestClassifier
        'estimator__n_estimators': grid_n_estimator,
        'estimator__criterion': grid_criterion,
        'estimator__max_depth': grid_max_depth,
        'estimator__oob_score': [True],
        'estimator__random_state': grid_seed
    }],
    [{  # GaussianProcessClassifier
        'estimator__max_iter_predict': grid_n_estimator,
        'estimator__random_state': grid_seed
    }]
]
    
import time
from sklearn.multiclass import OneVsRestClassifier


start_total = time.perf_counter()
for classifier, param in zip(classifiers, grid_params):
    start = time.perf_counter()
    print(classifier[1].__class__.__name__)
    best_search = GridSearchCV(estimator=OneVsRestClassifier(classifier[1]),
                               param_grid=param, cv=cv_split, 
                               scoring='roc_auc')
    best_search.fit(data_dummy, y)
    run = time.perf_counter() - start
    
    best_param = best_search.best_params_
    print("{}: {:.2f}s\n{}".format(classifier[1].__class__.__name__, run, best_param))
    temp_params = {}
    for k, v in best_param.items():
        temp_params[k.split('estimator__')[1]] = v
    classifier[1].set_params(**temp_params)
    
run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes'.format(run_total/60))



from sklearn.metrics import classification_report

grid_hard = ensemble.VotingClassifier(estimators=classifiers, voting='hard')
grid_hard.fit(X_train, y_train)
y_pred = grid_hard.predict(X_test)
print(classification_report(y_test, y_pred))



# Visually Exploring
g = sns.barplot(x='Pclass', y='Age', data=data_raw)

g = sns.boxplot(x='Fare', hue='Age', data=data_raw)

g = sns.distplot(data_raw['Age'])
g = sns.distplot(data_raw['SibSp'])
g = sns.distplot(data_raw['Parch'])
g = sns.distplot(data_raw['Fare'])

g = sns.distplot(data_raw['AgeBin'])