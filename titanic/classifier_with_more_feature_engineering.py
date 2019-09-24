# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:30:00 2019

@author: Joseph

Note: Working through Yassine Ghouzam's tutorial to get a better understanding
of feature engineering https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

# =================================================
# ============== Load and Check data ==============
# =================================================


# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
IDtest = test["PassengerId"]


# Outlier detection
from collections import Counter
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices 
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25) # 1st quartile (25%)
        Q3 = np.percentile(df[col], 75) # 3rd quartile (75%)
        IQR = Q3 - Q1 # Interquartile range (IQR)
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
    return multiple_outliers


outliers_to_drop = detect_outliers(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
#train.loc[outliers_to_drop]
train = train.drop(outliers_to_drop, axis=0).reset_index(drop=True)


# Check for null and missing values
data_cleaner = [train, test]

for dataset in data_cleaner:
    dataset.fillna(np.nan)    


# =================================================
# =============== Feature Analysis ================
# =================================================
    
# Correlation matrix between numerical values (SibSp Parch Age and Fare values)
# and Survived
g = sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), 
                annot=True, fmt='.2f', cmap='coolwarm')

# Explore SibSp feature vs Survived
g = sns.factorplot(x='SibSp', y='Survived', data=train, kind='bar', height=6, 
                   palette='muted')
g.despine(left=True)
g = g.set_ylabels('survival probability')

# Explore Parch feature vs Survived
g = sns.factorplot(x='Parch', y='Survived', data=train, kind='bar', height=6,
                   palette='muted')
g.despine(left=True)
g = g.set_ylabels('survival probability')

# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, 'Age')

# Explore Age distribution
g = sns.kdeplot(train['Age'][(train['Survived'] == 0) & (train['Age'].notnull())], color='Red', shade=True)
g = sns.kdeplot(train['Age'][(train['Survived'] == 1) & (train['Age'].notnull())], color='Blue', shade=True)
g.set_xlabel('Age')
g.set_ylabel('Frequency')
g = g.legend(['Not Survived', 'Survived'])

# fill missing Fare data
for dataset in data_cleaner:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    
# Explore Fare distribution
g = sns.distplot(train['Fare'], color='m', label='Skewness: %.2f' % (train['Fare'].skew()))
g = g.legend(loc='best')

# Apply log to Fare to reduce skewness distribution
for dataset in data_cleaner:
    dataset['Fare'] = dataset['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
    
g = sns.distplot(train['Fare'], color='b', label='Skewness: %.2f' % (train['Fare'].skew()))
g = g.legend(loc='best')


# Categorical Values
# Sex
g = sns.barplot(x='Sex', y='Survived', data=train)
g = g.set_ylabels('survival probability')

train[['Sex', 'Survived']].groupby('Sex').mean()

# Pclass
g = sns.catplot(x='Pclass', y='Survived', data=train, kind='bar', height=6,
                   palette='muted')
g.despine(left=True)
g = g.set_ylabels('survival probability')

# Explore Pclass vs Survived by Sex
g = sns.catplot(x='Pclass', y='Survived', hue='Sex', data=train, height=6,
                kind='bar', palette='muted')
g.despine(left=True)
g = g.set_ylabels('survival probability')

# fill Embarked na with mode
for dataset in data_cleaner:
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    
g = sns.catplot(x='Embarked', y='Survived', data=train, height=6, kind='bar',
                palette='muted')
g.despine(left=True)
g = g.set_ylabels('survival probability')

# Explore Pclass vs Embarked (hypothesis: more 1st class passengers are coming from Cherbourg)
g = sns.catplot(x='Pclass', col='Embarked', data=train, height=6, kind='count',
                palette='muted')
g.despine(left=True)
g = g.set_ylabels('count')

# =================================================
# =============== Fill missing values =============
# =================================================

# Explore Age vs Sex, Parch, Pclass and SibSp
g = sns.catplot(y='Age', x='Sex', data=train, kind='box')
g = sns.catplot(y='Age', x='Sex', hue='Pclass', data=train, kind='box')
g = sns.catplot(y='Age', x='Parch', data=train, kind='box')
g = sns.catplot(y='Age', x='SibSp', data=train, kind='box')

for dataset in data_cleaner:
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
    
# Age is negatively correlated with SibSp, Parch, and Pclass
g = sns.heatmap(train[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']].corr(), cmap='BrBG', annot=True)

# Filling missing value of Age
# Note: I wanted to keep the datasets separate, but seemed the the prediction
#       of Age needed to be between both datasets. Actually would probably be
#       better if we made a regression model to predict the age.
dataset = pd.concat(objs=[train, test], axis=0, sort=True).reset_index(drop=True)
index_NaN_age = list(dataset['Age'][dataset['Age'].isnull()].index)

for i in index_NaN_age:
    age_med = dataset['Age'].median()
    age_pred = dataset['Age'][((dataset['SibSp'] == dataset.loc[i]['SibSp']) & (dataset['Parch'] == dataset.iloc[i]['Parch']) & (dataset['Pclass'] == dataset.iloc[i]['Pclass']))].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med
        
g = sns.catplot(x='Survived', y='Age', data=train, kind='box')
g = sns.catplot(x='Survived', y='Age', data=train, kind='violin')

# =================================================
# =============== Feature Engineering =============
# =================================================

# Name/Title
dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]
dataset['Title'] = pd.Series(dataset_title)

g = sns.countplot(x='Title', data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)

# Convert to categoritcal values Title
dataset['Title'] = dataset['Title'].replace([
    'Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
], 'Rare')
dataset['Title'] = dataset['Title'].map({
    'Master': 0, 'Miss': 1, 'Ms': 1, 'Mme': 1, 
    'Mlle': 1, 'Mrs': 1, 'Mr': 2, 'Rare': 3
})
dataset['Title'] = dataset['Title'].astype(int)

g = sns.countplot(dataset['Title'])
g = g.set_xticklabels(['Master', 'Miss/Ms/Mme/Mlle/Mrs', 'Mr', 'Rare'])


g = sns.catplot(x='Title', y='Survived', data=dataset, kind='bar')
g = g.set_xticklabels(['Master', 'Miss-Mrs', 'Mr', 'Rare'])
g = g.set_ylabels('survival probability')

dataset.drop(labels=["Name"], axis=1, inplace=True)


# Family Size
dataset['Fsize'] = dataset['SibSp'] + dataset['Parch'] + 1

g = sns.catplot(x='Fsize', y='Survived', data=dataset, kind='point') # some weird color ValueError. Seems to be a matplotlib bug
g = g.set_ylabels('survival probability')

# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 3 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF']= dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

g = sns.catplot(x='Single', y='Survived', data=dataset, kind='bar')
g = g.set_ylabels('survival probability')
g = sns.catplot(x='SmallF', y='Survived', data=dataset, kind='bar')
g = g.set_ylabels('survival probability')
g = sns.catplot(x='MedF', y='Survived', data=dataset, kind='bar')
g = g.set_ylabels('survival probability')
g = sns.catplot(x='LargeF', y='Survived', data=dataset, kind='bar')
g = g.set_ylabels('survival probability')

# convert to indecator values Title and Embarked
dataset = pd.get_dummies(dataset, columns=['Title'])
dataset = pd.get_dummies(dataset, columns=['Embarked'], prefix='Em')


# Cabin
dataset['Cabin'][dataset['Cabin'].notnull()].head()

# Replace the Cabin number by the type of cabin 'X' if not
dataset['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])

g = sns.countplot(dataset['Cabin'], order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'])
g = sns.catplot(y='Survived', x='Cabin', data=dataset, kind='bar', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'])
g = g.set_ylabels('survival probability')


dataset = pd.get_dummies(dataset, columns=['Cabin'], prefix='Cabin')


# Ticket

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append('X')
        
dataset['Ticket'] = Ticket

#g = sns.countplot(dataset['Ticket'])

dataset = pd.get_dummies(dataset, columns=['Ticket'], prefix='T')

# create categorical values for Pclass
dataset['Pclass'] = dataset['Pclass'].astype('category')
dataset = pd.get_dummies(dataset, columns=['Pclass'], prefix='Pc')

# Drop useless variables
dataset.drop(labels=["PassengerId"], axis=1, inplace=True)


# =================================================
# ====================== Modeling =================
# =================================================

# Separate train dataset and test dataset
train = dataset[:len(train)]
test = dataset[len(train):]
test.drop(labels=['Survived'], axis=1, inplace=True)

# Separate train features and label
train['Survived'] = train['Survived'].astype(int)
y_train = train['Survived']
X_train = train.drop(labels=['Survived'], axis=1)


# Cross validate model with Kfold stratified cross val
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

# Modeling step Test different algorithms
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

random_state = 2
classifiers = [
    SVC(random_state=random_state),
    DecisionTreeClassifier(random_state=random_state),
    AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), 
        random_state=random_state, learning_rate=0.1),
    RandomForestClassifier(random_state=random_state),
    ExtraTreesClassifier(random_state=random_state),
    GradientBoostingClassifier(random_state=random_state),
    MLPClassifier(random_state=random_state),
    KNeighborsClassifier(),
    LogisticRegression(random_state=random_state),
    LinearDiscriminantAnalysis()
]

from sklearn.model_selection import cross_val_score
cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y=y_train, 
                                      scoring='accuracy', cv=kfold, n_jobs=4))
    
cv_means = []
cv_std= []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({'CrossValMeans': cv_means, 'CrossValerrors': cv_std,
                       'Algorithm': ['SVC', 'DecisionTree', 'AdaBoost', 
                                     'RandomForest', 'ExtraTrees', 
                                     'GradientBoosting', 
                                     'MultipleLayerPerceptron', 'KNeighbors', 
                                     'LogisticRegression', 
                                     'LinearDiscriminantAnalysis']})
    
g = sns.barplot('CrossValMeans', 'Algorithm', data=cv_res, palette='Set3', 
                orient='h', **{'xerr': cv_std})
g.set_xlabel('Mean Accuracy')
g = g.set_title('Cross Validation scores')


### Hyperparameter tuning for best models

# Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {
    'base_estimator__criterion': ['gini', 'entropy'],
    'base_estimator__splitter': ['best', 'random'],
    'algorithm': ['SAMME', 'SAMME.R'],
    'n_estimators': [1, 2],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]
}

from sklearn.model_selection import GridSearchCV
gsAdaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, 
                        scoring='accuracy', n_jobs=4, verbose=1)
gsAdaDTC.fit(X_train, y_train)
Ada_best = gsAdaDTC.best_estimator_

# ExtraTrees
ExtC = ExtraTreesClassifier()

ex_param_grid = {
    'max_depth': [None],
    'max_features': [1, 3, 10],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [False],
    'n_estimators': [100, 300],
    'criterion': ['gini', 'entropy']
}

gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, 
                      scoring='accuracy', n_jobs=4, verbose=1)
gsExtC.fit(X_train, y_train)
ExtC_best = gsExtC.best_estimator_

# RandomForest
RFC = RandomForestClassifier()

rf_param_grid = {
    'max_depth': [None],
    'max_features': [1, 3, 10],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [False],
    'n_estimators': [100, 300],
    'criterion': ['gini', 'entropy']
}

gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, 
                     scoring='accuracy', n_jobs=4, verbose=1)
gsRFC.fit(X_train, y_train)
RFC_best = gsRFC.best_estimator_

# Gradient boosting
GBC = GradientBoostingClassifier()
gb_param_grid = {
    'loss': ['deviance'],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [4, 8],
    'min_samples_leaf': [100, 150],
    'max_features': [0.3, 0.1]
}

gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, 
                     scoring='accuracy', n_jobs=4, verbose=1)
gsGBC.fit(X_train, y_train)
GBC_best = gsGBC.best_estimator_

# SVC
SVMC = SVC(probability=True)
svc_param_grid = {
    'kernel': ['rbf'],
    'gamma': [0.001, 0.01, 0.1, 1],
    'C': [1, 10, 50, 100, 200, 300, 1000]
}

gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, 
                      scoring='accuracy', n_jobs=4, verbose=1)
gsSVMC.fit(X_train, y_train)
SVMC_best = gsSVMC.best_estimator_


# Plotting learning curves
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, 
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
             label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validataion score')
    plt.legend(loc='best')
    return plt

g = plot_learning_curve(gsRFC.best_estimator_, 'RFC learning curves', X_train,
                        y_train, cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_, 'ExtraTrees learning curves',
                        X_train, y_train, cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_, 'SVC learning curves',
                        X_train, y_train, cv=kfold)
g = plot_learning_curve(gsAdaDTC.best_estimator_, 'AdaBoost learning curves',
                        X_train, y_train, cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_, 'GradientBoosting learning curves',
                        X_train, y_train, cv=kfold)


# Feature importance of tree based classifiers
nrows = ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', figsize=(15,15))

names_classifiers = [
    ('AdaBoosting', Ada_best),
    ('ExtraTrees', ExtC_best),
    ('RandomForest', RFC_best),
    ('GradientBoosting', GBC_best)
]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40], 
                        x=classifier.feature_importances_[indices][:40], 
                        orient='h', ax=axes[row][col])
        g.set_xlabel('Relative importance', fontsize=12)
        g.set_ylabel('Features', fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + ' feature importance')
        nclassifier += 1
        
        
test_Survived_RFC = pd.Series(RFC_best.predict(test), name='RFC')
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name='ExtC')
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name='SVC')
test_Survived_AdaC = pd.Series(Ada_best.predict(test), name='Ada')
test_Survived_GBC = pd.Series(GBC_best.predict(test), name='GBC')

ensemble_results = pd.concat([
    test_Survived_RFC, test_Survived_ExtC, test_Survived_AdaC, 
    test_Survived_GBC, test_Survived_SVMC
], axis=1)

g = sns.heatmap(ensemble_results.corr(), annot=True)


# Ensemble Modeling
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
                                       ('svc', SVMC_best), ('adac', Ada_best),
                                       ('gbc', GBC_best)], voting='soft',
                           n_jobs=4)
votingC = votingC.fit(X_train, y_train)

# Prediction
test_Survived = pd.Series(votingC.predict(test), name='Survived')
results = pd.concat([IDtest, test_Survived], axis=1)
results.to_csv("submissions/ensemble_voting_submission.csv")