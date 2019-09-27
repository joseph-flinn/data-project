# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



# ========== Load Data ==========
data_raw = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

print("Length of PassengerIds: %d" % (len(pd.concat([data_raw['PassengerId'], data_test['PassengerId']]))))
print("Length of unique PassengerIds: %d" % (len(pd.concat([data_raw['PassengerId'], data_test['PassengerId']]).unique())))

new_data = pd.concat([data_raw, data_test])

data_raw = new_data[new_data['Age'].notnull()]
data_test = new_data[new_data['Age'].isnull()]

data_cleaner = [data_raw, data_test]


# ========== Data Cleaning ==========
"""
- Correcting
- Completing
- Creating
- Converting
"""

# Null values        
data_raw.info()
data_raw.describe()

print('data_raw:\n%s' % (data_raw.isnull().sum()))
print('-'*20)
print('Test:\n%s' % (data_test.isnull().sum()))


# Create dependent variable
from sklearn.preprocessing import LabelEncoder
age_bins = pd.cut(data_raw['Age'], 5)
data_raw['AgeBin'] = age_bins
data_raw['AgeBin_code'] = LabelEncoder().fit_transform(data_raw['AgeBin']) 


for dataset in data_cleaner:
    # Fill missing data
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

    # Converting Data
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'X')
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['FareBin_code'] = LabelEncoder().fit_transform(dataset['FareBin'])

    # Feature Engineering
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    
print('data_raw:\n%s' % (data_raw.isnull().sum()))
print('-'*20)
print('Test:\n%s' % (data_test.isnull().sum()))
    
training_columns = ['FareBin_code', 'Parch', 'Pclass','SibSp', 'FamilySize', 'IsAlone']
dependent_column = 'AgeBin_code'
data_dummy = pd.get_dummies(data_raw[training_columns], columns=['Pclass', 'FareBin_code'])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_dummy)
data_processed = scaler.transform(data_dummy)


# Splitting the training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_processed, data_raw[dependent_column])