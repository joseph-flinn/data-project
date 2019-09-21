# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 09:48:27 2019

@author: Joseph

Note: After conducting an initial naive approach to predicting the 
survivability of the titanic passengers, I am following Steps outlined by 
LD Freeman that can be found on kaggle: 
https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
to learn more about the non-machine learning part of the data science pipeline.
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Load data ==========
data_raw = pd.read_csv("train.csv")
data_val = pd.read_csv("test.csv")

data1 = data_raw.copy(deep=True)

data_cleaner = [data1, data_val]

print(data_raw.info())
data_raw.sample(10)


# ========== Cleaning the data ==========
"""
The below is pulled from Freeman's Notebook for quick reference and memory
committing.

The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting:

    Correcting: Reviewing the data, there does not appear to be an aberrant or 
non-acceptable data inputs. In addition, we see we may have potential outliers 
in age and fare. However, since they are reasonable values, we will wait until
after we complete our exploratory analysis to determine if we should include
or exclude from the dataset. It should be noted, that if they were unreasonable
values, for example age = 800 instead of 80, then it's probably a safe decision
to fix now. However, we want to use caution when we modify data from its 
original value, because it may be necessary to create an accurate model.

    Completing: There are null values or missing data in the age, cabin, and 
embarked field. Missing values can be bad, because some algorithms don't know
how-to handle null values and will fail. While others, like decision trees, 
can handle null values. Thus, it's important to fix before we start modeling, 
because we will compare and contrast several models. There are two common 
methods, either delete the record or populate the missing value using a 
reasonable input. It is not recommended to delete the record, especially a 
large percentage of records, unless it truly represents and incomplete record. 
Instead, it's best to impute imssing values. A basic methodology for 
quantitative data is impute using mean, median, or mean + randomized standard
deviation. An intermediate methodology is to use the basic methodology based on
specific criteria; like the average age by class or embark port by fare and 
SES. There are more complex methodologies, however before deploying, it should 
be compared to the base model to determin if complexity truly adds value. For 
this dataset, age will be imputed with the median, the cabin attribute will be
dropped, and embark will be imputed with mode. Subsequent model iterations may
modify this decision to determine if it improves the model's accuracy.

    Creating: Feature engineering is when we used existing features to create
new features to determine if they provide new signals to predict our outcome. 
For this dataset, we will create a title feature to determine if it played a
role in survival. 

    Coverting: Last, but certainly not least, we'll deal with formatting. There
are no date or currency formats, but datatype formats. Our categorical data
imported as objects, which makes it difficult for mathematical calculations. 
For this dataset, we will convert object datatypes to categorical dummy 
variables. 
"""

print("Training columns with null values:\n", data1.isnull().sum())
print("-"*20)
print("Test/Validation columns with null values:\n", data_val.isnull().sum())
print("-"*20)

### COMPLETEING
# Completing missing values (do outliers affect this?)
for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# deleting columns that are not needed
drop_columns = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_columns, axis=1, inplace=True)

print(data1.isnull().sum())
print("-"*20)
print(data_val.isnull().sum())


### CREATING
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 # default initialization to 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # Update 
    
    dataset['Title'] = dataset['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    
    # Creating bins for Fare and Age
    # See: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.qcut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    
    stat_min = 10
    title_names = (data1['Title'].value_counts() < stat_min)
    data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    print(data1['Title'].value_counts())
    print('-'*20)
    
    data1.info()
    data_val.info()
    data1.sample(10)
    
###CONVERT
# code categorical data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    
Target = ['Survived']

data1_x = ['Sex','Pclass','Embarked','Title','SibSp','Parch','Age','Fare',
           'FamilySize','IsAlone']
data1_x_calc = ['Sex_Code','Pclass','Embarked_Code','Title_Code','SibSp',
                'Parch','Age','Fare']
data1_xy = Target + data1_x
print('Original X y: ', data1_xy, '\n')

data1_x_bin = ['Sex_Code','Pclass','Embarked_Code','Title_Code','FamilySize',
               'AgeBin_Code','FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X y: ', data1_xy_bin, '\n')

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X y: ', data1_xy_dummy, '\n')

data1_dummy.head()


print('Train columns with null values: ', data1.isnull().sum())
print('-'*20)
print(data1.info())
print('-'*20)

print('Test/Val columns with null values: ', data_val.isnull().sum())
print('-'*20)
print(data_val.info())
print('-'*20)

data_raw.describe(include='all')



# ========== Splitting the data ==========
