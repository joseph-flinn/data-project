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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ========== Load data ==========
data_raw = pd.read_csv("data/train.csv")
data_val = pd.read_csv("data/test.csv")

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
from sklearn import model_selection
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(
    data1[data1_x_calc], data1[Target], random_state=0
)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(
    data1[data1_x_bin], data1[Target], random_state=0
)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(
    data1_dummy[data1_x_dummy], data1[Target], random_state=0
)

print('Data1 Shape: {}'.format(data1.shape))
print('Train Shape: {}'.format(train1_x.shape))
print('Test1 Shape: {}'.format(test1_x.shape))

train1_x_bin.head()


# ========== Exploratory Analysis ==========
# Float type data
for x in data1_x: 
    if data1[x].dtype != 'float64':
        print('Survival Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*20, '\n')
        
# string data
print(pd.crosstab(data1['Title'], data1[Target[0]]))


# Visualizing. Plotted different ways for educational purposes
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans=True, meanline=True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(x=data1['Age'], showmeans=True, meanline=True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')


plt.subplot(233)
plt.boxplot(x=data1['FamilySize'], showmeans=True, meanline=True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x=[data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']],
         stacked=True, color=['g','r'], label=['Survived', 'Dead'])
plt.title('Fare Histogram Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x=[data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']],
         stacked=True, color=['g','r'], label=['Survived','Dead'])
plt.title('Age HIstogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x=[data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']],
         stacked=True, color=['g','r'], label=['Survived', 'Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


# Graphing individual features by survival
import seaborn as sns

fig, saxis = plt.subplots(2, 3, figsize=(16, 12))
sns.barplot(x='Embarked', y='Survived', data=data1, ax=saxis[0, 0])
sns.barplot(x='Pclass', y='Survived', order=[1,2,3], data=data1, ax=saxis[0, 1])
sns.barplot(x='IsAlone', y='Survived', order=[1,0], data=data1, ax=saxis[0, 2])
sns.pointplot(x='FareBin', y='Survived', data=data1, ax=saxis[1, 0])
sns.pointplot(x='AgeBin', y='Survived', data=data1, ax=saxis[1, 1])
sns.pointplot(x='FamilySize', y='Survived', data=data1, ax=saxis[1, 2])


# Graphing distribution of qualitative data: Pclass
# We know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(14, 12))
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data1, ax=axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data1, split=True, ax=axis2)
axis2.set_title('Pclass vs Age Survival Comparsion')
sns.boxplot(x='Pclass', y='FamilySize', hue='Survived', data=data1, ax=axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')

# Graphing distribution of qualitative data: Sex
# We know sex mattered in survival, no let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1, 3, figsize=(14,12))
sns.barplot(x='Sex', y='Survived', hue='Embarked', data=data1, ax=qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=data1, ax=qaxis[1])
axis2.set_title('Sex vs Pclass Survival Comparison')
sns.barplot(x='Sex', y='Survived', hue='IsAlone', data=data1, ax=qaxis[2])
axis3.set_title('Sex vs IsAlone Survival Comparison')


# more side-by-side comparisons
fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=(14, 12))

# how does family size factor with sex & survival compare
sns.pointplot(x='FamilySize', y='Survived', hue='Sex', data=data1, 
              palette={'male': 'blue', 'female': 'pink'},
              markers=['*', 'o'], linestyles=['-', '--'], ax=maxis1)

# how does class factor with sex & survival compare
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data1,
              palette={'male': 'blue', 'female': 'pink'},
              markers=['*', 'o'], linestyles=['-', '--'], ax=maxis2)


# how does embark port factor with class, sex and survival compare
e = sns.FacetGrid(data1, col='Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=0.95, palette='deep')
e.add_legend()

# plot distributions of age of passengers who survived or did not survive
a = sns.FacetGrid(data1, hue='Survived', aspect=4)
a.map(sns.kdeplot, 'Age', shade=True)
a.set(xlim=(0, data1['Age'].max()))
a.add_legend()

# histogram comparison of sex, class, and age by survival
h = sns.FacetGrid(data1, row='Sex', col='Pclass', hue='Survived')
h.map(plt.hist, 'Age', alpha=0.75)
h.add_legend()

# pair plots of entire dataset
pp = sns.pairplot(data1, hue='Survived', palette='deep', height=1.2,
                  diag_kind='kde', diag_kws=dict(shade=True), 
                  plot_kws=dict(s=10))
pp.set(xticklabels=[])


# correlation heatmap of dataset
def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14,12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    
    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink':0.9},
        ax=ax,
        annot=True,
        linewidths=0.1,
        vmax=1.0,
        linecolor='white',
        annot_kws={'fontsize':12}
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    
correlation_heatmap(data1)


# ========== Data Model ==========
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import discriminant_analysis
#from xgboost import XGBClassifier
MLA = [
    # Ensemble Methods  
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    # Naive Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    # Nearest Neighbors
    neighbors.KNeighborsClassifier(),
    
    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    # xgboost
    #XGBClassifier(),
    
    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis()
]

# split the dataset in cross-validation with a splitter class
cv_split = model_selection.ShuffleSplit(
        n_splits=10, test_size=0.3, train_size=0.6, random_state=0
)

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 
               'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

MLA_predict = data1[Target]

row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    # score model with CV
    cv_results = model_selection.cross_validate(
        alg, data1[data1_x_bin], data1[Target], cv=cv_split, 
        return_train_score=True
    )
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    # if this is a non-bias random sample, then +/- 3 std from the mean, should
    # statistically capture 99.7 of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
    
    row_index+=1
    
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)

sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='m')
plt.title('Machine Learning Algorithm Accuracy Score\n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')


# ========= Model Evaluation =========

# Coin Flip Algorithm
import random
from sklearn import metrics
for index, row in data1.iterrows():
    if random.random() > 0.5:
        data1.set_value(index, 'Random_Predict', 1)
    else:
        data1.set_value(index, 'Random_Predict', 0)
        
data1['Random_Score'] = 0
data1.loc[(data1['Survived'] == data1['Random_Predict']), 'Random_Score'] = 1
print('Coin Flip Model Accuracy: {:.2f}%'.format(data1['Random_Score'].mean()*100))
print('Coin Flip Model Accuracy w/SciKit {:2f}%'.format(metrics.accuracy_score(data1['Survived'], data1['Random_Predict'])*100))

pivot_female = data1[data1['Sex']=='female'].groupby(['Sex', 'Pclass', 'Embarked', 'FareBin'])['Survived'].mean()
print('Survival Decision Tree w/Female Node: \n', pivot_female)
pivot_male = data1[data1['Sex']=='male'].groupby(['Sex', 'Title'])['Survived'].mean()
print('\n\nSurvival Decision Tree w/Male Node: \n', pivot_male)


# hand made model using brain power
def mytree(df):
    Model = pd.DataFrame(data = {'Predict':[]})
    male_title = ['Master']
    
    for index, row in df.iterrows():
        # Q1: Were you on the Titanic; majority dead
        Model.loc[index, 'Predict'] = 0
        # Q2: Are you female; majority survived
        if (df.loc[index, 'Sex'] == 'female'):
            Model.loc[index, 'Predict'] = 1
        # Q3 A Female - Class and Question 4 Embarked gain minimum information
        
        # 5B Female - FareBin; set anything less than 0.5 in female node decision to tree back to 0
        if ((df.loc[index, 'Sex'] == 'female') &
                (df.loc[index, 'Pclass'] == 3) &
                (df.loc[index, 'Embarked'] == 'S') &
                (df.loc[index, 'Fare'] > 8)):
            Model.loc[index, 'Predict'] = 0
            
        # Q3B Male: Title; set anything greater than 0.5 to 1 for majority survived
        if ((df.loc[index, 'Sex'] == 'male') &
                (df.loc[index, 'Title'] in male_title)):
           Model.loc[index, 'Predict'] == 1
           
    return Model

Tree_Predict = mytree(data1)
print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(data1['Survived'], Tree_Predict)*100))
print(metrics.classification_report(data1['Survived'], Tree_Predict))

# Plot Accuracy Summary
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(data1['Survived'], Tree_Predict)
np.set_printoptions(precision=2)

class_names = ['Dead', 'Survived']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


# ========== Tune Model with Hyper Parameters ==========
# base model
dtree = tree.DecisionTreeClassifier(random_state=0)
base_results = model_selection.cross_validate(
        dtree, data1[data1_x_bin], data1[Target], cv=cv_split,
        return_train_score=True
)
dtree.fit(data1[data1_x_bin], data1[Target])

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

tune_model = model_selection.GridSearchCV(
    tree.DecisionTreeClassifier(), param_grid=param_grid, 
    scoring='roc_auc', cv=cv_split, return_train_score=True
)

tune_model.fit(data1[data1_x_bin], data1[Target])

print('AFTER DT Parameters: ', tune_model.get_params())
print('AFTER DT Training w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('AFTER DT Test w/bin score mean {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('AFTER DT Test w/bin score 3*STD: +/- {:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100))
print('-'*20)


# ========== Tune Model with Feature Selection ==========
from sklearn import feature_selection

# base model
print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape)
print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)
print('BEFORE DT RFE Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('BEFORE DT RFE Test w/bin score mean {:.2f}'.format(base_results['test_score'].mean()*100))
print('BEFORE DT RFE Test w/bin score 3*STD: +/- {:.2f}'.format(base_results['train_score'].mean()*100*3))
print('-'*20)

dtree_rfe = feature_selection.RFECV(
    dtree, step=1, scoring='accuracy', cv=cv_split
)
dtree_rfe.fit(data1[data1_x_bin], data1[Target])

X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(
        dtree, data1[X_rfe], data1[Target], cv=cv_split, 
        return_train_score=True
)

print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape)
print('AFTER DT RFE Training Columns New: ', X_rfe)
print('AFTER DT RFE Training w/bin score mean: {:.2f}'.format(rfe_results['train_score'].mean()*100))
print('AFTER DT RFE Test w/bin score mean: {:.2f}'.format(rfe_results['test_score'].mean()*100))
print('AFTER DT RFE Test w/bin score 3*S: {:.2f}'.format(rfe_results['test_score'].mean()*100*3))
print('-'*20)

rfe_tune_model = model_selection.GridSearchCV(
    tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc',
    cv=cv_split, return_train_score=True
)
rfe_tune_model.fit(data1[X_rfe], data1[Target])

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.get_params())
print('AFTER DT RFE Tuned Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('AFTER DT RFE Tuned Test w/bin score mean {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('AFTER DT RFE Tuned Test w/bin score 3*STD: +/- {:.2f}'.format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100))
print('-'*20)

# Graph MLA version of Decision Tree
"""
# Note: Not working because Anaconda isn't picking up the newly installed 
#       graphviz library
import graphviz
dot_data = tree.export_graphviz(
        dtree, out_file=None, feature_names=data1_x_bin, class_names=True
)
graph = graphviz.Source(dot_data)
graph
"""


# ========== Validate and Implement ==========
correlation_heatmap(MLA_predict)

# why choose one model when you can choose all with a voting classifier?
vote_est = [
    # Ensemble Methods
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    
    # Gaussian Process
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    # GLM
    ('lr', linear_model.LogisticRegressionCV()),
    
    # Naive Bayes
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    # Nearest Neighbor
    ('knn', neighbors.KNeighborsClassifier()),
    
    # SVM
    ('svc', svm.SVC(probability=True))
]

vote_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard',)
vote_hard_cv = model_selection.cross_validate(
    vote_hard, data1[data1_x_bin], data1[Target].values.ravel(), cv=cv_split, 
    return_train_score=True
)
vote_hard.fit(data1[data1_x_bin], data1[Target].values.ravel())

print('Hard Voting Training w/bin score mean: {:.2f}'.format(vote_hard_cv['train_score'].mean()*100))
print('Hard Voting Test w/bin score mean: {:.2f}'.format(vote_hard_cv['test_score'].mean()*100))
print('Hard Voting Test w/bin score 3*STD: {:.2f}'.format(vote_hard_cv['test_score'].mean()*100*3))
print('-'*20)

vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
vote_soft_cv = model_selection.cross_validate(
    vote_soft, data1[data1_x_bin], data1[Target].values.ravel(), cv=cv_split, 
    return_train_score=True
)
vote_soft.fit(data1[data1_x_bin], data1[Target].values.ravel())

print('Soft Voting Training w/bin score mean: {:.2f}'.format(vote_soft_cv['train_score'].mean()*100))
print('Soft Voting Test w/bin score mean: {:.2f}'.format(vote_soft_cv['test_score'].mean()*100))
print('Soft Voting Test w/bin score 3*STD: {:.2f}'.format(vote_soft_cv['test_score'].mean()*100*3))
print('-'*20)


# Hyperparameter Tune with GridSearchCV
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [0.1, 0.25, 0.5, 0.75, 1.0]
grid_learn = [0.01, 0.03, 0.05, 0.1, 0.25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, 0.03, 0.05, 0.10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed= [0]

grid_param = [
    [{  # AdaBoostClassifier
        'n_estimators': grid_n_estimator,
        'learning_rate': grid_learn,
        'random_state': grid_seed
    }],
    [{  # BaggingClassifier
        'n_estimators': grid_n_estimator,
        'max_samples': grid_ratio,
        'random_state': grid_seed
    }],
    [{  # ExtraTreesClassifier 
        'n_estimators': grid_n_estimator,
        'criterion': grid_criterion,
        'max_depth': grid_max_depth,
        'random_state': grid_seed
    }],
    [{  # GradientBoostingClassifier
        'learning_rate': [0.05],
        'n_estimators': [300],
        'max_depth': grid_max_depth,
        'random_state': grid_seed
    }],
    [{  # RandomForestClassifier
        'n_estimators': grid_n_estimator,
        'criterion': grid_criterion,
        'max_depth': grid_max_depth,
        'oob_score': [True],
        'random_state': grid_seed
    }],
    [{  # GaussianProcessClassifier
        'max_iter_predict': grid_n_estimator,
        'random_state': grid_seed
    }],
    [{  # LogisticRegressionCV
        'fit_intercept': grid_bool,
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'random_state': grid_seed
    }],
    [{  # BernoulliNB
        'alpha': grid_ratio
    }],
    [{  # GaussianNB
    }],
    [{  # KNeighborsClassifier
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }],
    [{  # SVC
        'C': [1, 2, 3, 4, 5],
        'gamma': grid_ratio,
        'decision_function_shape': ['ovo', 'ovr'],
        'probability': [True],
        'random_state': grid_seed
    }]
]

import time
start_total = time.perf_counter()
for clf, param in zip(vote_est, grid_param):
    start = time.perf_counter()
    best_search = model_selection.GridSearchCV(
        estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc',
        return_train_score=True
    )
    best_search.fit(data1[data1_x_bin], data1[Target].values.ravel())
    run = time.perf_counter() - start
    
    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param)
    
run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes'.format(run_total/60))

print('-'*20)


grid_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard',)
grid_hard_cv = model_selection.cross_validate(
    grid_hard, data1[data1_x_bin], data1[Target].values.ravel(), cv=cv_split, 
    return_train_score=True
)
grid_hard.fit(data1[data1_x_bin], data1[Target].values.ravel())

print('Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}'.format(grid_hard_cv['train_score'].mean()*100))
print('Hard Voting w/Tuned Hyperparameters  Test w/bin score mean: {:.2f}'.format(grid_hard_cv['test_score'].mean()*100))
print('Hard Voting w/Tuned Hyperparameters  Test w/bin score 3*STD: {:.2f}'.format(grid_hard_cv['test_score'].mean()*100*3))
print('-'*20)

grid_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
grid_soft_cv = model_selection.cross_validate(
    grid_soft, data1[data1_x_bin], data1[Target].values.ravel(), cv=cv_split, 
    return_train_score=True
)
grid_soft.fit(data1[data1_x_bin], data1[Target].values.ravel())

print('Soft Voting w/Tuned Hyperparameters  Training w/bin score mean: {:.2f}'.format(grid_soft_cv['train_score'].mean()*100))
print('Soft Voting w/Tuned Hyperparameters  Test w/bin score mean: {:.2f}'.format(grid_soft_cv['test_score'].mean()*100))
print('Soft Voting w/Tuned Hyperparameters  Test w/bin score 3*STD: {:.2f}'.format(grid_soft_cv['test_score'].mean()*100*3))
print('-'*20)


# prepare data for modelling
print(data_val.info())
print('-'*20)

data_val['Survived'] = mytree(data_val).astype(int)
# hard voting classifier w/full dataset modeling submission score
data_val['Survived'] = grid_hard.predict(data_val[data1_x_bin])

submit = data_val[['PassengerId', 'Survived']]
submit.to_csv("submissions/framework_submission.csv", index=False)

print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize=True))
submit.sample(10)
