# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:50:31 2019

@author: Joseph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



# ========== Load Data ==========
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


data_cleaner = [train, test]

test_col = train['Fare']

def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower) / (upper - lower)
    return y

g = sns.distplot(normalize(test_col))