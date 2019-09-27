# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:33:43 2019

@author: Joseph
"""

from collections import Counter
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices 
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    
    Note: this should be added to a src toolkit
    """
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    # select observations (rows) containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    observation_with_multiple_outliers = list(
        k for k, v in outlier_indices.items() if v > n
    )
    
    return observation_with_multiple_outliers
