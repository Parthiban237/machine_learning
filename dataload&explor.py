# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:36:50 2026

@author: CIS LAB
"""

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["Species"] = iris.target

print("First five records")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nDataset information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nFeature names")
print(iris.feature_names)

print("\n Target Names")
print(iris.target_names)

print("\nMissing Values")
print(df.isnull().sum())