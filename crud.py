# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:57:56 2026

@author: CIS LAB
"""

import pandas as pd
df = pd.read_csv("record.csv")

print("Original Data")
print(df)

new_record=pd.DataFrame({
    "student":["Kalai"],
    "rollno":[2239020014],
    "phno":[9876504321]
    })

df=pd.concat([df,new_record],ignore_index=True)

print("\nAfter Inserting Record")
print(df)

df = df.drop(index = 1)
print("\nAfter Deleting Record")
print(df)

df.to_csv("record_updated.csv",index=False)
print('\nUpdated csv saved successfully')