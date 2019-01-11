# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:09:03 2018

@author: hc
"""

list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]


list1[0]
list2[0:3]

dict1 = {'a': 1, 'b': 2, 'c': 'apple'}
dict1['b'] 


import pandas as pd
import numpy as np
s = pd.Series([1,3,6,5,44,1])
print(s)

pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
df = pd.DataFrame(np.arange(16).reshape(4,4),columns=['a','b','c','d'])
print(df)

df.columns
df.index
df.values
df.dtypes

df.head()
df.tail()
df.describe()

df[0:2]
df['a']
df[['a','b']]
df.loc[:,'a']
df.iloc[0,0]

df[df['a']>0]

df.iloc[0,0]=100
df
df['b']=99
df
df.loc[df['c']==10,'d']=88
df

df['a']+df['b']
df['d']=df['a']+df['b']
df

import matplotlib.pyplot as plt
plt.plot(df.index,df['a'])
plt.plot(df.index,df['b'])


