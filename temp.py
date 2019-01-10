# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
df1=pd.read_csv(r'desktop',engine='python')

df1.dtypes
df1.head()
df1.tail()
df1.columns

list=['abcd',786,2.23,'python',70.2]
tinylist=[123,'python']
print(list)
print(list[0])


s = pd.Series([1,3,6,5,44,1])
import pandas as pd
import numpy as np
print(s)
pd.Series([4, 7, -5, 3])
pd.Series((4, 7, -5, 3), index=['d', 'b', 'a', 'c'])
sdata = { 'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000 }
pd.Series(sdata)
pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
df = pd.DataFrame(np.arange(16).reshape(4,4),columns=['a','b','c','d'])
print(df)
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1, 1, 50)
y = 2*x + 1
plt.figure()
plt.plot(x, y)
plt.title('fig1')
plt.show()
l1, = plt.plot(x, y1, label='linear line')
l2, = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
plt.legend(loc='upper right')
plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best')

import seaborn
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

diamonds=pd.read_csv('diamonds.csv',engine='python')
diamonds.head()
diamonds.shape
ds=diamonds.sample(frac=0.1) #随机抽取10%的数据
ds.shape

import os
d=r'Desktop/Pdataset'
os.chdir(d)

diamonds=pd.read_csv('diamonds.csv',engine='python')
diamonds.head()
diamonds.shape
ds=diamonds.sample(frac=0.1) #随机抽取10%的数据
ds.shape

import requests
from lxml import etree

url='https://book.douban.com/subject/1084336/comments/'
r=requests.get(url).text

s=etree.HTML(r)
file=s.xpath('//*[@id="comments"]/ul/li/div[2]/p/span/text()')
print(file)
import requests
def get_content(url):
    r=requests.get(url).text 
    s=etree.HTML(r)
    file=s.xpath('//*[@id="comments"]/ul/li/div[2]/p/span/text()')
    
    return file

file='douban.txt'

for i in range(1,5):
    url="http://book.douban.com/subject/1084336/comments/hot?p={}".format(i)
    file=get_content(url)
    with open(file,'a+',encoding='utf-8') as f:
        for line in file:
            f.write(line+'\n') 


import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import jieba
from imageio import imread
import numpy as np

file_name='douban.txt'
with open(file_name,encoding='utf-8') as f:
    text=f.read() 
    
wordlist_after_jieba = jieba.cut(text, cut_all = False)
wl_space_split = " ".join(wordlist_after_jieba)

font_path=r'C:/System/Library/Fonts/SimHei.ttf'
back_coloring_path = 'xiaowangzi.jpeg' # 设置背景图片路径
back_coloring = imread(back_coloring_path)# 设置背景图片

wc = WordCloud(font_path=font_path,  # 设置字体
               background_color="white",  # 背景颜色
               max_words=2000,  # 词云显示的最大词数
               mask=back_coloring,  # 设置背景图片
               max_font_size=100,  # 字体最大值
               random_state=42,
               width=1000, height=860, margin=2,# 设置图片默认的大小,但是如果使用背景图片的话,那么保存的图片大小将会按照其大小保存,margin为词语边缘距离
               )

wc.generate(wl_space_split)

plt.figure(figsize=(24,18))
plt.imshow(wc)
plt.axis("off")
plt.show()

imgname='output.jpg'
wc.to_file(imgname) 
%matplotlib qt5 

from random import shuffle  # 导入随机函数 shuffle，用来打乱数据
import seaborn as sns
from random import seed
import pandas as pd
######## load data
filename = 'cs-data.csv'
data = pd.read_csv(filename, engine='python')
data.head()
data.shape
data.describe()

data.fillna(data.mean(),inplace=True)sns.boxplot(y="NumberOfTime30-59DaysPastDueNotWorse", data=data)
data = data[-(data['NumberOfTime30-59DaysPastDueNotWorse'] > 80)]
sns.boxplot(y="NumberOfTime30-59DaysPastDueNotWorse", data=data)

sns.boxplot(y="NumberOfOpenCreditLinesAndLoans", data=data)
data.describe()

data.columns

data.plot(kind='box', subplots=True, layout=(4,4),sharex=False, sharey=False, fontsize=8,figsize=(24,18))
sns.boxplot(y="age", data=data)
data = data[-(data['age'] > 100)]

sns.boxplot(y="NumberOfTime30-59DaysPastDueNotWorse", data=data)
data = data[-(data['NumberOfTime30-59DaysPastDueNotWorse'] > 80)]
sns.boxplot(y="NumberOfTime30-59DaysPastDueNotWorse", data=data)

sns.boxplot(y="NumberOfOpenCreditLinesAndLoans", data=data)

sns.boxplot(y="DebtRatio", data=data)
data = data[-(data['DebtRatio'] > 100000)]
sns.boxplot(y="DebtRatio", data=data)

sns.boxplot(y="MonthlyIncome", data=data)
data = data[-(data['MonthlyIncome'] > 500000)]
sns.boxplot(y="MonthlyIncome", data=data)


data.hist(xlabelsize=7,ylabelsize=7,figsize=(16,12))
grouped=data[['SeriousDlqin2yrs']].groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()

grouped.plot(kind='bar')

grouped=grouped.reset_index(name='cnt')

grouped['percent']=grouped['cnt']/grouped['cnt'].sum()

grouped

df1=data.groupby('age')['age'].count().reset_index(name='cnt_total')

df2=data.query('SeriousDlqin2yrs==1').groupby('age')['age'].count().reset_index(name='cnt_Dlq')

df3=pd.merge(df1,df2,how='left',on='age').fillna(0).eval('percent_Dlq=cnt_Dlq/cnt_total')

df3.plot(x='age',y='percent_Dlq')
