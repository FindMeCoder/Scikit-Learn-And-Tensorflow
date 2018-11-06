# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:48:28 2018

@author: 寻ME
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path) #解压方式
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()

#head()输出前五个数据和表头
housing.head()
#info输出每个特征元素的个数，因此可以查看缺失值，还可以查看数据类型和内存占用
housing.info()
#value_count 统计每个元素的总个数
housing["ocean_proximity"].value_counts()
#describe()可以将实数的最大值，最小值，方差，总个数，75%，50%，25%小值
housing.describe()
#hist输出实数域的直方图
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()
#分开测试集和训练集
import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=split_train_test(housing,0.2)
print(len(train_set),"train+",len(test_set),"test")

#可以采用每个样本的识别码来决定是否放入测试集。可以计算样本的hash值，取得最后一个值，若小于256*0.2则放入测试集

def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest[-1]<256*test_ratio

def split_train_test_by_id(data,test_ratio,id_column,hash):
    ids=data[id_column]
    id_test_set=ids.apply(lambda id_:test_set_check(id_,test_ratio,hash))
    return data.loc[~id_test_set],data.loc[id_test_set]

housing_with_id=housing.reset_index() #add an index
train_set,test_set=split_train_test_by_id(housing_with_id,0.2,len(housing),"index")
#skicit_learn 提供了简介的分开训练集和测试集的函数
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)

#将income_cat分为5类

#可以使用分层验证
housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["median_income"]<5,5.0,inplace=True)
#分层抽样
from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
      strat_train_set=housing.loc[train_index]
      strat_test_set=housing.loc[test_index]
    
#由于income_cat知识我们手动划分的特征，所以最后把他删除
      
for set in(strat_train_set,strat_test_set):
    set.drop(["income_cat"],axis=1,inplace=True)
    
#为了防止错误操作，我们将训练集复制一份
train_housing=strat_train_set.copy()
train_housing.plot(kind="scatter",x="longitude",y="latitude")    

train_housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True) 
plt.legend()  

#查看median_house_value与其他变量的线性相关性，越靠近1越相关，靠近-1为负相关
corr_matrix=train_housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#然而上述只是线性相关性，还有可能是非线性的关系
from pandas.tools.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms"]
scatter_matrix(housing[attributes],figsize=(12,8))

#可以利用特征组合，或许两个目标的相关性都不强，可是组合起来或许有较大的提升，还可以尝试组合不同的特征

#数据预处理

train_housing=strat_train_set.drop("median_house_value",axis=1)
train_housing_labels=strat_train_set["median_house_value"].copy()

#数据清洗：1.去掉缺失值的个体 2.去掉含有缺失值的特征 3. 给缺失值不上一些值（0，平均数，中位数）

#train_housing.dropna(subset=["total_bedrooms"]) #option 1
#train_housing.drop("total_bedrooms",axis=1) #option 2
median=train_housing["total_bedrooms"].median()
train_housing["total_bedrooms"].fillna(median)

#我们也可以对Scikit_learn存在对缺失值处理的类，可以使用imputer定义一个补缺失值的策略，最后使用fit方法执行操作

from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="median")
housing_num=train_housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

#处理文本类别特征，可以将文本编码为实数特征，对应的类为LableEncoder,使用fit_transform自动将文本编码

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
housing_cat=train_housing["ocean_proximity"]
housing_cat_encoded=encoder.fit_transform(housing_cat)
print(housing_cat)
print(housing_cat_encoded)

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
housing_cat_1hot=encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
#默认存放方式为稀疏矩阵，可以使用toarray查看

















    
    
    













    