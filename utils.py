# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 21:53
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm
import pandas  as pd

def load_data():
    df_train_feature=pd.read_csv('input/train_feature.csv')
    df_train_label=pd.read_csv('input/train_label.csv')
    df_train_feature=df_train_feature.groupby(by='日期').mean().reset_index()
    df_train_feature['辐照度'] = df_train_feature['辐照度'] * 8
    df_train=pd.merge(df_train_feature,df_train_label)
    df_train.drop(['时刻'],axis=1,inplace=True)

    df_test_feature=pd.read_csv('input/test_feature.csv')
    df_test_feature=df_test_feature.groupby(by='日期').mean().reset_index()
    df_test_feature['辐照度'] = df_test_feature['辐照度'] * 8
    df_test=df_test_feature.drop(['时刻'],axis=1)

    return df_train,df_test

