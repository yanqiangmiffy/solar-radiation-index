# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 02_new_feature.py 
@Time: 2018/10/18 19:17
@Software: PyCharm 
@Description:
"""


# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold





def to_one_day(data_df):
    data_df_date = data_df[['日期']].drop_duplicates().reset_index(drop=True)
    # print(data_df_date['日期'])
    for i in range(1, data_df['日期'].max()+1):
        data = data_df[data_df['日期'] == i].reset_index(drop=True)

        data_df_date.loc[i - 1, '2_风速'] = data.loc[0, '风速']
        data_df_date.loc[i - 1, '2_风向'] = data.loc[0, '风向']
        data_df_date.loc[i - 1, '2_温度'] = data.loc[0, '温度']
        data_df_date.loc[i - 1, '2_湿度'] = data.loc[0, '湿度']
        data_df_date.loc[i - 1, '2_气压'] = data.loc[0, '气压']

        # data_df_date.loc[i - 1, '5_辐照度'] = data.loc[1, '辐照度']
        data_df_date.loc[i - 1, '5_风速'] = data.loc[1, '风速']
        data_df_date.loc[i - 1, '5_风向'] = data.loc[1, '风向']
        data_df_date.loc[i - 1, '5_温度'] = data.loc[1, '温度']
        data_df_date.loc[i - 1, '5_湿度'] = data.loc[1, '湿度']
        data_df_date.loc[i - 1, '5_气压'] = data.loc[1, '气压']

        data_df_date.loc[i - 1, '8_辐照度'] = data.loc[2, '辐照度']
        data_df_date.loc[i - 1, '8_风速'] = data.loc[2, '风速']
        data_df_date.loc[i - 1, '8_风向'] = data.loc[2, '风向']
        data_df_date.loc[i - 1, '8_温度'] = data.loc[2, '温度']
        data_df_date.loc[i - 1, '8_湿度'] = data.loc[2, '湿度']
        data_df_date.loc[i - 1, '8_气压'] = data.loc[2, '气压']

        data_df_date.loc[i - 1, '11_辐照度'] = data.loc[3, '辐照度']
        data_df_date.loc[i - 1, '11_风速'] = data.loc[3, '风速']
        data_df_date.loc[i - 1, '11_风向'] = data.loc[3, '风向']
        data_df_date.loc[i - 1, '11_温度'] = data.loc[3, '温度']
        data_df_date.loc[i - 1, '11_湿度'] = data.loc[3, '湿度']
        data_df_date.loc[i - 1, '11_气压'] = data.loc[3, '气压']

        data_df_date.loc[i - 1, '14_辐照度'] = data.loc[4, '辐照度']
        data_df_date.loc[i - 1, '14_风速'] = data.loc[4, '风速']
        data_df_date.loc[i - 1, '14_风向'] = data.loc[4, '风向']
        data_df_date.loc[i - 1, '14_温度'] = data.loc[4, '温度']
        data_df_date.loc[i - 1, '14_湿度'] = data.loc[4, '湿度']
        data_df_date.loc[i - 1, '14_气压'] = data.loc[4, '气压']

        data_df_date.loc[i - 1, '17_辐照度'] = data.loc[5, '辐照度']
        data_df_date.loc[i - 1, '17_风速'] = data.loc[5, '风速']
        data_df_date.loc[i - 1, '17_风向'] = data.loc[5, '风向']
        data_df_date.loc[i - 1, '17_温度'] = data.loc[5, '温度']
        data_df_date.loc[i - 1, '17_湿度'] = data.loc[5, '湿度']
        data_df_date.loc[i - 1, '17_气压'] = data.loc[5, '气压']

        data_df_date.loc[i - 1, '20_辐照度'] = data.loc[6, '辐照度']
        data_df_date.loc[i - 1, '20_风速'] = data.loc[6, '风速']
        data_df_date.loc[i - 1, '20_风向'] = data.loc[6, '风向']
        data_df_date.loc[i - 1, '20_温度'] = data.loc[6, '温度']
        data_df_date.loc[i - 1, '20_湿度'] = data.loc[6, '湿度']
        data_df_date.loc[i - 1, '20_气压'] = data.loc[6, '气压']

        # data_df_date.loc[i - 1, '23_辐照度'] = data.loc[7, '辐照度']
        data_df_date.loc[i - 1, '23_风速'] = data.loc[7, '风速']
        data_df_date.loc[i - 1, '23_风向'] = data.loc[7, '风向']
        data_df_date.loc[i - 1, '23_温度'] = data.loc[7, '温度']
        data_df_date.loc[i - 1, '23_湿度'] = data.loc[7, '湿度']
        data_df_date.loc[i - 1, '23_气压'] = data.loc[7, '气压']
    return data_df_date


def creat_fea(data_df):
    data_df = to_one_day(data_df)
    # data_df['风速_mean'] = data_df[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].mean(axis=1)
    # data_df['风速_std'] = data_df[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].std(axis=1)
    # data_df['风速_min'] = data_df[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].min(axis=1)
    # data_df['风速_max'] = data_df[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].max(axis=1)
    #
    # data_df['辐照度_mean'] = data_df[[ '8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].mean(
    #     axis=1)
    # data_df['辐照度_std'] = data_df[['8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].std(
    #     axis=1)
    # data_df['辐照度_min'] = data_df[['8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].min(
    #     axis=1)
    # data_df['辐照度_max'] = data_df[[ '8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].max(
    #     axis=1)
    #
    # data_df['风向_mean'] = data_df[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].mean(axis=1)
    # data_df['风向_std'] = data_df[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].std(axis=1)
    # data_df['风向_min'] = data_df[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].min(axis=1)
    # data_df['风向_max'] = data_df[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].max(axis=1)
    #
    # data_df['温度_mean'] = data_df[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].mean(axis=1)
    # data_df['温度_std'] = data_df[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].std(axis=1)
    # data_df['温度_min'] = data_df[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].min(axis=1)
    # data_df['温度_max'] = data_df[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].max(axis=1)
    #
    # data_df['湿度_mean'] = data_df[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].mean(axis=1)
    # data_df['湿度_std'] = data_df[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].std(axis=1)
    # data_df['湿度_min'] = data_df[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].min(axis=1)
    # data_df['湿度_max'] = data_df[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].max(axis=1)
    #
    # data_df['气压_mean'] = data_df[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].mean(axis=1)
    # data_df['气压_std'] = data_df[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].std(axis=1)
    # data_df['气压_min'] = data_df[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].min(axis=1)
    # data_df['气压_max'] = data_df[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].max(axis=1)
    return data_df


def load_feature():
    df_train = pd.read_csv('input/train_feature.csv')
    label = pd.read_csv('input/train_label.csv')['电场实际太阳辐射指数']
    df_test = pd.read_csv('input/test_feature.csv')
    df_train=creat_fea(df_train)
    df_test=creat_fea(df_test)
    return df_train,df_test,label



