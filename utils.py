# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 21:53
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm
import pandas  as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

def load_data():
    df_train_feature=pd.read_csv('input/train_feature.csv')
    df_train_label=pd.read_csv('input/train_label.csv')
    df_train_feature=df_train_feature.groupby(by='日期').mean().reset_index()
    # df_train_feature['辐照度'] = df_train_feature['辐照度'] * 8
    df_train=pd.merge(df_train_feature,df_train_label)
    # df_train.drop(['时刻'],axis=1,inplace=True)

    df_test_feature=pd.read_csv('input/test_feature.csv')
    df_test=df_test_feature.groupby(by='日期').mean().reset_index()
    # df_test['辐照度'] = df_test['辐照度'] * 8
    # df_test=df_test.drop(['时刻'],axis=1)

    return df_train,df_test


def to_one_day(df_data):
    grouped=df_data.groupby('日期')
    quarters=df_data['时刻'].drop_duplicates()
    fea_cols=['辐照度','风速','风向','温度','湿度','气压']

    day_col_names=['日期']
    for qu in quarters:
        for col in fea_cols:
           day_col_names.append(str(qu)+'_'+col)
    day_features=[]
    for index,group in grouped:
        temp=[index]
        for i in range(group.shape[1]):
            temp.extend(group[fea_cols].iloc[i].tolist())
        day_features.append(temp)
    df_day_data=pd.DataFrame(day_features,columns=day_col_names,index=None)
    return df_day_data

# df_day_data=to_one_day(df_train_feature)
# print(df_day_data)


def add_poly_features(data,column_names):
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features


def cal_base(df_data):
    """
    # 计算 mean std min max
    :return:
    """
    # 计算每日的数据特征均值 方差、最大值和最小值
    col_names = df_data.columns.tolist()[1:]
    for i in range(6):
        temp = []
        for j in range(i, len(col_names), 6):
            temp.append(col_names[j])
        # print(df_data[temp])
        df_data[temp[0].split('_')[1] + '_mean'] = df_data[temp].mean(axis=1)  # axis 按行计算得到均值
        df_data[temp[0].split('_')[1] + 'std'] = df_data[temp].std(axis=1)
        df_data[temp[0].split('_')[1] + '_min'] = df_data[temp].min(axis=1)
        df_data[temp[0].split('_')[1] + '_max'] = df_data[temp].max(axis=1)
    return df_data


def process_fuzhaodu(df_data):
    """
    构建辐照度特征
    :param df_data:
    :return:
    """
    df_data.drop(columns=['2_辐照度', '5_辐照度', '23_辐照度'], inplace=True)
    df_data['8_辐照度0'] = df_data['8_辐照度'].map(lambda x: 0 if x > 0 else 1)
    df_data['20_辐照度0'] = df_data['20_辐照度'].map(lambda x: 0 if x > 0 else 1)
    df_data = pd.get_dummies(df_data, columns=['8_辐照度0', '20_辐照度0'])
    return df_data


def process_wind_direction(df_data):
    cols=['2_风向','5_风向','8_风向','11_风向',
          '14_风向','17_风向','20_风向','23_风向']
    EN = [0] * len(df_data)
    ES = [0] * len(df_data)
    WS = [0] * len(df_data)
    WN = [0] * len(df_data)
    # 风向_EN 风向_ES 风向_WS 风向_WN
    for index in range(len(df_data)):
        for dir in df_data.iloc[index][cols]:
            if 0<=dir<90:
                EN[index]+=1
            elif 90<=dir<180:
                ES[index] += 1
            elif 180<=dir<270:
                WS[index] += 1
            else:
                WN[index] += 1
    df_data['风向_EN']=EN
    df_data['风向_ES']=ES
    df_data['风向_WS']=WS
    df_data['风向_WN']=WN
    # print(df_data['风向_EN'])
    return df_data


def process_temp(df_data):
    """
    处理温度
    :return:
    """
    ## 计算温度日较差模型：日照平均值*温度差平方根
    df_data['温度_sub'] = np.sqrt(df_data['温度_max'] - df_data['温度_min'])
    return df_data


def create_fea(df_data):
    df_data=to_one_day(df_data)
    # 计算统计值
    df_data=cal_base(df_data)
    # # 处理辐照度
    # df_data = process_fuzhaodu(df_data)
    # 处理风速
    df_data=process_wind_direction(df_data)
    # 处理温度
    df_data=process_temp(df_data)

    # 增加二阶特征
    co = ['风速_mean', '辐照度_mean', '风向_mean', '温度_mean', '湿度_mean', '气压_mean']
    df_data = add_poly_features(df_data, co)

    df_data['mean_fsfx'] = df_data['风速_mean'] * df_data['风向_mean']
    df_data['mean_sdqy'] = df_data['气压_mean'] * df_data['湿度_mean']
    df_data['mean_fswd'] = df_data['风速_mean'] * df_data['温度_mean']
    df_data['mean_fswd_add'] = df_data['风速_mean'] + df_data['温度_mean']

    for i in range(0, len(df_data)):
        if i <1:
            df_data.loc[i, 'old_辐照度_mean'] = df_data.loc[i, '辐照度_mean']
        elif i < 2:
            df_data.loc[i, 'old_辐照度_mean'] = (df_data.loc[i, '辐照度_mean'] + df_data.loc[i-1, '辐照度_mean'])/2
        else:
            df_data.loc[i, 'old_辐照度_mean'] = (df_data.loc[i, '辐照度_mean'] + df_data.loc[i - 1, '辐照度_mean']+ df_data.loc[i - 2, '辐照度_mean']) / 3

    return df_data


def load_feature():
    df_train = pd.read_csv('input/train_feature.csv')
    label = pd.read_csv('input/train_label.csv')['电场实际太阳辐射指数']
    df_test = pd.read_csv('input/test_feature.csv')
    df_train=create_fea(df_train)
    df_test=create_fea(df_test)
    return df_train,df_test,label


# df_train,df_test,label=load_feature()
