# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 21:53
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm
import pandas  as pd
from sklearn.preprocessing import PolynomialFeatures

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


def create_fea(df_data):
    df_data=to_one_day(df_data)
    col_names = df_data.columns.tolist()[1:]
    for i in range(6):
        temp=[]
        for j in range(i,len(col_names),6):
            temp.append(col_names[j])
        # print(df_data[temp])
        df_data[temp[0].split('_')[1]+'_mean']=df_data[temp].mean(axis=1)
        df_data[temp[0].split('_')[1]+'std']=df_data[temp].std(axis=1)
        df_data[temp[0].split('_')[1]+'_min']=df_data[temp].min(axis=1)
        df_data[temp[0].split('_')[1]+'_max']=df_data[temp].max(axis=1)

    co = ['风速_mean', '辐照度_mean', '风向_mean', '温度_mean', '湿度_mean', '气压_mean']

    df_data = add_poly_features(df_data, co)
    # print(df_data.columns)
    return df_data


def load_feature():
    df_train = pd.read_csv('input/train_feature.csv')
    label = pd.read_csv('input/train_label.csv')['电场实际太阳辐射指数']
    df_test = pd.read_csv('input/test_feature.csv')
    df_train=create_fea(df_train)
    df_test=create_fea(df_test)
    return df_train,df_test,label

