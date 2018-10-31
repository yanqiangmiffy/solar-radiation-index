# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 02_xgb_cv_poly.py 
@Time: 2018/10/24 18:13
@Software: PyCharm 
@Description:
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from utils import load_feature

df_train, df_test, label = load_feature()
scaler=StandardScaler()
X=df_train.drop(['日期'], axis=1, inplace=False)
X=scaler.fit_transform(X)
y=label.values

# 预测结果的数据
sub_x=df_test.drop(['日期'], axis=1, inplace=False)
sub_x=scaler.fit_transform(sub_x)

kf = KFold(n_splits=5, random_state=123, shuffle=True)
clf=XGBRegressor(objective='reg:linear',
                 n_estimators=1000,
                 min_child_weight=1,
                 learning_rate=0.01,
                 max_depth=5,
                 n_jobs=4,
                 subsample=0.6,
                 colsample_bytree=0.4,
                 colsample_bylevel=1)

scores=[]
for index, (train_index, test_index) in enumerate(kf.split(X)):
    # print(index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=30,
            eval_metric=['mae'],
            verbose=False)

    # 验证模型
    test_pred=clf.predict(X_test)
    scores.append(mean_absolute_error(y_test,test_pred))

    # 预测结果
    sub_pred=clf.predict(sub_x)
    if index == 0:
        df_test['prediction'] = sub_pred
    else:
        df_test['prediction']= df_test['prediction']+sub_pred
print('score list:',scores)
print(np.mean(scores))

df_test['prediction']=df_test['prediction']/5

df_test.rename(columns={'日期':'time'},inplace=True)
df_test[['time', 'prediction']].to_csv('result/02_xgb_cv_poly.csv', index=False)