# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 22:32
# @Author  : quincyqiang
# @File    : 1_ensemble_regressor.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor,HuberRegressor,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from utils import load_feature
from sklearn.metrics import mean_absolute_error

SEED=222
df_train,df_test,label=load_feature()
# scaler=StandardScaler()

def get_train_test(test_size=0.2):
    X = df_train.drop(['日期'], axis=1, inplace=False)
    # X = scaler.fit_transform(X)
    y = label.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test=get_train_test()

def get_models():
    """Generate a library of base learners."""
    mlp=MLPRegressor()
    lin = LinearRegression()
    dt=DecisionTreeRegressor()
    sgd=SGDRegressor()
    hub=HuberRegressor()
    knn=KNeighborsRegressor()
    svm=SVR()
    rf = RandomForestRegressor(max_depth=4,random_state=SEED)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    ab = AdaBoostRegressor(random_state=SEED)
    xgb = XGBRegressor(objective='reg:linear',
                 n_estimators=1000,
                 min_child_weight=1,
                 learning_rate=0.01,
                 max_depth=5,
                 n_jobs=4,
                 subsample=0.6,
                 colsample_bytree=0.4,
                 colsample_bylevel=1)

    lgb = LGBMRegressor(n_estimators=1000)
    models = {
        # 'mlp':mlp,
        'linear': lin,
        # 'decision tree':dt,
        # 'sgd':sgd,
        # 'hub':hub,
        # 'knn':knn,
        # 'svm':svm,
        'random forest': rf,
        'gbm': gb,
        # 'ab': ab,
        'xgb': xgb,
        'lgb': lgb
        }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((y_test.shape[0], len(model_list)))
    x_sub = df_test.drop(['日期'], axis=1, inplace=False)
    # x_sub=scaler.transform(x_sub)
    P_sub = np.zeros((x_sub.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    P_sub = pd.DataFrame(P_sub)
    print("训练各个模型")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X_train, y_train)
        P.iloc[:, i] = m.predict(X_test)
        P_sub.iloc[:, i] = m.predict(x_sub)
        cols.append(name)
        print("done")
    P.columns = cols
    P_sub.columns = cols
    print("Done.\n")
    return P,P_sub


def score_models(P, y):
    """Score model in prediction DF"""
    print("评价每个模型.")
    for m in P.columns:
        score = mean_absolute_error(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")


def predict(P_sub):
    df_test['prediction'] = P_sub.mean(axis=1)
    df_test['time']=df_test['日期']
    df_test[['time', 'prediction']].to_csv('result/01_ensemble_regressor.csv', index=False)
    print("predictin done")


models = get_models()
P,P_sub = train_predict(models)
score_models(P, y_test)
print("Mean absolute error regression loss: %.3f" % mean_absolute_error(y_test, P.mean(axis=1)))
predict(P_sub)