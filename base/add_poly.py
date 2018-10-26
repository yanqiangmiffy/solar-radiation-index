# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


test = pd.read_csv('../input/test_feature.csv')
# print(test)
train = pd.read_csv('../input/train_feature.csv')

label = pd.read_csv('../input/train_label.csv')
scaler=StandardScaler()

def to_one_day(data_df):
    train = data_df
    train['气压'] = train['气压'].apply(lambda x: 69000 if x < 60000 else x)

    train_date = train[['日期']].drop_duplicates().reset_index(drop=True)
    for i in range(1, train['日期'].max()+1):
        # print(i)
        data = train[train['日期'] == i].reset_index(drop=True)

        # train_date.loc[i - 1, '2_辐照度'] = data.loc[0, '辐照度']
        train_date.loc[i - 1, '2_风速'] = data.loc[0, '风速']
        train_date.loc[i - 1, '2_风向'] = data.loc[0, '风向']
        train_date.loc[i - 1, '2_温度'] = data.loc[0, '温度']
        train_date.loc[i - 1, '2_湿度'] = data.loc[0, '湿度']
        train_date.loc[i - 1, '2_气压'] = data.loc[0, '气压']

        # train_date.loc[i - 1, '5_辐照度'] = data.loc[1, '辐照度']
        train_date.loc[i - 1, '5_风速'] = data.loc[1, '风速']
        train_date.loc[i - 1, '5_风向'] = data.loc[1, '风向']
        train_date.loc[i - 1, '5_温度'] = data.loc[1, '温度']
        train_date.loc[i - 1, '5_湿度'] = data.loc[1, '湿度']
        train_date.loc[i - 1, '5_气压'] = data.loc[1, '气压']

        train_date.loc[i - 1, '8_辐照度'] = data.loc[2, '辐照度']
        train_date.loc[i - 1, '8_风速'] = data.loc[2, '风速']
        train_date.loc[i - 1, '8_风向'] = data.loc[2, '风向']
        train_date.loc[i - 1, '8_温度'] = data.loc[2, '温度']
        train_date.loc[i - 1, '8_湿度'] = data.loc[2, '湿度']
        train_date.loc[i - 1, '8_气压'] = data.loc[2, '气压']

        train_date.loc[i - 1, '11_辐照度'] = data.loc[3, '辐照度']
        train_date.loc[i - 1, '11_风速'] = data.loc[3, '风速']
        train_date.loc[i - 1, '11_风向'] = data.loc[3, '风向']
        train_date.loc[i - 1, '11_温度'] = data.loc[3, '温度']
        train_date.loc[i - 1, '11_湿度'] = data.loc[3, '湿度']
        train_date.loc[i - 1, '11_气压'] = data.loc[3, '气压']

        train_date.loc[i - 1, '14_辐照度'] = data.loc[4, '辐照度']
        train_date.loc[i - 1, '14_风速'] = data.loc[4, '风速']
        train_date.loc[i - 1, '14_风向'] = data.loc[4, '风向']
        train_date.loc[i - 1, '14_温度'] = data.loc[4, '温度']
        train_date.loc[i - 1, '14_湿度'] = data.loc[4, '湿度']
        train_date.loc[i - 1, '14_气压'] = data.loc[4, '气压']

        train_date.loc[i - 1, '17_辐照度'] = data.loc[5, '辐照度']
        train_date.loc[i - 1, '17_风速'] = data.loc[5, '风速']
        train_date.loc[i - 1, '17_风向'] = data.loc[5, '风向']
        train_date.loc[i - 1, '17_温度'] = data.loc[5, '温度']
        train_date.loc[i - 1, '17_湿度'] = data.loc[5, '湿度']
        train_date.loc[i - 1, '17_气压'] = data.loc[5, '气压']

        train_date.loc[i - 1, '20_辐照度'] = data.loc[6, '辐照度']
        train_date.loc[i - 1, '20_风速'] = data.loc[6, '风速']
        train_date.loc[i - 1, '20_风向'] = data.loc[6, '风向']
        train_date.loc[i - 1, '20_温度'] = data.loc[6, '温度']
        train_date.loc[i - 1, '20_湿度'] = data.loc[6, '湿度']
        train_date.loc[i - 1, '20_气压'] = data.loc[6, '气压']

        # train_date.loc[i - 1, '23_辐照度'] = data.loc[7, '辐照度']
        train_date.loc[i - 1, '23_风速'] = data.loc[7, '风速']
        train_date.loc[i - 1, '23_风向'] = data.loc[7, '风向']
        train_date.loc[i - 1, '23_温度'] = data.loc[7, '温度']
        train_date.loc[i - 1, '23_湿度'] = data.loc[7, '湿度']
        train_date.loc[i - 1, '23_气压'] = data.loc[7, '气压']

    # print(train_date)
    return train_date

# def get_old():

def add_poly_features(data,column_names):
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features

def creat_fea(data_df):
    train = data_df

    # print(train)
    train = to_one_day(train)
    train['风速_mean'] = train[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].mean(axis=1)
    train['风速_std'] = train[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].std(axis=1)
    train['风速_min'] = train[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].min(axis=1)
    train['风速_max'] = train[['2_风速', '5_风速', '8_风速', '11_风速', '14_风速', '17_风速', '20_风速', '23_风速']].max(axis=1)

    train['辐照度_mean'] = train[[ '8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].mean(
        axis=1)
    train['辐照度_std'] = train[['8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].std(
        axis=1)
    train['辐照度_min'] = train[['8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].min(
        axis=1)
    train['辐照度_max'] = train[[ '8_辐照度', '11_辐照度', '14_辐照度', '17_辐照度', '20_辐照度']].max(
        axis=1)

    train['风向_mean'] = train[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].mean(axis=1)
    train['风向_std'] = train[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].std(axis=1)
    train['风向_min'] = train[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].min(axis=1)
    train['风向_max'] = train[['2_风向', '5_风向', '8_风向', '11_风向', '14_风向', '17_风向', '20_风向', '23_风向']].max(axis=1)

    train['温度_mean'] = train[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].mean(axis=1)
    train['温度_std'] = train[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].std(axis=1)
    train['温度_min'] = train[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].min(axis=1)
    train['温度_max'] = train[['2_温度', '5_温度', '8_温度', '11_温度', '14_温度', '17_温度', '20_温度', '23_温度']].max(axis=1)

    train['湿度_mean'] = train[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].mean(axis=1)
    train['湿度_std'] = train[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].std(axis=1)
    train['湿度_min'] = train[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].min(axis=1)
    train['湿度_max'] = train[['2_湿度', '5_湿度', '8_湿度', '11_湿度', '14_湿度', '17_湿度', '20_湿度', '23_湿度']].max(axis=1)

    train['气压_mean'] = train[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].mean(axis=1)
    train['气压_std'] = train[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].std(axis=1)
    train['气压_min'] = train[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].min(axis=1)
    train['气压_max'] = train[['2_气压', '5_气压', '8_气压', '11_气压', '14_气压', '17_气压', '20_气压', '23_气压']].max(axis=1)

    co = ['风速_mean', '辐照度_mean', '风向_mean', '温度_mean', '湿度_mean', '气压_mean']
    train = add_poly_features(train, co)


    train['mean_fsfx'] = train['风速_mean'] * train['风向_mean']
    train['mean_sdqy'] = train['气压_mean'] * train['湿度_mean']
    train['mean_fswd'] = train['风速_mean'] * train['温度_mean']
    train['mean_fswd_add'] = train['风速_mean'] + train['温度_mean']

    # 16916722606765264   16907986019043922  16906963592903954 16901332152913065  16885640983206002


    for i in range(0, len(train)):
        if i <1:
            train.loc[i, 'old_辐照度_mean'] = train.loc[i, '辐照度_mean']
        elif i < 2:
            train.loc[i, 'old_辐照度_mean'] = (train.loc[i, '辐照度_mean'] + train.loc[i-1, '辐照度_mean'])/2
        else:
            train.loc[i, 'old_辐照度_mean'] = (train.loc[i, '辐照度_mean'] + train.loc[i - 1, '辐照度_mean']+ train.loc[i - 2, '辐照度_mean']) / 3
    return train

def get_result(train_df, test_df):
    train = train_df
    train = creat_fea(train)

    test = test_df
    test = creat_fea(test)

    sub = test[['日期']]

    del train['日期']
    del test['日期']

    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)

    # X = train.values
    X = train
    y = label['电场实际太阳辐射指数'].values
    # re_test = test.values
    re_test = test
    mean_re=[]

    k_fold = KFold(n_splits=5, shuffle=True, random_state=50)
    for index, (train_index, test_index) in enumerate(k_fold.split(X)):
        # print(index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = xgb.XGBRegressor(objective='reg:linear', n_estimators=1000, min_child_weight=1,
                                 learning_rate=0.01, max_depth=5, n_jobs=4,
                                 subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1)

        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_test, y_test)],
                  early_stopping_rounds=30, eval_metric=['mae'],verbose=2)

        vali_pre = model.predict(X_test)
        score = mean_absolute_error(y_test, vali_pre)
        mean_re.append(score)

        pred_result = model.predict(re_test)
        sub['prediction'] = pred_result
        if index == 0:
            re_sub = sub
        else:
            re_sub = re_sub+sub
    re_sub = re_sub/5
    re_sub = re_sub.rename(columns={'日期':'time'})
    # print(re_sub)
    print('score list:',mean_re)
    print(np.mean(mean_re))
    return re_sub



sub_1 = get_result(train, test)

sub_1.to_csv('../result/xgb_poly_scale.csv', index=False)



