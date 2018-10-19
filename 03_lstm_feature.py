# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 03_lstm_feature.py 
@Time: 2018/10/19 17:58
@Software: PyCharm 
@Description:
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from feature import load_feature


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

df_train, df_test, y = load_feature()
SEED=222
stand_scaler=StandardScaler()
mm_scaler=MinMaxScaler()


def get_train_test(df_train,y):
    X=df_train.drop(['日期'], axis=1, inplace=False)
    X=stand_scaler.fit_transform(X)
    X=mm_scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_test, y_train, y_test=get_train_test(df_train,y)
X_train=X_train.reshape(X_train.shape[0],1, X_train.shape[1])
X_test=X_test.reshape(X_test.shape[0],1, X_test.shape[1])

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# 评估模型

# make a prediction
yhat = model.predict(X_test)
mse = mean_absolute_error(y_test, yhat)
print('Test MAE: %.3f' % mse)


# 预测模型
x_sub = df_test.drop(['日期'], axis=1, inplace=False)
x_sub=np.array(x_sub.values)

x_sub=stand_scaler.transform(x_sub)
x_sub=mm_scaler.transform(x_sub)

x_sub=x_sub.reshape(x_sub.shape[0],1, x_sub.shape[1])
y_pred=model.predict(x_sub)
df_test['prediction'] =y_pred
df_test['time']=df_test['日期']
df_test[['time', 'prediction']].to_csv('result/03_lstm_feature.csv', index=False)
