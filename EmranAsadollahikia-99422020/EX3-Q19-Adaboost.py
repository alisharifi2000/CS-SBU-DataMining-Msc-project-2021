import pandas as pd
import numpy as np
from load_data import cal_acc, return_dataset
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

y_test, y_tr, x_test, x_tr = return_dataset()

reg = AdaBoostRegressor(random_state=0, learning_rate=1, n_estimators=100, loss='linear')
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("random_state=0, learning_rate=1, n_estimators=100, loss='linear'")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))


reg = AdaBoostRegressor(random_state=0, learning_rate=1, n_estimators=100, loss='linear')
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("random_state=0, learning_rate=1, n_estimators=100, loss='linear'")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))


reg = AdaBoostRegressor(random_state=0, learning_rate=1, n_estimators=100, loss='linear')
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("random_state=0, learning_rate=1, n_estimators=100, loss='linear'")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))


reg = AdaBoostRegressor(random_state=0, learning_rate=1, n_estimators=100, loss='linear')
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("random_state=0, learning_rate=1, n_estimators=100, loss='linear'")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))
