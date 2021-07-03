import pandas as pd
import numpy as np
from load_data import cal_acc, return_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

y_test, y_tr, x_test, x_tr = return_dataset()

reg = RandomForestRegressor(max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))

reg = RandomForestRegressor(max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))

reg = RandomForestRegressor(max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))

reg = RandomForestRegressor(max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('-----------------------------')
print('parameters:')
print("max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1")
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('accuracy:', cal_acc(predict_test, y_test))
