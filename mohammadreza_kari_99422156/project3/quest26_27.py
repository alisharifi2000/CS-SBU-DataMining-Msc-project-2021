import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor


pd.set_option('display.max_columns', None)
df = pd.read_csv('datas/training_data.csv')

df = df.drop(['id', 'era', 'data_type'], axis=1)


def data_split(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2)

    x_tr = train.drop(columns=['target'])
    x_test = test.drop(columns=['target'])

    y_tr = train[['target']]
    y_test = test[['target']]

    y_tr = y_tr.to_numpy().reshape((len(y_tr),))
    y_test = y_test.to_numpy().reshape((len(y_test),))

    x_tr = x_tr.to_numpy()
    x_test = x_test.to_numpy()

    return y_tr, y_test, x_tr, x_test

# question 2


def return_accuracy(y_pred, y):
    z = np.where((y * 1.1 > y_pred) & (y * 0.90 < y_pred), 1, 0)
    return np.count_nonzero(z == 1) / z.size


y_tr, y_test, x_tr, x_test = data_split(df)
print("data loaded")

reg = linear_model.BayesianRidge()
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('BayesianRidge accuracy:', return_accuracy(predict_test, y_test))
print('BayesianRidge RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')


reg = LinearRegression()
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('LinearRegression accuracy:', return_accuracy(predict_test, y_test))
print('LinearRegression RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')


regr = DecisionTreeRegressor(max_depth=8)
regr.fit(x_tr, y_tr)
predict_test = regr.predict(x_test)
print('DecisionTree accuracy:', return_accuracy(predict_test, y_test))
print('DecisionTree RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')


#### voting

reg_dt = DecisionTreeRegressor(max_depth=8)
reg_lin = LinearRegression()
reg_bays = linear_model.BayesianRidge()

reg = VotingRegressor([('reg_dt', reg_dt), ('reg_lin', reg_lin),
                      ('reg_bays', reg_bays)])
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('voting accuracy with all 3 models:', return_accuracy(predict_test, y_test))
print('voting RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')

## bagging

reg = BaggingRegressor(base_estimator=reg_bays,
                       n_estimators=3, random_state=0)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('BayesianRidge bagging with 3 estimator accuracy:', return_accuracy(predict_test, y_test))
print('BayesianRidge bagging RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')


## bagging

reg = BaggingRegressor(base_estimator=reg_lin,
                       n_estimators=4, random_state=0)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('linear regression bagging with 4 estimator accuracy:', return_accuracy(predict_test, y_test))
print('linear regression bagging RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')


### bagging


reg = BaggingRegressor(base_estimator=reg_lin,
                       n_estimators=4, random_state=0)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('linear regression bagging with 4 estimator accuracy:', return_accuracy(predict_test, y_test))
print('linear regression bagging RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')



### bagging


reg = BaggingRegressor(base_estimator=reg_dt,
                       n_estimators=5, random_state=0)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('decision tree bagging with 5 estimator accuracy:', return_accuracy(predict_test, y_test))
print('decision tree bagging RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')

### bagging combined with voting

vot_reg = VotingRegressor([('reg_lin', reg_lin), ('reg_bays', reg_bays)])

reg = BaggingRegressor(base_estimator=vot_reg,
                       n_estimators=5, random_state=0)
reg.fit(x_tr, y_tr)
predict_test = reg.predict(x_test)
print('voting and bagging combined accuracy:', return_accuracy(predict_test, y_test))
print('voting and bagging combined RMSE:', mean_squared_error(y_test, predict_test, squared=False))
print('score:', reg.score(x_test, y_test))
print('spearman correlation:', stats.spearmanr(predict_test, y_test), '\n')

