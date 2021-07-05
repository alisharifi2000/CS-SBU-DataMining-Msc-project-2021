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
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor


def return_accuracy(y_pred, y):
    z = np.where((y * 1.1 > y_pred) & (y * 0.90 < y_pred), 1, 0)
    return np.count_nonzero(z == 1) / z.size


def mdy_to_ymd(d):
    return datetime.strptime(d, '%b %d, %Y').strftime('%Y-%m-%d')


df = pd.read_csv('bitcoin_data.csv')
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')


for i in range(1, 16):
    df['p'+str(i)] = df['Price'].shift(periods=-1 * i, fill_value=0)


df_values = df[['Price', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
                'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15']]


df_test = df.iloc[:550, :]
df_train = df.iloc[550:, :]

y_test = df_test[['Price']].to_numpy().reshape((len(df_test),))
y_tr = df_train[['Price']].to_numpy().reshape((len(df_train),))

x_test = df_test[['p1', 'p2', 'p3', 'p4', 'p5', 'p6',
                 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15']].to_numpy()
x_tr = df_train[['p1', 'p2', 'p3', 'p4', 'p5', 'p6',
                'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15']].to_numpy()


regr0 = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
regr0.fit(x_tr, y_tr)
predict_test = regr0.predict(x_test)
print('RidgeCV accuracy:', return_accuracy(predict_test, y_test))
print('RidgeCV RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr1 = RandomForestRegressor(max_depth=None, random_state=0, criterion='mse', min_samples_leaf=1)
regr1.fit(x_tr, y_tr)
predict_test = regr1.predict(x_test)
print('RandomForest with default parameters accuracy:', return_accuracy(predict_test, y_test))
print('RandomForest with default parameters RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr1 = RandomForestRegressor(max_depth=2, random_state=0, criterion='mse', min_samples_leaf=1)
regr1.fit(x_tr, y_tr)
predict_test = regr1.predict(x_test)
print('RandomForest with max_depth=2 accuracy:', return_accuracy(predict_test, y_test))
print('RandomForest with max_depth=2 RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr1 = RandomForestRegressor(max_depth=10, random_state=0, criterion='mse', min_samples_leaf=1)
regr1.fit(x_tr, y_tr)
predict_test = regr1.predict(x_test)
print('RandomForest with max_depth=10 accuracy:', return_accuracy(predict_test, y_test))
print('RandomForest with max_depth=10 RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr1 = RandomForestRegressor(max_depth=10, random_state=0, criterion='mae', min_samples_leaf=1)
regr1.fit(x_tr, y_tr)
predict_test = regr1.predict(x_test)
print('RandomForest with max_depth=10 and criterion=mae accuracy:', return_accuracy(predict_test, y_test))
print('RandomForest with max_depth=10 and criterion=mae RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr1 = RandomForestRegressor(max_depth=10, random_state=0, criterion='mse', min_samples_leaf=6)
regr1.fit(x_tr, y_tr)
predict_test = regr1.predict(x_test)
print('RandomForest with max_depth=10 and criterion=mse and min_samples_leaf=6 accuracy:', return_accuracy(predict_test, y_test))
print('RandomForest with max_depth=10 and criterion=mse and min_samples_leaf=6 RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr2 = DecisionTreeRegressor(max_depth=10)
regr2.fit(x_tr, y_tr)
predict_test = regr2.predict(x_test)
print('DecisionTree accuracy:', return_accuracy(predict_test, y_test))
print('DecisionTree RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr3 = SVR(kernel='rbf')
regr3.fit(x_tr, y_tr)
predict_test = regr3.predict(x_test)
print('SVR accuracy:', return_accuracy(predict_test, y_test))
print('SVR RMSE:', mean_squared_error(y_test, predict_test, squared=False))

regr4 = AdaBoostRegressor(random_state=0, learning_rate=1, n_estimators=100, loss='linear')
regr4.fit(x_tr, y_tr)
predict_test = regr4.predict(x_test)
print('AdaBoost with default parameters accuracy:', return_accuracy(predict_test, y_test))
print('AdaBoost with default parameters RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr5 = AdaBoostRegressor(random_state=0, learning_rate=0.1, n_estimators=100, loss='linear')
regr5.fit(x_tr, y_tr)
predict_test = regr5.predict(x_test)
print('AdaBoost with learning rate 0.1 accuracy:', return_accuracy(predict_test, y_test))
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr6 = AdaBoostRegressor(random_state=0, learning_rate=1, n_estimators=100, loss='square')
regr6.fit(x_tr, y_tr)
predict_test = regr6.predict(x_test)
print('AdaBoost with loss=square accuracy:', return_accuracy(predict_test, y_test))
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr7 = AdaBoostRegressor(random_state=0, learning_rate=0.1, n_estimators=500, loss='square')
regr7.fit(x_tr, y_tr)
predict_test = regr7.predict(x_test)
print('AdaBoost with loss=square and n_estimators = 500 and learning_rate=0.1 accuracy:', return_accuracy(predict_test, y_test))
print('RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr8 = linear_model.BayesianRidge()
regr8.fit(x_tr, y_tr)
predict_test = regr8.predict(x_test)
print('BayesianRidge accuracy:', return_accuracy(predict_test, y_test))
print('BayesianRidge RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr9 = MLPRegressor(random_state=1, max_iter=1500)
regr9.fit(x_tr, y_tr)
predict_test = regr9.predict(x_test)
print('MLPRegressor accuracy:', return_accuracy(predict_test, y_test))
print('MLPRegressor RMSE:', mean_squared_error(y_test, predict_test, squared=False))


regr10 = LinearRegression()
regr10.fit(x_tr, y_tr)
predict_test = regr10.predict(x_test)
print('LinearRegression accuracy:', return_accuracy(predict_test, y_test))
print('LinearRegression RMSE:', mean_squared_error(y_test, predict_test, squared=False))


### boosting

regr11 = GradientBoostingRegressor(random_state=0)
regr11.fit(x_tr, y_tr)
predict_test = regr11.predict(x_test)
print('XGBoost accuracy:', return_accuracy(predict_test, y_test))
print('XGBoost RMSE:', mean_squared_error(y_test, predict_test, squared=False))



# best 4 -> BayesianRidge, LinearRegression, MLPRegressor, XGBoost

## voting

bay_reg = linear_model.BayesianRidge()
lin_reg = LinearRegression()
MLP_reg = MLPRegressor(random_state=1, max_iter=1500)
XG_reg = GradientBoostingRegressor(random_state=0)

vot_reg = VotingRegressor([('bay_reg', bay_reg), ('lin_reg', lin_reg),
                      ('MLP_reg', MLP_reg), ('XG_reg', XG_reg)])
vot_reg.fit(x_tr, y_tr)
predict_test = vot_reg.predict(x_test)
print('voting accuracy:', return_accuracy(predict_test, y_test))
print('voting RMSE:', mean_squared_error(y_test, predict_test, squared=False))


### bagging

bays_bagging_reg = BaggingRegressor(base_estimator=bay_reg,
                                    n_estimators=15, random_state=0)
bays_bagging_reg.fit(x_tr, y_tr)
predict_test = bays_bagging_reg.predict(x_test)
print('BayesianRidge bagging accuracy:', return_accuracy(predict_test, y_test))
print('BayesianRidge bagging RMSE:', mean_squared_error(y_test, predict_test, squared=False))


bays_bagging_reg = BaggingRegressor(base_estimator=lin_reg,
                                    n_estimators=20, random_state=0)
bays_bagging_reg.fit(x_tr, y_tr)
predict_test = bays_bagging_reg.predict(x_test)
print('LinearRegression bagging accuracy:', return_accuracy(predict_test, y_test))
print('LinearRegression bagging RMSE:', mean_squared_error(y_test, predict_test, squared=False))


