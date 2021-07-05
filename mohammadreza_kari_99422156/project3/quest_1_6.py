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
df = pd.read_csv('train.csv')
from sklearn import svm
from sklearn.metrics import accuracy_score


def data_split(dataframe, runFast = False):
    train, test = train_test_split(dataframe, test_size=0.2)
    if runFast:
        train, test = train_test_split(dataframe, test_size=0.8)

    x_tr = train.drop(columns=['price_range'])
    x_test = test.drop(columns=['price_range'])

    y_tr = train[['price_range']]
    y_test = test[['price_range']]

    y_tr = y_tr.to_numpy().reshape((len(y_tr),))
    y_test = y_test.to_numpy().reshape((len(y_test),))

    x_tr = x_tr.to_numpy()
    x_test = x_test.to_numpy()

    return y_tr, y_test, x_tr, x_test

# question 2


y_tr, y_test, x_tr, x_test = data_split(df)


clf = svm.SVC(gamma='auto', kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = linear, test score:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


clf = SVC(gamma='auto', kernel='rbf')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = rbf, test score:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


clf = SVC(gamma='auto', kernel='poly')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = poly and degree = 3, test score:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


clf = SVC(gamma='auto', kernel='poly', degree=5)
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = poly and degree = 5, test score:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


clf = SVC(gamma='auto', kernel='poly', degree=2)
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = poly and degree = 2, test accuracy:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


#### regression for svm

clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))
clf.fit(x_tr, y_tr)
print("kernel type = linear, and regression test accuracy:", clf.score(x_test, y_test))

### sigmoid

clf = SVC(gamma='auto', kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = sigmoid, C = 4, test accuracy:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


## hard margin
clf = SVC(gamma='auto', kernel='linear', C=4)
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = lin, C = 4, test accuracy:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))

## soft margin
clf = SVC(gamma='auto', kernel='linear', C=0.1)
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = lin, C = 0.1, test accuracy:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


#### binning battery power

df['battery_power'] = pd.cut(df[['battery_power']].to_numpy().reshape(len(df)), 4, labels=[1, 2, 3, 4]).astype('float')
print(df['battery_power'].head())
y_tr, y_test, x_tr, x_test = data_split(df, )

clf = SVC(gamma='auto', kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = linear and battery power is binned , test accuracy:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))

#### one hot encoding

bat_encode = pd.get_dummies(df.battery_power, prefix='battery_power')

df = df.drop('battery_power', axis=1)

df = df.join(bat_encode)


n_cores_encode = pd.get_dummies(df.n_cores, prefix='n_cores')

df = df.drop('n_cores', axis=1)

df = df.join(n_cores_encode)


fc_encode = pd.get_dummies(df.fc, prefix='fc')

df = df.drop('fc', axis=1)

df = df.join(fc_encode)

y_tr, y_test, x_tr, x_test = data_split(df)

clf = SVC(gamma='auto', kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = linear and one hot encoding for categorical features , test accuracy:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


#### new feature area

df['area'] = df['px_height'] * df['px_width']
df['area'] = (df['area']-df['area'].mean())/df['area'].std()
y_tr, y_test, x_tr, x_test = data_split(df)

clf = SVC(kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = linear with new feature named area , test accuracy:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))

#### log transform

df_log = copy.deepcopy(df)

df_log['mobile_wt'] = np.log(df_log['mobile_wt']+1)
df_log['px_height'] = np.log(df_log['px_height']+1)
df_log['px_width'] = np.log(df_log['px_width']+1)
df_log['ram'] = np.log(df_log['ram']+1)

y_tr, y_test, x_tr, x_test = data_split(df_log)

clf = SVC(gamma='auto', kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = linear and log transformed data:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))

#### normalize data


normalized_df = copy.deepcopy(df)

normalized_df['mobile_wt'] = (df['mobile_wt']-df['mobile_wt'].mean())/df['mobile_wt'].std()
normalized_df['px_height'] = (df['px_height']-df['px_height'].mean())/df['px_height'].std()
normalized_df['px_width'] = (df['px_width']-df['px_width'].mean())/df['px_width'].std()
normalized_df['ram'] = (df['ram']-df['ram'].mean())/df['ram'].std()


y_tr, y_test, x_tr, x_test = data_split(normalized_df)

clf = SVC(gamma='auto', kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = linear and normalized data:", clf.score(x_test, y_test))
print("test accuracy:", accuracy_score(predict, y_test))


#### all together


clf = SVC(gamma='auto', kernel='linear')
clf.fit(x_tr, y_tr)
predict = clf.predict(x_test)
print("kernel type = linear all one hot encoding and others combined together")
print("test accuracy:", accuracy_score(predict, y_test))

