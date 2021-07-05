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

df = pd.read_csv('train.csv')


def data_split(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2)

    x_tr = train.drop(columns=['price_range'])
    x_test = test.drop(columns=['price_range'])

    y_tr = train[['price_range']]
    y_test = test[['price_range']]

    y_tr = y_tr.to_numpy().reshape((len(y_tr),))
    y_test = y_test.to_numpy().reshape((len(y_test),))

    x_tr = x_tr.to_numpy()
    x_test = x_test.to_numpy()

    return y_tr, y_test, x_tr, x_test


y_tr, y_test, x_tr, x_test = data_split(df)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_tr, y_tr)
print("desicion tree test accuracy:", clf.score(x_test, y_test))

### set depth and samples leaf

clf = DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_leaf=10)
clf.fit(x_tr, y_tr)
print("desicion tree, max_depth = 10 and min_leaf = 10. test accuracy:", clf.score(x_test, y_test))


clf = DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_leaf=15)
clf.fit(x_tr, y_tr)
print("desicion tree, max_depth = 10 and min_leaf = 20. test accuracy:", clf.score(x_test, y_test))


clf = DecisionTreeClassifier(random_state=0, max_depth=15, min_samples_leaf=20)
clf.fit(x_tr, y_tr)
print("desicion tree, max_depth = 15 and min_leaf = 20. test accuracy:", clf.score(x_test, y_test))



clf = DecisionTreeClassifier(random_state=0, max_depth=3)
clf.fit(x_tr, y_tr)
print("desicion tree max depth = 3. test accuracy:", clf.score(x_test, y_test))

### pruning

clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=40)
clf.fit(x_tr, y_tr)
print("desicion tree purning test accuracy:", clf.score(x_test, y_test))


### random forest

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(x_tr, y_tr)
print("random forest test accuracy:", clf.score(x_test, y_test))

