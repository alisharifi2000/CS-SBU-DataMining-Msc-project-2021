# -*- coding: utf-8 -*-
"""

@author: Sayeh

Spyder Editor
"""

#%% Imorting data and packaging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from numpy import linalg
import cvxopt
import cvxopt.solvers
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import csv
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
# Load libraries
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import math

df1 = pd.read_csv (r'C:\Users\Asus\Desktop\Data Mining\03_assignment\data\mobile price\train.csv')

#**************************WORKING WITH FIRST(mobile price) DATASET*************************
#%%Understanding data for the second dataset

#to get a better understanding of how our data looks
df1.shape
df1.info() #no null data here
df1.describe()
df1.head(10)
df1.columns
#duplicated datapoints 
df1.duplicated(keep=False).any() #no duplicates found

#deleting outliers

for cols in df1.columns:
    if df1[cols].dtype == 'int64' or df1[cols].dtype == 'float64':
        upper_range = df1[cols].mean() + 3 * df1[cols].std()
        lower_range = df1[cols].mean() - 3 * df1[cols].std()
        
        indexs = df1[(df1[cols] > upper_range) | (df1[cols] < lower_range)].index
        df1 = df1.drop(indexs)

#dropping not useful datapoints
df1 = df1.drop(df1[df1['px_height'] == 0.0].index)

#%%Understanding data for the second dataset for questions of bitcoin dataset 
df2 = pd.read_csv (r'C:\Users\Asus\Desktop\Data Mining\03_assignment\data\Bitcoin 16-25\Bitcoin_Historical_Data_Investing.com.csv')

#to get a better understanding of how our data looks
df2.shape
df2.info() #no null data here
df2.describe()
df2.head(10)
df2.columns
#duplicated datapoints 
df2.duplicated(keep=False).any() #no duplicates found
df2.isna()
#columns with null data 
df2.isna().sum()/len(df2) #no null data

#%% correlation
df1.corr().price_range.sort_values()

corr = df1.corr()
corr
f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corr, square = True ,annot = True)

#%% train and test split

y = df1["price_range"].values
x_data=df1.drop(["price_range"],axis=1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)

#%% svm classifier 1

svm=SVC(kernel='rbf',random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

#%% svm classifier 2 kernel sigmoid

svm=SVC(kernel='sigmoid',random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

#%% svm classifier 2 kernel poly deg3

svm=SVC(kernel='poly', degree=3,random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

#%% svm classifier 2 kernel poly deg 10

svm=SVC(kernel='poly', degree=10,random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

#%% svm classifier 2 kernel linear 

svm=SVC(kernel='linear',random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

#%% svm classifier kernel rbf and gamma Kernel coefficient set on scale

svm=SVC(kernel='rbf',random_state=None, gamma='scale')
svm.fit(x_train,y_train)

y_pred = svm.predict(x_test)

print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, pos_label='positive', average='micro'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, pos_label='positive', average='micro'))

#%% svm classifier kernel rbf and gamma Kernel coefficient set on auto

svm=SVC(kernel='rbf',random_state=None, gamma='auto')
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, pos_label='positive', average='micro'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, pos_label='positive', average='micro'))

#this part of the code is extra and for better understanding margins
#%% an example of soft margin hard margin and visialization for svm with plots
#I got this part of the code from https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
             
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(polynomial_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=1000.1)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

        
    #test_linear()
    #test_non_linear()
    #test_soft()

    


#%% for matgin soft and hard

svm=SVC(C=1.0,kernel='linear',random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

svm=SVC(C=10.0,kernel='linear',random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))


svm=SVC(C=100.0,kernel='linear',random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))


## FEATURE ENGINEERING
#%%binning on battery power

#****equal width binning*****
fig, ax = plt.subplots()
df1['battery_power'].hist(color='#A9C5D3', edgecolor='black',  
                          grid=False)
ax.set_title('battery powr histogram', fontsize=12)
ax.set_xlabel('battery_power', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
#bin1
df1['battery_power_bin1'] = np.array(np.floor(np.array(df1['battery_power']) / 1000.))
df1[['battery_power', 'battery_power_bin1']].iloc[1071:1076]
df1['battery_power_bin1'].value_counts()
#df1 = df1.drop(columns=['battery_power'])

#bin2
df1['battery_power_bin2'] = np.array(np.floor(np.array(df1['battery_power']) / 100.))
df1['battery_power_bin2']
df1[['battery_power', 'battery_power_bin2']].iloc[1071:1076]
df1['battery_power_bin2'].value_counts()
#*****bin3 addaptive******
fig, ax = plt.subplots()
df1['battery_power'].hist(bins=30, color='#A9C5D3', 
                             edgecolor='black', grid=False)
ax.set_title('battery powr histogram 2', fontsize=12)
ax.set_xlabel('battery_power', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

#quantiled histogram bin3
fig, ax = plt.subplots()
df1['battery_power'].hist(bins=30, color='#A9C5D3', 
                             edgecolor='black', grid=False)

quantile_list = [0, .25, .75, 1.]
quantiles = df1['battery_power'].quantile(quantile_list)
quantiles

for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)
ax.set_title('Battery power Histogram with Quantiles', 
             fontsize=12)
ax.set_xlabel('battery_power', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

quantile_labels = ['0-25Q', '25-75Q', '75-100Q']
df1['battery_power_bin3'] = pd.qcut(df1['battery_power'],q=quantile_list)


df1[['battery_power', 'battery_power_bin1' ,'battery_power_bin2','battery_power_bin3']].iloc[1071:1076]
#df1 = df1.drop(columns=['battery_power_bin3','battery_power_bin1' ,'battery_power_bin2'])

#%%one hot encoding and dummies

columns = ['blue','dual_sim','four_g','three_g','touch_screen','wifi']
df1["blue"]=df1["blue"].astype(str)
df1["dual_sim"]=df1["dual_sim"].astype(str)
df1["four_g"]=df1["four_g"].astype(str)
df1["three_g"]=df1["three_g"].astype(str)
df1["touch_screen"]=df1["touch_screen"].astype(str)
df1["wifi"]=df1["wifi"].astype(str)


dummies_feature = pd.get_dummies(df1[columns])
dummies_feature.head()
dummies_feature.shape
df1 = pd.concat([df1, dummies_feature], axis=1)
df1.head()
df1 = df1.drop(columns=columns)
df1.head()
df1.info()
df1.shape

#%% transformation methods

#distribution plot for all features
df1.hist(grid=False,figsize=(10, 6),bins=30)

#applying log transform method for px_ height
df1['px_height_log'] = np.log(df1["px_height"])
#applying square root transform method for px_ height
df1['px_height_sqrt'] = np.sqrt(df1["px_height"])
df1 = df1.drop(columns=['px_height'])

#px_height pre transformation
sns.set(rc={'figure.figsize':(15,10)})
sns.distplot(df1["px_height"],kde_kws={"label": 'px_height'}, bins=20)

#px_height log transformation
sns.set(rc={'figure.figsize':(15,10)})
sns.distplot(np.log(df1["px_height"]),kde_kws={"label": 'px_height_log'}, bins=20)

#px_height sqrt transformation
sns.set(rc={'figure.figsize':(15,10)})
sns.distplot(np.sqrt(df1["px_height"]),kde_kws={"label": 'px_height_sqrt'}, bins=20)

#%% adding a surface feature

df1['sc_surface'] = df1['sc_h'] * df1['sc_w']

#%%svm for the feature engineering part
# I ran each targeted cell for feature engineering and ran this svm model afterwards to get results
svm=SVC(kernel='rbf',random_state=None)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

print(classification_report(y_pred, y_test))

#%% Decision tree classifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
print("dt train accuracy:",dtc.score(x_train,y_train))
print("dt test accuracy:",dtc.score(x_test,y_test))

print(classification_report(y_pred, y_test))

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%% DT different parameters criterion

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train,y_train)
#y_pred = dtc.predict(x_test)

print("dt train accuracy:",dtc.score(x_train,y_train))
print("dt test accuracy:",dtc.score(x_test,y_test))

#%% DT different parameters splitter

dtc = DecisionTreeClassifier(splitter='random')
dtc.fit(x_train,y_train)
#y_pred = dtc.predict(x_test)

print("dt train accuracy:",dtc.score(x_train,y_train))
print("dt test accuracy:",dtc.score(x_test,y_test))

#%% DT different parameters min_samples_leaf

dtc = DecisionTreeClassifier(min_samples_leaf=10)
dtc.fit(x_train,y_train)
#y_pred = dtc.predict(x_test)

print("dt train accuracy:",dtc.score(x_train,y_train))
print("dt test accuracy:",dtc.score(x_test,y_test))

#%% DT different parameters max_depth

dtc = DecisionTreeClassifier(max_depth=10)
dtc.fit(x_train,y_train)
#y_pred = dtc.predict(x_test)

print("dt train accuracy:",dtc.score(x_train,y_train))
print("dt test accuracy:",dtc.score(x_test,y_test))

#%%pruning DT

 
dtc = DecisionTreeClassifier(ccp_alpha=0.1)
dtc.fit(x_train,y_train)
#y_pred = dtc.predict(x_test)

print("dt train accuracy:",dtc.score(x_train,y_train))
print("dt test accuracy:",dtc.score(x_test,y_test)) 

#%% Random forest classifier

rdf = RandomForestClassifier()
rdf.fit(x_train,y_train)
y_pred = rdf.predict(x_test)
print("rdf train accuracy:",rdf.score(x_train,y_train))
print("rdf test accuracy:",rdf.score(x_test,y_test))

print(classification_report(y_pred, y_test))









# -*- coding: utf-8 -*-
"""

@author: Sayeh

Spyder Editor
"""

#**************************WORKING WITH SECOND(Bitcoin) DATASET*************************
#%% For questions 16,17,19,20,22,23


def RMSE(y_test, y_pred, error_rate):

    # y_test_with_error_rate = y_test.copy()
    error_rate_list = np.array([
        y_test*(1-error_rate), y_test*(1+error_rate)
    ]).reshape(len(y_test), 2)

    indexes_1 = np.where(
        y_pred <= error_rate_list[:, 1]
    )[0]
    indexes_2 = np.where( 
        y_pred >= error_rate_list[:, 0]
    )[0]
    indexes = set(indexes_1).intersection(list(indexes_2))

    for index in indexes:
        y_pred[index] = y_test[index]
    
    rmse = np.sqrt(
        abs(
            np.average(y_test) - np.average(y_pred)
        )
    )
    return rmse
    
    



def q_16():
    def mdy_to_ymd(d):
        return datetime.strptime(d, '%b %d, %Y').strftime('%Y-%m-%d')

    with open('Bitcoin 16-25/Bitcoin.csv') as csv_file:
        file = csv.reader(csv_file, delimiter=',')
        file = list(file)
        file = np.array(file)

        for i in range(1, len(file)):
            for j in range(len(file[0])):
                if j == 0:
                    file[i, j] = mdy_to_ymd(file[i, j])
                elif j == 5:
                    if 'M' in file[i, j]:
                        file[i, j] = float(file[i, j].replace('%', '').replace('M', '')) * 10**6
                    if 'K' in file[i, j]:
                        file[i, j] = float(file[i, j].replace('%', '').replace('K', '')) * 10**3
                elif j == 6:
                    file[i, j] = float(file[i, j].replace('%', '').replace('%', ''))
                else:
                    file[i, j] = np.float64(file[i, j].replace(',', ''))
                    # file[i, j] = file[i, j].replace(',', '').astype(np.float)
                    # print(file[i, j])
                    # print(type(file[i, j]))
        file = np.flip(file, 0)

        while len(np.where(file == '-')[0]) != 0:
            index = np.where(file == '-')[0][0]
            file = np.delete(file, index, 0)

        train_data = None
        test_data = None
        index = np.where(file[:, 0] == '2020-01-01')[0][0]
        train_data = file[:index, 1:]
        test_data = file[index+1:-2, 1:]

    train_data = train_data.astype(np.float64)
    train_data = train_data.round(decimals=2)
    test_data = test_data.astype(np.float64)
    test_data = test_data.round(decimals=2)

    return train_data, test_data


def q_17(X_train, X_test, y_train, y_test):

    def lstm(X_train, X_test, y_train, y_test):

        X_train = y_train[1:]
        y_train = y_train[:-1]

        X_test = y_test[1:]
        y_test = y_test[:-1]
        
        X_test = np.reshape(X_test, (len(X_test), 1, 1))
        X_train = np.reshape(X_train, (len(X_train), 1, 1))

        # create and fit the LSTM network
        model = Sequential()
        look_back = len(X_train[0])
        model.add(LSTM(4, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=2)

        # make predictions
        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)

        # calculate root mean squared error
        # print(testPredict)
        trainScore = metrics.mean_squared_error(y_train, trainPredict[:,0], squared=False)
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = metrics.mean_squared_error(y_test, testPredict[:,0], squared=False)
        print('Test Score: %.2f RMSE' % (testScore))

    def arima(X_train, X_test, y_train, y_test):
        train, test = y_train, y_test
        history = [x for x in train]
        predictions = list()
        # walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        
        rmse = metrics.mean_squared_error(test, predictions, squared=False)
        print('Test RMSE: %.3f' % rmse)

    def XGboost(X_train, X_test, y_train, y_test):
        def xgboost_forecast(train, testX):
            # transform list into array
            train = np.asarray(train)
            # split into input and output columns
            trainX, trainy = train[:-1], train[-1]
            # fit model
            model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
            model.fit(trainX, trainy)
            # make a one-step prediction
            yhat = model.predict([testX])
            return yhat[0]

        predictions = list()
        # split dataset
        train, test = y_train, y_test
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)-1):
            # split test row into input and output columns
            testX, testy = test[i], test[i+1]
            # fit model on history and make a prediction
            yhat = xgboost_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        
        # estimate prediction error
        # error = mean_absolute_error(test[:, -1], predictions)
        rmse = metrics.mean_squared_error(test[:, -1], predictions, squared=False)
        print('Test RMSE: %.3f' % rmse)


    # lstm(X_train, X_test, y_train, y_test)
    # arima(X_train, X_test, y_train, y_test)
    # XGboost(X_train, X_test, y_train, y_test)





def q_19(X_train, X_test, y_train, y_test):

    print('19')
    y_train = y_train.astype('int')
    y_train.round(decimals=2)
    y_test = y_test.astype('int')
    y_test.round(decimals=2)


    result = {}
    for n_estimators in range(50, 1000, 300):
        result[n_estimators] = [[], []]
        for learning_rate in range(1, 4):
            # Create adaboost classifer object
            abc = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
            )
            # Train Adaboost Classifer
            model = abc.fit(X_train, y_train)

            #Predict the response for test dataset
            y_pred = model.predict(X_test)

            # Model Accuracy, how often is the classifier correct?
            rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
            result[n_estimators][0].append(rmse)
            
            print('---------------------------')
            print(f"Model RMSE : {rmse}")

            rmse = RMSE(y_test, y_pred, error_rate=0.05) 
            result[n_estimators][1].append(rmse)
            
            print(f"Model RMSE : {rmse}")

    colors = [
        'b--',
        'g--',
        'r--',
        'm--',
        'y--',
        'b*-',
        'g*-',
        'r*-',
        'm*-',
        'y*-',
    ]
    color_index = 0
    for key, value in result.items():
        label = f'n_estimators : {key}'
        plt.plot(value[0], colors[color_index], label=label)
        plt.legend()
        color_index += 1
    
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE')

    plt.figure()
    color_index = 0
    for key, value in result.items():
        label = f'n_estimators : {key}'
        plt.plot(value[1], colors[color_index], label=label)
        plt.legend()
        color_index += 1
    
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE with Error Rate')
    
    plt.show()


def q_20(X_train, X_test, y_train, y_test):

    print('20')
    y_train = y_train.astype('int')
    y_train.round(decimals=2)
    y_test = y_test.astype('int')
    y_test.round(decimals=2)


    result = [[], []]
    for n_estimators in range(20, 130, 20):
        # Create random forest classifer object
        abc = RandomForestClassifier(
            n_estimators=n_estimators,
        )
        # Train Adaboost Classifer
        model = abc.fit(X_train, y_train)

        #Predict the response for test dataset
        y_pred = model.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print('----------------------------')

        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        result[0].append(rmse)
        print(f"Model RMSE : {rmse}")
        
        rmse = RMSE(y_test, y_pred, error_rate=0.05) 
        result[1].append(rmse)
        print(f"Model RMSE : {rmse}")

    colors = [
        'b*-',
        'g--',
        'r--',
        'm--',
        'y--',
        'b*-',
        'g*-',
        'r*-',
        'm*-',
        'y*-',
    ]


    n_estimators_list = list(
        range(20, 140, 20)
    )

    color_index = 0
    plt.plot(n_estimators_list, result[0], colors[color_index])
    color_index += 1
    plt.xlabel(f'N estimators')
    plt.ylabel('RMSE')
    
    plt.figure()
    color_index = 0
    plt.plot(n_estimators_list, result[1], colors[color_index])
    color_index += 1
    plt.xlabel(f'N estimators')
    plt.ylabel('RMSE with Error Rate')
    
    plt.show()


def q_22(X_train, X_test, y_train, y_test):

    yp_test = []
    for i in range(len(y_test)-1):
        if y_test[i+1] >= y_test[i]:
            yp_test.append(1)
        else:
            yp_test.append(0)
    yp_test = np.array(yp_test)

    x_test = y_test.copy()[:-1]
    x_test = np.reshape(x_test, (len(x_test), 1, 1))
    
    # yp_test = np.reshape(yp_test, (len(yp_test), 1, 1))

    yp_train = []
    for i in range(len(y_train)-1):
        if y_train[i+1] >= y_train[i]:
            yp_train.append(1)
        else:
            yp_train.append(0)
    yp_train = np.array(yp_train)

    x_train = y_train.copy()[:-1]

    x_train = np.reshape(x_train, (len(x_train), 1, 1))
    
    # yp_train = np.reshape(yp_train, (len(yp_train), 1, 1))


    # create and fit the LSTM network
    model = Sequential()
    look_back = 1
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, yp_train, epochs=500, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)

    # invert predictions
    # trainPredict = scaler.inverse_transform(trainPredict)
    # yp_train = scaler.inverse_transform([yp_train])
    # testPredict = scaler.inverse_transform(testPredict)
    # yp_test = scaler.inverse_transform([yp_test])
    # calculate root mean squared error
    print(testPredict[:,0].shape)
    print(yp_test.shape)
    trainScore = metrics.mean_squared_error(yp_train, trainPredict[:,0], squared=False)
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = metrics.mean_squared_error(yp_test, testPredict[:,0], squared=False)
    print('Test Score: %.2f RMSE' % (testScore))



def q_23(X_train, X_test, y_train, y_test):

    yp_test = []
    for i in range(len(y_test)-1):
        if y_test[i+1] >= y_test[i]:
            yp_test.append(1)
        else:
            yp_test.append(0)
    yp_test = np.array(yp_test)

    X_test = X_test[:-1]
    y_test = y_test[:-1]
    print(X_test.shape)
    X_test = np.c_[X_test, yp_test]

    # x_test = y_test.copy()[:-1]
    X_test = np.reshape(X_test, (len(X_test), len(X_test[0]), 1))
    
    # yp_test = np.reshape(yp_test, (len(yp_test), 1, 1))

    yp_train = []
    for i in range(len(y_train)-1):
        if y_train[i+1] >= y_train[i]:
            yp_train.append(1)
        else:
            yp_train.append(0)
    yp_train = np.array(yp_train)
    X_train = X_train[:-1]
    y_train = y_train[:-1]
    X_train = np.c_[X_train, yp_train]

    # x_train = y_train.copy()[:-1]

    X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
    
    # yp_train = np.reshape(yp_train, (len(yp_train), 1, 1))


    # create and fit the LSTM network
    model = Sequential()
    look_back = len(X_train[0])
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    # calculate root mean squared error
    print(testPredict)
    trainScore = metrics.mean_squared_error(y_train, trainPredict[:,0], squared=False)
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = metrics.mean_squared_error(y_test, testPredict[:,0], squared=False)
    print('Test Score: %.2f RMSE' % (testScore))


#%% now lets investigate

if __name__ == "__main__":

    train_data, test_data = q_16()

    # q_17(
    #     X_train=train_data[:, 1:],
    #     X_test=test_data[:, 1:],
    #     y_train=train_data[:, 0],
    #     y_test=test_data[:, 0],
    # )

    # q_19(
    #     X_train=train_data[:, 1:],
    #     X_test=test_data[:, 1:],
    #     y_train=train_data[:, 0],
    #     y_test=test_data[:, 0],
    # )

    # q_20(
    #     X_train=train_data[:, 1:],
    #     X_test=test_data[:, 1:],
    #     y_train=train_data[:, 0],
    #     y_test=test_data[:, 0],
    # )

    # q_22(
    #     X_train=train_data[:, 1:],
    #     X_test=test_data[:, 1:],
    #     y_train=train_data[:, 0],
    #     y_test=test_data[:, 0],
    # )
    q_23(
        X_train=train_data[:, 1:],
        X_test=test_data[:, 1:],
        y_train=train_data[:, 0],
        y_test=test_data[:, 0],
    )




