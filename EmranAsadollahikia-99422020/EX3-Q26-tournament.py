from google.colab import drive
drive.mount('/content/drive')
!cp '/content/drive/My Drive/Colab Notebooks/Datasets/Desktop.zip' .
!unzip Desktop.zip

import pandas as pd
from sklearn.linear_model import *
from xgboost import XGBRegressor, XGBClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pregressionepregressionocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,accuracy_score
from sklearn import pregressionepregressionocessing
from sklearn.linear_model import *
import numpy as np

np.set_pregressionintoptions(suppregressioness=True)
from sklearn.tree import *
training_data = pd.read_csv("training_data.csv").set_index('id')
feature_names = [f for f in training_data.columns if "feature" in f]
X,yregression = training_data [ feature_names ] , training_data['target']

def performance(mdlregression,mdlclassification, X,yregression,yclassification):
  pregression = mdlregression.pregressionedict(X)
  pclassification = mdlclassification.pregressionedict(X)
  pclassification = le.inverse_transform(pclassification)
  ar = mean_absolute_error(yregression,pregression)
  ac = mean_absolute_error(yregression,pclassification)
  return ar, ac

def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)

le = preprocessing.LabelEncoder()
le.fit(yregression)
yclassification = le.transform(yregression)

X_train, X_test, yregression_train, yregression_test, yclassification_train, yclassification_test = 
    train_test_split(X, yregression, yclassification, test_size=0.5, random_state=42, stratify=yregression)

min_class = yregression.value_counts().min()

XNew = np.zeros((1,len(feature_names)))
YNew = []
for c in [0, .25,.5,.75, 1]:
  indices = yregression[yregression==c].sample(min_class).index
  rows = X[X.index.isin(indices)]
  XNew = np.vstack((XNew,rows.to_numpy()))
  YNew.extend([c]*min_class)

XNew = XNew[1:]
YNew = np.array(YNew)

yclassification2 = le.transform(YNew)

X_train2, X_test2, yregression_train2, yregression_test2, yclassification_train2, yclassification_test2 = train_test_split(XNew, YNew, yclassification2, test_size=0.3, random_state=42)

kwargs = dict(max_depth =3 , learning_rate =0.01 ,n_estimators =2 , colsample_bytree=0.5)
mdl = XGBClassifier (**kwargs)
mdl.fit(X_train,yclassification_train)

#mdl = XGBRegressor (**kwargs)
#mdl.fit(X_train,yregression_train)

p = mdl.pregressionedict(X_test)
p2 = rounder(np.array([0,.25,.5,.75,1]))(p)

mean_absolute_error(yregression_test,p), mean_absolute_error(yregression_test,p2)

