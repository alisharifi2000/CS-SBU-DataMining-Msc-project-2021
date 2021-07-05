!pip install pystan==2.19.1.1
!pip install prophet

!pip install kaggle
!pip install scikit-learn --upgrade
import glob
import zipfile
import pandas as pd
from google.colab import files
from sklearn.model_selection import train_test_split
 
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
 
!kaggle datasets download -d iabhishekofficial/mobile-price-classification
 
file = glob.glob('*.zip')[0]
with zipfile.ZipFile(file, 'r') as zip_ref:
  zip_ref.extractall('files')

import scipy
import warnings
import itertools
import numpy as np
import scipy.stats
import scipy.linalg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import *
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import *
sns.set()
warnings.filterwarnings('ignore')

def onehot_encode(df, column):    
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column, dtype=bool)
    df = pd.concat([dummies, df], axis=1)
    df = df.drop([column], axis=1)   
    return df 
  
def network(df):
  df['network'] = None
  df.loc[(df['three_g']==0) & (df['four_g']==0),'network'] = 0
  df.loc[(df['three_g']==1) & (df['four_g']==0),'network'] = 1
  df.loc[(df['three_g']==1) & (df['four_g']==1),'network'] = 2
  df.drop(['three_g','four_g'],axis=1, inplace=True)

def preprocess(df):
  if 'id' in list(df):
    df.drop('id', inplace=True,axis=1)
  df = df[df.px_height>0]
  df = df[df.px_width>0]
  # df = df[df.sc_h>0]
  # df = df[df.sc_w>0]
  # df.drop(['m_dep','mobile_wt'], inplace=True,axis=1)
  
  network(df)
  # df['total_cpu'] = df['clock_speed'] * df['n_cores']
  # df['total_cpu'] = pd.cut(df['total_cpu'], 3, labels=False)
  df['lcd'] = (np.sqrt(df['sc_h']**2+df['sc_w']**2)*.3937).round(2)
  df['ppi'] = (np.sqrt(df['px_height']**2+df['px_width']**2) / df['lcd']).round()
  df['aspect_ratio'] = (df['px_width']/df['px_height']).round(2)
  
  # df['n_cam'] = None
  # df.loc[(df['fc']==0) & (df['pc']==0),'n_cam'] = 0
  # df.loc[(df['fc']>0) & (df['pc']==0),'n_cam'] = 1
  # df.loc[(df['fc']==0) & (df['pc']>0),'n_cam'] = 1
  # df.loc[(df['fc']>0) & (df['pc']>0),'n_cam'] = 2
  
  # df['cpu'] = pd.cut(df['clock_speed'], 5, labels=False)
  # df.drop('clock_speed',axis=1, inplace=True)
  
  df['battery'] = pd.cut(df['battery_power'], 5, labels=False)
  # df.drop('battery_power',axis=1, inplace=True)
  

  # df['mem'] = pd.cut(df['ram'], 10, labels=False)
  # df.drop('ram',axis=1, inplace=True)

  # for col in ['blue','dual_sim','touch_screen','wifi']:
  #   df = onehot_encode(df, col)
  
  # df['battery_power'] = df['battery_power'].round(-1)
  # df['int_memory'] = df['int_memory'].round(-1)
  
  cols = list(df)
  if 'price_range' in cols:
    cols.remove('price_range')
    df = df[cols + ['price_range']]
  return df

train = pd.read_csv('files/train.csv')
test = pd.read_csv('files/test.csv')

train = preprocess(train)
test = preprocess(test)

y = train.iloc[:,-1].to_numpy()
X = train.iloc[:,:-1].to_numpy()

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

parameters = [{'svc__kernel': ['rbf'], 'svc__gamma': [1e-1, 1e-3, 'scale'], 'svc__C': [ 1, 100, 1000]},
              {'svc__kernel': ['linear'], 'svc__C': [1, 100, 1000]},
              {'svc__kernel': ['poly'], 'svc__degree':[2, 3], 'svc__C': [1, 100, 1000]}
              ]

pipe = Pipeline(steps=[('scale', StandardScaler()), ('svc', SVC(class_weight='balanced'))])

clf = GridSearchCV(pipe, parameters,return_train_score=True,verbose=2)
clf.fit(X, y)

df = pd.DataFrame(clf.cv_results_)
df.drop(['mean_fit_time','std_fit_time','mean_score_time','std_score_time','split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'std_test_score', 'rank_test_score', 'split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score', 'std_train_score','params'],axis=1, inplace=True)
df.columns = ['C', 'Gamma (RBF)', 'Kernel', 'Degree(Poly)', 'Test Accuracy', 'Train Accuracy']
df = df[['Kernel', 'C', 'Gamma (RBF)', 'Degree(Poly)', 'Test Accuracy', 'Train Accuracy']]
df.fillna('-',inplace=True)
df['Test Accuracy'] = 100 * df['Test Accuracy'].round(4)
df['Train Accuracy'] = 100 * df['Train Accuracy'].round(4)
print(df.to_csv('svm-result.csv',index=False))

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import tree
parameters = {'dc__criterion': ['gini','entropy'], 'dc__splitter':['random', 'best'], 'dc__max_depth':[5,10,15,20]}

pipe = Pipeline(steps=[('scale', StandardScaler()), ('dc', tree.DecisionTreeClassifier())])

clf = GridSearchCV(pipe, parameters,return_train_score=True,verbose=2)
clf.fit(X, y)

clf.cv_results_['mean_test_score']

df = pd.DataFrame(clf.cv_results_)
df.drop(['mean_fit_time','std_fit_time','mean_score_time','std_score_time','split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'std_test_score', 'rank_test_score', 'split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score', 'std_train_score','params'],axis=1, inplace=True)
df.columns = ['Criterion', 'Max depth' ,'Splitter', 'Test Accuracy', 'Train Accuracy']
# df = df[['Kernel', 'C', 'Gamma (RBF)', 'Degree(Poly)', 'Test Accuracy', 'Train Accuracy']]
# df.fillna('-',inplace=True)
df['Test Accuracy'] = 100 * df['Test Accuracy'].round(4)
df['Train Accuracy'] = 100 * df['Train Accuracy'].round(4)
# df
print(df.to_csv('dc-result.csv',index=False))

fig,ax = plt.subplots(1,1,figsize=(30,30))
c=clf.best_estimator_['dc']
tree.plot_tree(c,ax=ax, max_depth=3,feature_names=list(train.columns),fontsize=19, impurity=False, label='root')
None

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier


parameters = {'dc__criterion': ['gini','entropy'], 'dc__n_estimators':[10,100], 'dc__max_depth':[5,10,15,20]}

pipe = Pipeline(steps=[('scale', StandardScaler()), ('dc', RandomForestClassifier())])

clf = GridSearchCV(pipe, parameters,return_train_score=True,verbose=2)
clf.fit(X, y)

clf.cv_results_['mean_test_score']

df = pd.DataFrame(clf.cv_results_)
df

df.drop(['mean_fit_time','std_fit_time','mean_score_time','std_score_time','split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'std_test_score', 'rank_test_score', 'split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score', 'std_train_score','params'],axis=1, inplace=True)
df.columns = ['Criterion', 'Max depth' ,'# of estimations', 'Test Accuracy', 'Train Accuracy']
# df = df[['Kernel', 'C', 'Gamma (RBF)', 'Degree(Poly)', 'Test Accuracy', 'Train Accuracy']]
# df.fillna('-',inplace=True)
df['Test Accuracy'] = 100 * df['Test Accuracy'].round(4)
df['Train Accuracy'] = 100 * df['Train Accuracy'].round(4)
# df
print(df.to_csv('rf-result.csv',index=False))
