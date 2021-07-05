# -*- coding: utf-8 -*-
"""
@author: Sayeh
by spyder
"""
#%%importing data and packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as F1


df1 = pd.read_csv (r'C:\Users\Asus\Desktop\Data Mining\02_assignment\data\part1\immo_data.csv')
df2 = pd.read_csv (r'C:\Users\Asus\Desktop\Data Mining\02_assignment\data\part3\train.csv')

#%%Understanding data for first dataset

#to get a better understanding of how our data looks
df1.shape
df1.info()
df1.describe()
df1.head(10)
df1.columns
#duplicated datapoints 
df1.duplicated(keep=False).any()
#the mean for each numeric feature?
df1._get_numeric_data().mean()
#useless columns deletion '',
df1 = df1.drop(columns=['description'])
df1 = df1.drop(columns=['facilities', 'telekomHybridUploadSpeed'])
df1 = df1.drop(columns=['scoutId'])
df1 = df1.drop(columns=['geo_bln','date'])
df1 = df1.drop(columns=['houseNumber','livingSpaceRange','firingTypes'])
df1 = df1.drop(columns=['streetPlain','street','geo_krs','regio2','regio3'])
df1 = df1.drop(df1[df1['livingSpace'] == 0.0].index)
df1 = df1.drop(df1[df1['totalRent'] == 0.0].index)

#how null data looks like
df1.isna()

df1.columns

#columns with null data 
df1.isna().sum()/len(df1)

#which columns hve >50 percent null
df1.columns[((df1.isna().sum()/len(df1)) > 0.50)]

#removing those columns with >50 null 
df1 = df1.drop(columns=df1.columns[((df1.isna().sum()/len(df1)) > 0.50)])
df1.columns
df1.shape

#filling the remaining data with means and max seen data
    #numeric
df1._get_numeric_data().mean()
df1.fillna(df1._get_numeric_data().mean(),inplace = True)
df1.isna().sum()
    #Categorical most seens
for cols in df1.columns:
    if df1[cols].dtype == 'object' or df1[cols].dtype == 'bool':
        print('column : ',cols)
        print(df1[cols].value_counts().head())
        
     #Fill categorical nan with most seen data in each feature
for cols in df1.columns:
    if df1[cols].dtype == 'object' or df1[cols].dtype == 'bool':
        print('cols : {} , value : {}'.format(cols , df1[cols].value_counts().head(1).index[0]))
        df1[cols].fillna(df1[cols].value_counts().head(1).index[0],inplace = True)

#to check if nan left        
df1.isna().sum()
df1.shape
df1.info()
#no null data left

#outliers finding and removing    
for cols in df1.columns:
    if df1[cols].dtype == 'int64' or df1[cols].dtype == 'float64':
        upper_range = df1[cols].mean() + 3 * df1[cols].std()
        lower_range = df1[cols].mean() - 3 * df1[cols].std()
        
        indexs = df1[(df1[cols] > upper_range) | (df1[cols] < lower_range)].index
        df1 = df1.drop(indexs)        
 
df1.info()
df1.shape

#%%categorical feature inspection 
#the number of categories in each feature
for cols in df1.columns:
    if df1[cols].dtype == 'object' or df1[cols].dtype == 'bool':
        print('cols : {} , unique values : {}'.format(cols,df1[cols].nunique()))
       
    
#the categories and their count in each feature
for cols in df1.columns:
    if df1[cols].dtype == 'object' or df1[cols].dtype == 'bool':
        print('cols : {} ,\n {}'.format(cols,df1[cols].value_counts()))

#regio1 edition based on plots from base rent amounts in line 75
df1['regio1'].value_counts()
def edit_regio1(x):
    if x in ['Hamburg','Bremen','Saarland']:
        return 'other'
    else:
        return x
df1['regio1_'] = df1['regio1'].apply(edit_regio1)
df1 = df1.drop(columns = ['regio1'])
df1['regio1_'].value_counts()*100 / len(df1)

    #visualization
#plt_regio1_= df1['regio1_'].value_counts().plot(kind='bar')

#heatingType
df1['heatingType'].value_counts()
others = list(df1['heatingType'].value_counts().tail(12).index)
def edit_heatingType(x):
    if x in others:
        return 'other'
    else:
        return x

df1['heatingType_'] = df1['heatingType'].apply(edit_heatingType)
df1 = df1.drop(columns = ['heatingType'])
df1['heatingType_'].value_counts()*100 / len(df1)

#telekomTvOffer
df1['telekomTvOffer'].value_counts()
others = list(df1['telekomTvOffer'].value_counts().tail(2).index)
def edit_telekomTvOffer(x):
    if x in others:
        return 'other'
    else:
        return x

df1['telekomTvOffer_'] = df1['telekomTvOffer'].apply(edit_telekomTvOffer)
df1 = df1.drop(columns = ['telekomTvOffer'])
df1['telekomTvOffer_'].value_counts()*100 / len(df1)

#typeOfFlat
df1['typeOfFlat'].value_counts()
others = list(df1['typeOfFlat'].value_counts().tail(6).index)
def edit_typeOfFlat(x):
    if x in others:
        return 'other'
    else:
        return x

df1['typeOfFlat_'] = df1['typeOfFlat'].apply(edit_typeOfFlat)
df1 = df1.drop(columns = ['typeOfFlat'])
df1['typeOfFlat_'].value_counts()*100 / len(df1)

#condition
df1['condition'].value_counts()
others = list(df1['condition'].value_counts().tail(4).index)
def edit_condition(x):
    if x in others:
        return 'other'
    else:
        return x

df1['condition_'] = df1['condition'].apply(edit_condition)
df1 = df1.drop(columns = ['condition'])
df1['condition_'].value_counts()*100 / len(df1)

#%%normalization of numeric data
for cols in df1.columns:
    if df1[cols].dtype == 'int64' or df1[cols].dtype == 'float64':
         if cols != 'livingSpace':
              df1[cols] = ((df1[cols] - df1[cols].mean())/(df1[cols].std())) 
       
#%%correlation

df1.corr().livingSpace.sort_values()

#%%Dummy variables for categorical
columns = []
for cols in df1.columns:
    if df1[cols].dtype == 'object' or df1[cols].dtype == 'bool':
        columns.append(cols)
columns

dummies_feature = pd.get_dummies(df1[columns])
dummies_feature.head()
dummies_feature.shape
df1 = pd.concat([df1, dummies_feature], axis=1)
df1.head()
df1 = df1.drop(columns=columns)
df1.head()
df1.info()

#%%shuffling data and dividing target and feature, test and train

#most correlated feature(noRooms) and living space as target
df1 = shuffle(df1)
y = df1['livingSpace'].values
x = df1['noRooms'].values

print(x.shape)
print(y.shape)

#splitting test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#print(x_train.shape)
#print(y_train.shape)

#print(x_test.shape)
#print(y_test.shape)

#%%linear regr no package and accuracy function
#y^=w∗x+b

def LinearRegr_grad(x, y, lr=0.001, epochs=500):
    w = 0
    b = 0
    n = np.float(x.shape[0])
    losses = []
    
    for i in range(epochs):
        if i % 100 == 0:
            print(i)
        
        y_hat = w * x + b
        
        mse = (1/n) * np.sum((y - y_hat)**2)
        losses.append(mse)

        dw = (-2/n) * np.sum(np.dot(x.T, (y - y_hat)))
        db = (-2/n) * np.sum(y - y_hat)

        w = w - lr * dw
        b = b - lr * db
    
    return w, b


def accur(actual, predicted):
    truelb = 0
    for i in range(len(actual)):
        if predicted[i] >= actual[i]*0.9 and predicted[i] <= actual[i]*1.1:
            truelb += 1
    return truelb / float(len(actual)) * 100.0


#%% k fold (5,10)
# 5fold cv
kf = KFold(n_splits=5, random_state=None)

acc_score = []
errors = []

for train_index , test_index in kf.split(x):
    

     
    w, b = LinearRegr_grad(x_train, y_train, epochs=2000)
    y_hat = w * x_test + b
    
    acc_score.append(accur(y_test, y_hat))
    errors.append(mse(y_test, y_hat))



print('Folds Accuracy:')
for i in acc_score:
    print('%.3f' %i)
print('Average Accuracy: %.3f' % (np.mean(acc_score)))

print('\nFolds MSE:')
for i in errors:
    print('%.3f' %i)
print('Average MSE: %.3f' % (np.mean(errors)))

# 10fold
kf = KFold(n_splits=10, random_state=None)

acc_score = []
errors = []

for train_index , test_index in kf.split(x):
    
    w, b = LinearRegr_grad(x_train, y_train, epochs=2000)
    y_hat = w * x_test + b
    
    acc_score.append(accur(y_test, y_hat))
    errors.append(mse(y_test, y_hat))
    
#%%10 fold

kf = KFold(n_splits=10, random_state=None)

acc_score = []
errors = []

for train_index , test_index in kf.split(x):
    

     
    w, b = LinearRegr_grad(x_train, y_train, epochs=2000)
    y_hat = w * x_test + b
    
    acc_score.append(accur(y_test, y_hat))
    errors.append(mse(y_test, y_hat))



print('Folds Accuracy:')
for i in acc_score:
    print('%.3f' %i)
print('Average Accuracy: %.3f' % (np.mean(acc_score)))

print('\nFolds MSE:')
for i in errors:
    print('%.3f' %i)
print('Average MSE: %.3f' % (np.mean(errors)))

# 10fold
kf = KFold(n_splits=10, random_state=None)

acc_score = []
errors = []

for train_index , test_index in kf.split(x):
    
    w, b = LinearRegr_grad(x_train, y_train, epochs=2000)
    y_hat = w * x_test + b
    
    acc_score.append(accur(y_test, y_hat))
    errors.append(mse(y_test, y_hat))    

#%%models with package

df1 = shuffle(df1)
y = df1['livingSpace'].values
x = df1['noRooms'].values


cv = KFold(n_splits=5, random_state=1, shuffle=True)

model = LinearRegression()

acc = cross_val_score(model, x, y, cv=cv, n_jobs=-1)
error = -cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

print('Folds Accuracy:')
for i in acc:
    print('%.3f' %i)
print('Average Accuracy: %.3f' % (np.mean(acc)))

print('\nFolds MSE:')
for i in error:
    print('%.3f' %i)
print('Average MSE: %.3f' % (np.mean(error)))


cv = KFold(n_splits=5, random_state=1, shuffle=True)

rdg = Ridge()

acc = cross_val_score(rdg, x, y, cv=cv, n_jobs=-1)
error = cross_val_score(rdg, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

print('Folds Accuracy:')
for i in acc:
    print('%.3f' %i)
print('Average Accuracy: %.3f' % (np.mean(acc)))

print('\nFolds MSE:')
for i in error:
    print('%.3f' %i)
print('Average MSE: %.3f' % (np.mean(error)))
cv = KFold(n_splits=5, random_state=1, shuffle=True)

lso = Lasso()

acc = cross_val_score(lso, x, y, cv=cv, n_jobs=-1)
error = -cross_val_score(lso, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

print('Folds Accuracy:')
for i in acc:
    print('%.3f' %i)
print('Average Accuracy: %.3f' % (np.mean(acc)))

print('\nFolds MSE:')
for i in error:
    print('%.3f' %i)
print('Average MSE: %.3f' % (np.mean(error)))
#%% another aproach package

k=5
cv = KFold(n_splits=k, random_state=None)


# create model
model = LinearRegression()

scoring = {'accuracy': 'accuracy',
           'recall': 'recall',
           'precision': 'precision',
           'roc_auc': 'roc_auc'}
# evaluate model
cross_val_scores = cross_validate(model, x, y, cv=cv, scoring=scoring)

# report performance




















#%%Understanding data for the second dataset

#to get a better understanding of how our data looks
df2.shape
df2.info() #no null data here
df2.describe()
df2.head(10)
df2.columns
#duplicated datapoints 
df2.duplicated(keep=False).any() #no duplicates found

#%%normalization of numeric data??? do we need it
for cols in df2.columns:
    if df2[cols].dtype == 'int64' or df2[cols].dtype == 'float64':
         if cols != 'price_range':
              df2[cols] = ((df2[cols] - df2[cols].mean())/(df2[cols].std())) 
       

#%%spliting features and target
df2 = shuffle(df2)
x = df2.drop('price_range', axis=1)
y = df2['price_range']

#splitting test & train 80-20

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

#logistic regression for mobile cost classification 4 class
model = LogisticRegression()
model.fit(x_train,y_train)
predict = model.predict(x_test)

print(classification_report(y_test, predict))

#how are the 4 classes of price distributed in test and train
y_test.value_counts()
y_train.value_counts()
y_train

#2-class start
#%% working with two classes 0 and 1 for price range
#4 classes into 2 by adding all >0 into class 1
def edit_priceclass(x):
    if x != 0:
        return 1
    else:
        return x

y_test = y_test.apply(edit_priceclass)
y_train = y_train.apply(edit_priceclass)

y_test.value_counts()
y_train.value_counts()

#logistic regression for mobile cost classification 2 class
model = LogisticRegression()
model.fit(x_train,y_train)
predict = model.predict(x_test)

print(classification_report(y_test, predict))

#%%balancing the data and running regression with balances classes for 2-class
y.value_counts()

ny = list()
for i in y:
    if i==0:
        ny.append(0)
    else:
        ny.append(1)
ny = np.array(ny)

for i in range(2):
    print(i, '-> ', np.count_nonzero(ny == i))
    
oversample = SMOTE(sampling_strategy='minority')
sm_x ,sm_y = oversample.fit_resample(x,ny)

for i in range(2):
    print(i, '-> ', np.count_nonzero(sm_y == i))
    
x_train, x_test, y_train, y_test = train_test_split(sm_x, sm_y, test_size=0.2, random_state=101)

model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

#2class end


#%% Forward selection (https://pythonhealthcare.org/2020/01/04/feature-selection-2-model-forward-selection/)

#standardising data 
def standardise_data(x_train, x_test):
    
    # Initialise a new scaling object for normalising input data
    sc = StandardScaler() 
    # Set up the scaler just on the training set
    sc.fit(x_train)
    # Apply the scaler to the training and test sets
    train_std = sc.transform(x_train)
    test_std = sc.transform(x_test)
    
    return train_std, test_std

# Create list to store accuracies and chosen features
roc_auc_by_feature_number = []
chosen_features = []

# Initialise chosen features list and run tracker
available_features = list(x)
run = 0
number_of_features = len(list(x))

# Loop through feature list to select next feature
while len(available_features)> 0:

    # Track and pront progress
    run += 1
    print ('Feature run {} of {}'.format(run, number_of_features))
    
    # Convert DataFrames to NumPy arrays
    y_np = y.values
    
    # Reset best feature and accuracy
    best_result = 0
    best_feature = ''

    # Loop through available features
    for feature in available_features:

        # Create copy of already chosen features to avoid orginal being changed
        features_to_use = chosen_features.copy()
        # Create a list of features from features already chosen + 1 new feature
        features_to_use.append(feature)
        # Get data for features, and convert to NumPy array
        X_np = x[features_to_use].values
        
        # Set up lists to hold results for each selected features
        test_auc_results = []
    
        # Set up k-fold training/test splits
        number_of_splits = 5
        skf = StratifiedKFold(n_splits = number_of_splits)
        skf.get_n_splits(X_np, y)
    
        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
            
            # Get X and Y train/test
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Get X and Y train/test
            X_train_std, X_test_std = standardise_data(X_train, X_test)
    
            # Set up and fit model
            model = LogisticRegression(solver='lbfgs')
            model.fit(X_train_std,y_train)
    
            # Predict test set labels
            y_pred_test = model.predict(X_test_std)
            
            # Calculate accuracy of test sets
            accuracy_test = np.mean(y_pred_test == y_test)
          
            # Get ROC AUC
            probabilities = model.predict_proba(X_test_std)
            probabilities = probabilities[:, 1] # Probability of 'survived'
            fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label=3)

            roc_auc = auc(fpr, tpr)
            test_auc_results.append(roc_auc)
        
        # Get average result from all k-fold splits
        feature_auc = np.mean(test_auc_results)
        
        # Update chosen feature and result if this feature is a new best
        if feature_auc > best_result:
            best_result = feature_auc
            best_feature = feature

    # Add mean accuracy and AUC to record of accuracy by feature number
    roc_auc_by_feature_number.append(best_result)
    chosen_features.append(best_feature)
    available_features.remove(best_feature)

# Put results in DataFrame
results = pd.DataFrame()
results['feature to add'] = chosen_features
results['ROC AUC'] = roc_auc_by_feature_number

results

#determining which features to use for log regr
results['feature to add'][results['ROC AUC']>0.540].values

#features we will use for model
x_fsel = x[['mobile_wt', 'fc', 'pc', 'battery_power', 'px_width','touch_screen']]

x_train, x_test, y_train, y_test = train_test_split(x_fsel, y, test_size=0.2, random_state=101)

#logistic regr on picked features by forward selection
model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

#%% log regr with PCA 

pca = PCA(n_components=6)
pca.fit(x)
x_pca = pca.transform(x)

#spliting test and train with pca in mind
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=101)

#log regr for pca picked features
model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

#%%backward selection for features

# Create list to store accuracies and chosen features
roc_auc_by_feature_number = []
chosen_features = []

# Initialise chosen features list and run tracker
available_features = list(x)
run = 0
number_of_features = len(list(x))

# Create initial reference performance
reference_auc = 1.0 # used to compare reduction in AUC

# Loop through feature list to select next feature
while len(available_features)> 1:

    # Track and pront progress
    run += 1
    print ('Feature run {} of {}'.format(run, number_of_features-1))
    
    # Convert DataFrames to NumPy arrays
    y_np = y.values
    
    # Reset best feature and accuracy
    best_result = 1.0
    best_feature = ''

    # Loop through available features
    for feature in available_features:

        # Create copy of already chosen features to avoid orginal being changed
        features_to_use = available_features.copy()
        # Create a list of features to use by removing 1 feature
        features_to_use.remove(feature)
        # Get data for features, and convert to NumPy array
        X_np = x[features_to_use].values
        
        # Set up lists to hold results for each selected features
        test_auc_results = []
    
        # Set up k-fold training/test splits
        number_of_splits = 5
        skf = StratifiedKFold(n_splits = number_of_splits)
        skf.get_n_splits(X_np, y)
    
        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
            
            # Get X and Y train/test
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Get X and Y train/test
            X_train_std, X_test_std = standardise_data(X_train, X_test)
    
            # Set up and fit model
            model = LogisticRegression(solver='lbfgs')
            model.fit(X_train_std,y_train)
    
            # Predict test set labels
            y_pred_test = model.predict(X_test_std)
            
            # Calculate accuracy of test sets
            accuracy_test = np.mean(y_pred_test == y_test)
          
            # Get ROC AUC
            probabilities = model.predict_proba(X_test_std)
            probabilities = probabilities[:, 1] # Probability of 'survived'
            fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label=3)
            roc_auc = auc(fpr, tpr)
            test_auc_results.append(roc_auc)
        
        # Get average result from all k-fold splits
        feature_auc = np.mean(test_auc_results)
    
        # Update chosen feature and result if this feature is a new best
        # We are looking for the smallest drop in performance
        drop_in_performance = reference_auc - feature_auc
        if drop_in_performance < best_result:
            best_result = drop_in_performance
            best_feature = feature
            best_auc = feature_auc
                
    # k-fold splits are complete    
    # Add mean accuracy and AUC to record of accuracy by feature number
    roc_auc_by_feature_number.append(best_auc)
    chosen_features.append(best_feature)    
    available_features.remove(best_feature)
    reference_auc = best_auc

# Add last remaining feature
chosen_features += available_features
roc_auc_by_feature_number.append(0)
    
# Put results in DataFrame
# Reverse order of lists with [::-1] so best features first
results = pd.DataFrame()
results['feature removed'] = chosen_features[::-1]
results['ROC AUC'] = roc_auc_by_feature_number[::-1]
results

#determining which features to use for log regr
results['feature removed'][results['ROC AUC']>0.54].values

x_bsel = x[['four_g', 'blue', 'int_memory', 'wifi']]

x_train, x_test, y_train, y_test = train_test_split(x_bsel, y, test_size=0.2, random_state=101)

#log regr for backward selection picked features
model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

#%%Kfold cross validation for the second dataset
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score

# prepare the cross-validation procedure
k=5
cv = KFold(n_splits=k, random_state=None)


# create model
clf = LogisticRegression()

scoring = {'accuracy': 'accuracy',
           'recall': 'recall',
           'precision': 'precision',
           'roc_auc': 'roc_auc'}
# evaluate model
cross_val_scores = cross_validate(clf, x, y, cv=cv, scoring=scoring)

# report performance

