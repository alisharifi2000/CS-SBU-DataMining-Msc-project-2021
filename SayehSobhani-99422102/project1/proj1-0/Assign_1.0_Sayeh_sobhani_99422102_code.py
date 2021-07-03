# -*- coding: utf-8 -*-
"""
@author: Sayeh Sobhani
written in Spyder
"""
#%%Importing packages and dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


dfr = pd.read_csv(r'C:\Users\Asus\Desktop\Data Mining\01_assignment\1.0\files and dataset and plots\immo_data.csv')


#%%Understanding and cleaning dataset

#to get a better understanding of how our data looks we look at the first 10 rows
dfr.shape
dfr.head(10)

#lets see some info about this dataset and its features
dfr.info()

dfr.describe()

#the mean for each numeric feature?
#miangin = dfr._get_numeric_data().mean()
#print(miangin)

#useless columns deletion 
dfr = dfr.drop(columns=['description'])
dfr = dfr.drop(columns=['facilities'])
dfr = dfr.drop(columns=['scoutId'])
dfr = dfr.drop(columns=['geo_bln','date'])
dfr = dfr.drop(columns=['houseNumber','livingSpaceRange','firingTypes'])
dfr = dfr.drop(columns=['streetPlain','street','geo_krs','regio2','regio3'])
dfr = dfr.drop(dfr[dfr['livingSpace'] == 0.0].index)
dfr = dfr.drop(dfr[dfr['totalRent'] == 0.0].index)

#how null data looks like
dfr.isna()

dfr.columns

#columns with null data 
dfr.isna().sum()/len(dfr)

#which columns hve >50 percent null
dfr.columns[((dfr.isna().sum()/len(dfr)) > 0.50)]

#removing those columns with >50 null 
dfr = dfr.drop(columns=dfr.columns[((dfr.isna().sum()/len(dfr)) > 0.50)])
dfr.columns
dfr.shape

#filling the remaining data with means and max seen data
    #numeric
dfr._get_numeric_data().mean()
dfr.fillna(dfr._get_numeric_data().mean(),inplace = True)
dfr.isna().sum()
    #Categorical most seens
for cols in dfr.columns:
    if dfr[cols].dtype == 'object' or dfr[cols].dtype == 'bool':
        print('column : ',cols)
        print(dfr[cols].value_counts().head())
        
     #Fill categorical nan with most seen data in each feature
for cols in dfr.columns:
    if dfr[cols].dtype == 'object' or dfr[cols].dtype == 'bool':
        print('cols : {} , value : {}'.format(cols , dfr[cols].value_counts().head(1).index[0]))
        dfr[cols].fillna(dfr[cols].value_counts().head(1).index[0],inplace = True)
#to check if nan left        
dfr.isna().sum()
dfr.shape
dfr.info()

"Some Visualization"
#regio1 categorical by base rent amounts PLOT visualization
#dfr['regio1'].value_counts()
#g = sns.FacetGrid(dfr, col='regio1', col_wrap=4)
#g = g.map(plt.hist, 'livingSpace', bins=20, range=(100,4000))

#outliers finding and removing    
for cols in dfr.columns:
    if dfr[cols].dtype == 'int64' or dfr[cols].dtype == 'float64':
        upper_range = dfr[cols].mean() + 3 * dfr[cols].std()
        lower_range = dfr[cols].mean() - 3 * dfr[cols].std()
        
        indexs = dfr[(dfr[cols] > upper_range) | (dfr[cols] < lower_range)].index
        dfr = dfr.drop(indexs)        
 
dfr.info()
dfr.shape
        
#%%categorical feature inspection 
#the number of categories in each feature
for cols in dfr.columns:
    if dfr[cols].dtype == 'object' or dfr[cols].dtype == 'bool':
        print('cols : {} , unique values : {}'.format(cols,dfr[cols].nunique()))
       
    
#the categories and their count in each feature
for cols in dfr.columns:
    if dfr[cols].dtype == 'object' or dfr[cols].dtype == 'bool':
        print('cols : {} ,\n {}'.format(cols,dfr[cols].value_counts()))

#regio1 edition based on plots from base rent amounts in line 75
dfr['regio1'].value_counts()
def edit_regio1(x):
    if x in ['Hamburg','Bremen','Saarland']:
        return 'other'
    else:
        return x
dfr['regio1_'] = dfr['regio1'].apply(edit_regio1)
dfr = dfr.drop(columns = ['regio1'])
dfr['regio1_'].value_counts()*100 / len(dfr)

    #visualization
plt_regio1_= dfr['regio1_'].value_counts().plot(kind='bar')

#heatingType
dfr['heatingType'].value_counts()
others = list(dfr['heatingType'].value_counts().tail(12).index)
def edit_heatingType(x):
    if x in others:
        return 'other'
    else:
        return x

dfr['heatingType_'] = dfr['heatingType'].apply(edit_heatingType)
dfr = dfr.drop(columns = ['heatingType'])
dfr['heatingType_'].value_counts()*100 / len(dfr)

#telekomTvOffer
dfr['telekomTvOffer'].value_counts()
others = list(dfr['telekomTvOffer'].value_counts().tail(2).index)
def edit_telekomTvOffer(x):
    if x in others:
        return 'other'
    else:
        return x

dfr['telekomTvOffer_'] = dfr['telekomTvOffer'].apply(edit_telekomTvOffer)
dfr = dfr.drop(columns = ['telekomTvOffer'])
dfr['telekomTvOffer_'].value_counts()*100 / len(dfr)

#typeOfFlat
dfr['typeOfFlat'].value_counts()
others = list(dfr['typeOfFlat'].value_counts().tail(6).index)
def edit_typeOfFlat(x):
    if x in others:
        return 'other'
    else:
        return x

dfr['typeOfFlat_'] = dfr['typeOfFlat'].apply(edit_typeOfFlat)
dfr = dfr.drop(columns = ['typeOfFlat'])
dfr['typeOfFlat_'].value_counts()*100 / len(dfr)

#condition
dfr['condition'].value_counts()
others = list(dfr['condition'].value_counts().tail(4).index)
def edit_condition(x):
    if x in others:
        return 'other'
    else:
        return x

dfr['condition_'] = dfr['condition'].apply(edit_condition)
dfr = dfr.drop(columns = ['condition'])
dfr['condition_'].value_counts()*100 / len(dfr)

#%%normalization of numeric data
for cols in dfr.columns:
    if dfr[cols].dtype == 'int64' or dfr[cols].dtype == 'float64':
         if cols != 'livingSpace':
              dfr[cols] = ((dfr[cols] - dfr[cols].mean())/(dfr[cols].std())) 
       
      
#%%correlation matrix
corr = dfr.corr()
corr
f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corr, square = True ,annot = True)

#%%Hypotheisi tests
dfr["regio1_"]
    #relation of noRoomsRange&livingSpace
stats.stats.spearmanr(dfr['livingSpace'],dfr['noRoomsRange'])

    #relation regio1 and rent
fstat, pval = stats.f_oneway(*[dfr.baseRent[dfr.regio1_ == s]
for s in dfr.regio1_.unique()])
print("Oneway Anova totalRent ~ edit region1 F=%.2f, p-value=%E" % (fstat, pval))
    
    #living space relation with no of floors
stats.stats.spearmanr(dfr['livingSpace'],dfr['numberOfFloors'])    

    #relation of having kitchen vs living space
stats.pointbiserialr(dfr['hasKitchen'],dfr["livingSpace"])

    #relation btween owning a kitchen and paying rent in total
stats.pointbiserialr(dfr['totalRent'],dfr["hasKitchen"])  
  
#%%Dummy variables for categorical
columns = []
for cols in dfr.columns:
    if dfr[cols].dtype == 'object' or dfr[cols].dtype == 'bool':
        columns.append(cols)
columns

dummies_feature = pd.get_dummies(dfr[columns])
dummies_feature.head()
dummies_feature.shape
dfr = pd.concat([dfr, dummies_feature], axis=1)
dfr.head()
dfr = dfr.drop(columns=columns)
dfr.head()
dfr.info()

#%%shuffling data and dividing target and feature, test and train
dfr = shuffle(dfr)
y = dfr['livingSpace'].values
x = dfr.drop(columns = ['livingSpace']).values

print(x.shape)
print(y.shape)

#splitting test and train
train_size = int(0.8 * x.shape[0])
train_size

x_train = x[:train_size]
y_train = y[:train_size]

print(x_train.shape)
print(y_train.shape)

x_test = x[train_size:]
y_test = y[train_size:]

print(x_test.shape)
print(y_test.shape)

#%%linear regr no package

#y^=w1∗x1+w2∗x2+⋯+w51∗x51+bias is the line we're lookin for
#setting parameters "w" and "b" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(51)
n = x_train.shape[0]

#Sets learning rate
lr = 0.001

#number of epochs
n_epochs = 500

test_error =  y_test - np.sum((w * x_test) + b , axis = 1)
test_mse = (test_error**2).mean()           
print('init MSE : ',test_mse)
              
for epoch in range(n_epochs):
    error = y_train - np.sum((w * x_train) + b , axis = 1)
    
    if epoch % 100 == 0:    
        print('epoch {} , MSE : {}'.format(epoch,(error**2).mean()))
    
    w_grad = [0] * x_train.shape[1]
    b_grad = 0
    
    for i in range(x_train.shape[1]):
        w_grad[i] = -1 * (x_train[:,i] * error).mean()
        w[i] = w[i] - (lr * w_grad[i])
        
    b_grad = -1 * error.mean()
    b = b - (lr*b_grad)
              

test_error = y_test - np.sum((w * x_test) + b , axis = 1)
test_mse = (test_error**2).mean() 
print('Final MSE : ',test_mse)

#comparison of the predicted value and the value in dataset (Accuracy)
y_pred = np.sum((w * x_test) + b , axis = 1)
temp = pd.DataFrame({'test':y_test,'pred':y_pred})
temp.head()

temp['upper_range'] = temp['test'] * 1.2
temp['lower_range'] = temp['test'] * 0.8

temp[(temp['upper_range'] >=temp['pred']) & (temp['pred'] >= temp['lower_range'])].shape[0] * 100/temp.shape[0]

#%%only noRooms feature linear regr no package

X = dfr['noRooms'].values
X.shape
X_train = X[:train_size]
print(X_train.shape)
X_test = X[train_size:]
print(X_test.shape)

np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)
n = X_train.shape[0]

# Sets learning rate
lr = 0.1


n_epochs = 500              
for epoch in range(n_epochs):
    error = y_train - ((w * X_train) + b)
    
    if epoch % 100 == 0:    
        print('epoch {} , MSE : {}'.format(epoch,(error**2).mean()))
     
    # adaptive learning rate
    if epoch % 200 == 0:
        lr = lr * 0.1
        
    
    w_grad = 0 
    b_grad = 0
    
    w_grad = -1 * (X_train * error).mean()
    w = w - (lr * w_grad)
        
    b_grad = -1 * error.mean()
    b = b - (lr*b_grad)
              
test_error = y_test - ((w * X_test) + b)
test_mse = (test_error**2).mean() 
print('Final MSE : ',test_mse)
print('Final learning rate : ',lr)    

 

#%%Linear regr with scikitlearn package all features
Linear = LinearRegression()
Linear.fit(x_train,y_train)
print("the coeficients are:")
print(Linear.coef_)
print("the intercept is:")
print(Linear.intercept_)

y_pred = Linear.predict(x_test)
mean_squared_error(y_pred,y_test)

temp = pd.DataFrame({'test':y_test,'pred':y_pred})
temp.head()

temp['upper_range'] = temp['test'] * 1.2
temp['lower_range'] = temp['test'] * 0.8

temp[(temp['upper_range'] >=temp['pred']) & (temp['pred'] >= temp['lower_range'])].shape[0] * 100/temp.shape[0]

#%%Linear regr package floor feature

X = dfr['numberOfFloors'].values
X.shape
X_train = X[:train_size]
print(X_train.shape)
X_test = X[train_size:]
print(X_test.shape)

Linear = LinearRegression()
Linear.fit(x_train,y_train)
print("the coeficients are:")
print(Linear.coef_)
print("the intercept is:")
print(Linear.intercept_)

y_pred = Linear.predict(x_test)
mean_squared_error(y_pred,y_test)

temp = pd.DataFrame({'test':y_test,'pred':y_pred})
temp.head()

temp['upper_range'] = temp['test'] * 1.2
temp['lower_range'] = temp['test'] * 0.8

temp[(temp['upper_range'] >=temp['pred']) & (temp['pred'] >= temp['lower_range'])].shape[0] * 100/temp.shape[0]

#%%number of floors feature linear regr no package

X = dfr['numberOfFloors'].values
X.shape
X_train = X[:train_size]
print(X_train.shape)
X_test = X[train_size:]
print(X_test.shape)

np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)
n = X_train.shape[0]

# Sets learning rate
lr = 0.1


n_epochs = 500              
for epoch in range(n_epochs):
    error = y_train - ((w * X_train) + b)
    
    if epoch % 100 == 0:    
        print('epoch {} , MSE : {}'.format(epoch,(error**2).mean()))
     
    # adoptive learning rate
    if epoch % 200 == 0:
        lr = lr * 0.1
        
    
    w_grad = 0 
    b_grad = 0
    
    w_grad = -1 * (X_train * error).mean()
    w = w - (lr * w_grad)
        
    b_grad = -1 * error.mean()
    b = b - (lr*b_grad)
              
test_error = y_test - ((w * X_test) + b)
test_mse = (test_error**2).mean() 
print('Final MSE : ',test_mse)
print('Final learning rate : ',lr)   

#%%PCA

#pov 0.99
pca = PCA(0.99)
x_pca = pca.fit_transform(x)
x_pca.shape
x_pca_train = x_pca[:train_size]
print(x_pca_train.shape)
x_pca_test = x_pca[train_size:]
print(x_pca_test.shape)

#pov 0.95
pca = PCA(0.95)
x_pca = pca.fit_transform(x)
x_pca.shape
x_pca_train = x_pca[:train_size]
print(x_pca_train.shape)
x_pca_test = x_pca[train_size:]
print(x_pca_test.shape)

#pov 0.90
pca = PCA(0.90)
x_pca = pca.fit_transform(x)
x_pca.shape
x_pca_train = x_pca[:train_size]
print(x_pca_train.shape)
x_pca_test = x_pca[train_size:]
print(x_pca_test.shape)

#pov 0.70
pca = PCA(0.70)
x_pca = pca.fit_transform(x)
x_pca.shape
x_pca_train = x_pca[:train_size]
print(x_pca_train.shape)
x_pca_test = x_pca[train_size:]
print(x_pca_test.shape)


#%% linear regr no package different margin
#margin or 0.1
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(51)
n = x_train.shape[0]

#Sets learning rate
lr = 0.001

#number of epochs
n_epochs = 500

test_error =  y_test - np.sum((w * x_test) + b , axis = 1)
test_mse = (test_error**2).mean()           
print('init MSE : ',test_mse)
              
for epoch in range(n_epochs):
    error = y_train - np.sum((w * x_train) + b , axis = 1)
    
    if epoch % 100 == 0:    
        print('epoch {} , MSE : {}'.format(epoch,(error**2).mean()))
    
    w_grad = [0] * x_train.shape[1]
    b_grad = 0
    
    for i in range(x_train.shape[1]):
        w_grad[i] = -1 * (x_train[:,i] * error).mean()
        w[i] = w[i] - (lr * w_grad[i])
        
    b_grad = -1 * error.mean()
    b = b - (lr*b_grad)
              

test_error = y_test - np.sum((w * x_test) + b , axis = 1)
test_mse = (test_error**2).mean() 
print('Final MSE : ',test_mse)

#comparison of the predicted value and the value in dataset
y_pred = np.sum((w * x_test) + b , axis = 1)
temp = pd.DataFrame({'test':y_test,'pred':y_pred})
temp.head()

temp['upper_range'] = temp['test'] * 1.1
temp['lower_range'] = temp['test'] * 0.9

temp[(temp['upper_range'] >=temp['pred']) & (temp['pred'] >= temp['lower_range'])].shape[0] * 100/temp.shape[0]

