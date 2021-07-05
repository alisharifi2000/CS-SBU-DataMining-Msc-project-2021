import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from collections import Counter
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def readData(dataName):
    df = pd.read_csv(dataName )
    return df


def read_bit_data(dataName):
    df = pd.read_csv(dataName,thousands=',' ) #, sep='\t',
    return df


def read_tournement_data(dataName):
    df = pd.read_csv(dataName ).set_index("id")
    return df


def plot_result(acc,per,recall,f1,title) :
    
    results=[round(np.mean(acc),3),round(np.mean(per),3),round(np.mean(recall),3),round(np.mean(f1),3)]
    plt.figure(figsize=(10,8),dpi=100)
    plt.bar(["accuracy","percision","recall","f1_score"] ,results,color="rgby")
    for index,value in enumerate(results):
        plt.text(index,value,str(value))
    plt.title(title)
    # plt.xticks(rotation=90)
    if not os.path.exists("mobile-output"):
        os.makedirs("mobile-output")
    plt.savefig("mobile-output/"+title)
    plt.show()
    
    
    
    
def SVM_func_base(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC()
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label,"SVM with: default parameters")
    
def SVM_func_2(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC(kernel="linear",gamma="auto")
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label,"SVM with: kernel=linear,gamma=auto")
    

def SVM_func_3(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC(kernel="linear",gamma="scale")
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label,"SVM with: kernel=linear,gamma=scale")

def SVM_func_4(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC(kernel="poly",gamma="auto")
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label,"SVM with: kernel=poly , gamma=auto")


def SVM_func_5(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC(kernel="poly",gamma="scale")
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label,"SVM with: kernel=poly , ganna=scale")

def SVM_func_6(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC(C=4, class_weight="balanced",decision_function_shape="ovo")
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label,"SVM with: C=4 , class_weight=balanced , decision_function_shape=ovo")    
     
def SVM_func_feature_engineering(train_data,test_data,train_label,test_label,my_title="" ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC(kernel="linear",gamma="auto")
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label,my_title)
    

    
def plot_results(y_pred,test_label,my_title):
    Con_matrix=confusion_matrix(test_label, y_pred)
    classfi_report=classification_report(test_label, y_pred,output_dict=True)
    
    fig, ax = plot_confusion_matrix(conf_mat=Con_matrix,
                                show_absolute=True,
                                show_normed=False,
                                colorbar=True)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax.set_title('confusion_matrix of -'+my_title)
    plt.show()    
    # save to array
    accuracy =accuracy_score(test_label, y_pred)
    percision = classfi_report['macro avg']['precision'] 
    recall = classfi_report['macro avg']['recall']    
    f1_score = classfi_report['macro avg']['f1-score']
    results=[np.round(accuracy,3),np.round(percision,3),np.round(recall,3),np.round(f1_score,3)]
    
    plt.figure(figsize=(10,8),dpi=100)
    plt.bar(["accuracy","percision","recall","f1_score"] ,results,color="rgby")
    for index,value in enumerate(results):
        plt.text(index,value,str(value))
    plt.title( my_title)
    # plt.xticks(rotation=90)
    if not os.path.exists("mobile-output"):
        os.makedirs("mobile-output")
    plt.savefig("mobile-output/"+my_title)
    plt.show()




def soft_margin_svm(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    svmm = SVC(kernel="linear", C=0.1 )
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label," soft margin with c=01 ")    
    
    
    
def hard_margin_svm(train_data,test_data,train_label,test_label ):

    #can cause overfitting(tryes to seperate whole of points with hyperplane)
    print("------------SVM is starting-------------- ")
    svmm = SVC(kernel="linear",C=1000)
    svmm.fit(train_data, train_label)
    y_pred = svmm.predict(test_data)
    plot_results(y_pred,test_label," hard margin with c= 1000")    
    
#######################
#read data
#######################

train_data_path="mobile_data/train.csv"
test_data_path="mobile_data/test.csv"
data_train=readData(train_data_path)
data_test=readData(test_data_path)
label_idx=data_train.columns.get_loc("price_range")
label_train=np.array(data_train.iloc[:,label_idx])
data_train=data_train.drop(["price_range"],axis=1)
data_test=data_test.drop(["id"],axis=1)
data_all=np.concatenate((data_train,data_test))
 
 
##############################################
#           data normalization(scaling)
##############################################
scaler = MinMaxScaler()
scaler.fit(data_all)
data_all_normalized=scaler.transform(np.array(data_all))

data_train_normalized=data_all_normalized[:len(data_train),:]
data_test_normalized=data_all_normalized[len(data_train):,:]

######################################################################
#              SVM on main data
#######################################################################
x_train, x_test, y_train, y_test, = train_test_split(data_train_normalized, label_train, test_size=0.2, random_state=0,shuffle=True)

SVM_func_base(x_train, x_test, y_train, y_test)
SVM_func_2(x_train, x_test, y_train, y_test)
SVM_func_3(x_train, x_test, y_train, y_test)
SVM_func_4(x_train, x_test, y_train, y_test)
SVM_func_5(x_train, x_test, y_train, y_test)
SVM_func_6(x_train, x_test, y_train, y_test)
soft_margin_svm(x_train, x_test, y_train, y_test)
hard_margin_svm(x_train, x_test, y_train, y_test)

######################################################################
#                  binning with bins=5
#######################################################################

data_train_q5=data_train.copy()
data_train_q5["battery_power"]=pd.cut(data_train_q5['battery_power'],bins=5)
onehot_columns=pd.get_dummies(data_train_q5["battery_power"], prefix='bt_pow')
data_train_q5.drop("battery_power",axis=1,inplace=True)
data_train_q5_onehot=pd.concat([data_train_q5,onehot_columns],axis=1)
#normalize data
scaler = MinMaxScaler()
scaler.fit(data_train_q5_onehot)
data_train_q5_onehot_norm=scaler.transform(np.array(data_train_q5_onehot))
#call svm on data
x_train, x_test, y_train, y_test, = train_test_split(data_train_q5_onehot_norm, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test,"binning(bins=5) and onehot encoding")


######################################################################
#                  binning with bins=10
#######################################################################

data_train_q10=data_train.copy()
data_train_q10["battery_power"]=pd.cut(data_train_q10['battery_power'], bins=10)
onehot_columns=pd.get_dummies(data_train_q10["battery_power"], prefix='bt_pow')
data_train_q10.drop("battery_power",axis=1,inplace=True)
data_train_q10_onehot=pd.concat([data_train_q10,onehot_columns],axis=1)
#normalize data
scaler = MinMaxScaler()
scaler.fit(data_train_q10_onehot)
data_train_q10_onehot_norm=scaler.transform(np.array(data_train_q10_onehot))
#call svm on data
x_train, x_test, y_train, y_test, = train_test_split(data_train_q10_onehot_norm, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test,"binning(bins=10) and onehot encoding")


######################################################################
#                  binning with bins=20
#######################################################################

data_train_q20=data_train.copy()
data_train_q20["battery_power"]=pd.cut(data_train_q20['battery_power'], bins=20)
onehot_columns=pd.get_dummies(data_train_q20["battery_power"], prefix='bt_pow')
data_train_q20.drop("battery_power",axis=1,inplace=True)
data_train_q20_onehot=pd.concat([data_train_q20,onehot_columns],axis=1)
#normalize data
scaler = MinMaxScaler()
scaler.fit(data_train_q20_onehot)
data_train_q20_onehot_norm=scaler.transform(np.array(data_train_q20_onehot))
#call svm on data
x_train, x_test, y_train, y_test, = train_test_split(data_train_q20_onehot_norm, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test,"binning(bins=20) and onehot encoding")


######################################################################
#                  binning with bins=30
#######################################################################

data_train_q30=data_train.copy()
data_train_q30["battery_power"]=pd.cut(data_train_q30['battery_power'],bins=30)
onehot_columns=pd.get_dummies(data_train_q30["battery_power"], prefix='bt_pow')
data_train_q30.drop("battery_power",axis=1,inplace=True)
data_train_q30_onehot=pd.concat([data_train_q30,onehot_columns],axis=1)
#normalize data
scaler = MinMaxScaler()
scaler.fit(data_train_q30_onehot)
data_train_q30_onehot_norm=scaler.transform(np.array(data_train_q30_onehot))
#call svm on data
x_train, x_test, y_train, y_test, = train_test_split(data_train_q30_onehot_norm, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test,"binning(bins=30) and onehot encoding")


# diffrent ways to bin data
# bin_labels_5 = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
# df['quantile_ex_3'] = pd.qcut(df['ext price'],q=[0, .2, .4, .6, .8, 1],labels=bin_labels_5)
# df['quantile_ex_3'] = pd.qcut(df['ext price'],q=[0, .2, .4, .6, .8, 1],labels=False,)


######################################################################
#                  #create new feature(volume)
#######################################################################

data_train_new_feature=data_train.copy()
data_train_new_feature["volume"]=data_train["px_height"] * data_train["px_width"] 
#normalize data
scaler = MinMaxScaler()
scaler.fit(data_train_new_feature)
data_train_new_feature_norm=scaler.transform(np.array(data_train_new_feature))
#call svm on data
x_train, x_test, y_train, y_test, = train_test_split(data_train_new_feature_norm, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test," - New feature (Volume)")
 

######################################################################
#                        #Log transform
#######################################################################

data_train_log_transform = np.log(data_train)
data_train_log_transform.replace([np.inf, -np.inf], np.nan, inplace=True)
imp = SimpleImputer(strategy='mean')
imp.fit( data_train_log_transform )
data_train_log_transform_imputed=imp.transform(data_train_log_transform)
x_train, x_test, y_train, y_test, = train_test_split(data_train_log_transform_imputed, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test," - Log Transform ")
 

######################################################################
#                        #standard scaler
#######################################################################
 
scaler = StandardScaler()
scaler.fit(data_train)
data_train_standard_scaler=scaler.transform(data_train)
x_train, x_test, y_train, y_test, = train_test_split(data_train_standard_scaler, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test," - standard scaler transform ")

######################################################################
#                        # apply all feature engineering
#######################################################################

#create new feature
data_train_new_feature=data_train.copy()
data_train_new_feature["volume"]=data_train["px_height"] * data_train["px_width"] 

#create bins
data_train_q5=data_train_new_feature.copy()
data_train_q5["battery_power"]=pd.cut(data_train_q5['battery_power'],bins=5)
onehot_columns=pd.get_dummies(data_train_q5["battery_power"], prefix='bt_pow')
data_train_q5.drop("battery_power",axis=1,inplace=True)
data_train_q5_onehot=pd.concat([data_train_q5,onehot_columns],axis=1)

#log transform
data_train_log_transform = np.log(data_train_q5_onehot)
data_train_log_transform.replace([np.inf, -np.inf], np.nan, inplace=True)
imp = SimpleImputer(strategy='mean')
imp.fit( data_train_log_transform )
data_train_log_transform_imputed=imp.transform(data_train_log_transform)


x_train, x_test, y_train, y_test, = train_test_split(data_train_log_transform_imputed, label_train, test_size=0.2, random_state=0,shuffle=True)
SVM_func_feature_engineering(x_train, x_test, y_train, y_test,"applying all feature engineering parts together")









#########################################################################
#########################################################################

#                             Decision Tree                              #

#########################################################################
#########################################################################

from sklearn.tree import DecisionTreeClassifier



data_train_Decision_tree=data_train.copy()
#normalize data
scaler = MinMaxScaler()
scaler.fit(data_train_Decision_tree)
data_train_Decision_tree_norm=scaler.transform(np.array(data_train_Decision_tree))
#call svm on data
x_train, x_test, y_train, y_test, = train_test_split(data_train_Decision_tree_norm, label_train, test_size=0.2, random_state=0,shuffle=True)

#define decision tree different functions

def DT_func_base(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier()
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: default parameters")
    
def DT_func_2(train_data,test_data,train_label,test_label ):
    
 
    print("------------SVM is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="gini",max_depth=5)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=gini,max_depth=5")
    

def DT_func_3(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="gini",max_depth=20)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=gini,max_depth=20")


def DT_func_4(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="gini",max_depth=50)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=gini,max_depth=50")


def DT_func_5(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=50")

def DT_func_6(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=20)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=20")



def DT_func_7(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,min_samples_split=5)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=5,min_samples_split=5")




def DT_func_8(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,min_samples_split=10)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=5,min_samples_split=10")



def DT_func_9(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,max_features=10)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=5,max_features=10")



def DT_func_10(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,max_features=15)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=5,max_features=15")





#call to decision tree with different parameters and features
DT_func_base(x_train, x_test, y_train, y_test )
DT_func_2(x_train, x_test, y_train, y_test,  )
DT_func_3(x_train, x_test, y_train, y_test,  )
DT_func_4(x_train, x_test, y_train, y_test,  )
DT_func_5(x_train, x_test, y_train, y_test,  )
DT_func_6(x_train, x_test, y_train, y_test,  )
DT_func_7(x_train, x_test, y_train, y_test,  )
DT_func_8(x_train, x_test, y_train, y_test,  )
DT_func_9(x_train, x_test, y_train, y_test,  )
DT_func_10(x_train, x_test, y_train, y_test,  )


#pruning Decision tree
def DT_prune_1(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,ccp_alpha=0.1)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=5,max_features=15")

#pruning Decision tree
def DT_prune_2(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,ccp_alpha=0.01)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=5,max_features=15")


#pruning Decision tree
def DT_prune_3(train_data,test_data,train_label,test_label ):
    
 
    print("------------Decision tree  is starting-------------- ")
    DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,ccp_alpha=0.001)
    DT.fit(train_data, train_label)
    y_pred = DT.predict(test_data)
    plot_results(y_pred,test_label,"decision tree with: criterion=entropy,max_depth=5,max_features=15")


DT_prune_1(x_train, x_test, y_train, y_test,  )
DT_prune_2(x_train, x_test, y_train, y_test,  )
DT_prune_3(x_train, x_test, y_train, y_test,  )


# clf = DecisionTreeClassifier(random_state=0)
# path = clf.cost_complexity_pruning_path(x_train, y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities


 
#different decision tree types:
#https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart
#decision tree pruning:
#https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

#While utilizing a decision tree algorithm, it is not essential to standardize or normalize the data that has been collected. It can handle both continuous and categorical variables.
#The execution of a Decision tree algorithm must be possible without having to scale the data as well.
#While utilizing the decision tree algorithm, it is not necessary to credit the missing values.
#Unlike the traditional pre-processing steps of data, the pre-processing steps require lesser coding and analysis in a decision tree making model.
#Unlike the traditional pre-processing steps of data, the pre-processing steps are time-saving in a decision tree making model.
#The Tree is reasonably interpretable
#A Decision tree model is very intuitive and easy to explain to technical teams as well as stakeholders
 
 
#########################################################################
#                             Random Forest                             #
#########################################################################
 
from sklearn.ensemble import RandomForestClassifier


print("------------Random Forest is starting-------------- ")
random_forest = RandomForestClassifier(criterion="entropy",max_depth=20,n_estimators=300 )
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
plot_results(y_pred,y_test,"  Random Forest with: criterion=entropy,max_depth=5, n_estimators=200")



#########################################################################
#                         Bitcoin Data analysis                         #
#########################################################################
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

#read bitcoin data
bit_data_path="bitcoin_data.csv"
bit_data=read_bit_data(bit_data_path)

#change date column format
bit_data["Date"]=bit_data["Date"].apply(lambda date:datetime.strptime(date, '%b %d, %Y').strftime('%Y-%m-%d')) 



#preprocess data

bit_data["Price"]=pd.to_numeric(bit_data["Price"])


bit_data["Vol."]=bit_data["Vol."].apply(lambda x:x.split(sep="K")[0] )
bit_data["Vol."]=bit_data["Vol."].apply(lambda x:x.split(sep="M")[0])
bit_data.drop(bit_data[bit_data["Vol."]=="-"].index,inplace=True)
bit_data["Vol."]=bit_data["Vol."].apply(lambda x: float(x) )
bit_data["Change %"]=bit_data["Change %"].apply(lambda x:x.split(sep="%")[0] )
bit_data["Change %"]=bit_data["Change %"].apply(lambda x: float(x) )


#Split data into train and test sets
train_data_idx=bit_data.index[bit_data["Date"]=="2020-01-01"].tolist()[0]
bit_data.drop("Date",axis=1,inplace=True)

scaler = MinMaxScaler()
scaler.fit(bit_data)
bit_data_normalized=scaler.transform(np.array(bit_data))
bit_data_normalized=pd.DataFrame(data=bit_data_normalized,columns=bit_data.columns)

bit_train_data=bit_data_normalized.iloc[train_data_idx:,:]
bit_train_label=bit_train_data["Price"].values
bit_train_data.drop("Price",axis=1,inplace=True)
bit_test_data=bit_data_normalized.iloc[:train_data_idx,:]
bit_test_label=bit_test_data["Price"].values
bit_test_data.drop("Price",axis=1,inplace=True)
bit_train_data=np.array(bit_train_data)
bit_test_data=np.array(bit_test_data)
bit_train_label=np.array(bit_train_label)
bit_test_label=np.array(bit_test_label)




def get_acc(y_real,y_pred):
   y_real=np.array(y_real).reshape(-1)
   y_pred=np.array(y_pred).reshape(-1)
   diff=np.abs(y_real-y_pred)
   threshold=np.abs(0.1*y_real)
   true_inst=diff[diff<threshold]
   acc=(len(true_inst)/len(y_pred))*100
   return acc

def plot_result_regression(rs_mse,rs_acc,rs_r2,title) :
    results=[round(np.mean(rs_mse),2),round(np.mean(rs_acc),2),round(np.mean(rs_r2),2)]
    plt.figure(figsize=(10,8),dpi=100)
    plt.bar(["mse-mean ","acc-mean","r2-mean"] ,results,color="byr")
    for index,value in enumerate(results):
        plt.text(index,value,str(value))
    plt.title(title)
    # plt.xticks(rotation=90)
    if not os.path.exists("home-output"):
        os.makedirs("home-output")
    plt.savefig("home-output/"+title)
    plt.show()
    
    
def linear_regression_func(train_data,test_data,train_label,test_label ):
    print("------------linear regression is starting-------------- ")
    regr = LinearRegression()
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])

    np.savetxt("bit_output/linear_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")
 
def ridge_regression_func(train_data,test_data,train_label,test_label ):
    print("------------ridge regression is starting-------------- ")
    regr = Ridge(alpha=1.0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])

    np.savetxt("bit_output/ridge_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")
 

def lasso_regression_func(train_data,test_data,train_label,test_label ):
    print("------------lasso regression is starting-------------- ")
    regr = linear_model.Lasso(alpha=1.0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/lasso_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")
 
 

def knn_regression_func(train_data,test_data,train_label,test_label ):
    print("------------knn regression is starting-------------- ")
    regr =KNeighborsRegressor(n_neighbors=5)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/knn_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")
 
    
 
def dt_regression_func(train_data,test_data,train_label,test_label ):
    print("------------decision tree regression is starting-------------- ")
    regr =DecisionTreeRegressor(max_depth=7)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/decision tree_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")
 
def mlp_regression_func(train_data,test_data,train_label,test_label ):
    print("------------mlp regression is starting-------------- ")
    regr =MLPRegressor(random_state=1, max_iter=300)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/mlp_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  
 
def random_frst_regression_func(train_data,test_data,train_label,test_label ):
    print("------------random forest regression is starting-------------- ")
    regr =RandomForestRegressor(max_depth=10, random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/random_forest_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

 
    
def bagging_regression_func(train_data,test_data,train_label,test_label ):
    print("------------ bagging regression is starting-------------- ")
    regr =BaggingRegressor(n_estimators=300, random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/bagging_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

def voting_regression_func(train_data,test_data,train_label,test_label ):
    print("------------voting regression is starting-------------- ")
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    regr = VotingRegressor([('lr', r1), ('rf', r2)])
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/voting_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  


def adaboost_regression_func(train_data,test_data,train_label,test_label ):
    print("------------adaboost regression is starting-------------- ")
 
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/adaboost_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

def boosting_regression_func(train_data,test_data,train_label,test_label ):
    print("------------boosting regression is starting-------------- ")
 
    regr =GradientBoostingRegressor(random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/boosting_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  




linear_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
ridge_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
lasso_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
knn_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
mlp_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
dt_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
random_frst_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
adaboost_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
voting_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
bagging_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
boosting_regression_func(bit_train_data,bit_test_data,bit_train_label,bit_test_label)


#########################################################################################
#                       running ensemble learnuing with best models                     #
#########################################################################################


def bagging_model(train_data,test_data,train_label,test_label ):
    print("------------ bagging regression is starting-------------- ")
    regr =BaggingRegressor (MLPRegressor(random_state=1, max_iter=200),n_estimators=100, random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/ensemble_bagging_using_mlp_regression.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

def voting_model(train_data,test_data,train_label,test_label ):
    print("------------voting regression is starting-------------- ")
    r1 = LinearRegression()
    r2 = Ridge(alpha=1.0)
    regr = VotingRegressor([('lr', r1), ('rf', r2)])
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/ensemble_voting_using_linear_and_ridge_regression.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  


def boosting_model(train_data,test_data,train_label,test_label ):
    print("------------boosting regression is starting-------------- ")
 
    regr =GradientBoostingRegressor(learning_rate=0.5,random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/ensemble_boosting_regression_result_on_bit.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  



voting_model(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
bagging_model(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
boosting_model(bit_train_data,bit_test_data,bit_train_label,bit_test_label)



#########################################################################################
#                       running adaboost with different parameters                      #
#########################################################################################

def adaboost_1(train_data,test_data,train_label,test_label ):
    print("------------adaboost regression is starting-------------- ")
 
    regr = AdaBoostRegressor(random_state=0 )
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/adaboost_with_different_parameters/difault_parameters.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  



def adaboost_2(train_data,test_data,train_label,test_label ):
    print("------------adaboost regression is starting-------------- ")
 
    regr = AdaBoostRegressor(base_estimator=LinearRegression(),random_state=0 )
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/adaboost_with_different_parameters/base_estimator_LinearRegression().csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter="," )  


def adaboost_3(train_data,test_data,train_label,test_label ):
    print("------------adaboost regression is starting-------------- ")
 
    regr = AdaBoostRegressor(learning_rate=0.001,random_state=0 )
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/adaboost_with_different_parameters/learning_rate_001.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter="," )  


def adaboost_4(train_data,test_data,train_label,test_label ):
    print("------------adaboost regression is starting-------------- ")
 
    regr = AdaBoostRegressor(loss="exponential",random_state=0 )
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/adaboost_with_different_parameters/loss_square.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter="," )  

def adaboost_5(train_data,test_data,train_label,test_label ):
    print("------------adaboost regression is starting-------------- ")
 
    regr = AdaBoostRegressor(n_estimators=200,random_state=0 )
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/adaboost_with_different_parameters/n_estimators_200.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter="," )  


adaboost_1(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
adaboost_2(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
adaboost_3(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
adaboost_4(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
adaboost_5(bit_train_data,bit_test_data,bit_train_label,bit_test_label)



#########################################################################################
#                      running  random forest with different parameters                 #
#########################################################################################

def random_frst_1(train_data,test_data,train_label,test_label ):
    print("------------random forest regression is starting-------------- ")
    regr =RandomForestRegressor( max_depth=12, random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/random_forest_with_different_parameters/default_parameters.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

def random_frst_2(train_data,test_data,train_label,test_label ):
    print("------------random forest regression is starting-------------- ")
    regr =RandomForestRegressor( max_depth=3, random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/random_forest_with_different_parameters/max_deep_3.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  
    

def random_frst_3(train_data,test_data,train_label,test_label ):
    print("------------random forest regression is starting-------------- ")
    regr =RandomForestRegressor( max_depth=9, random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/random_forest_with_different_parameters/max_deep_9.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

def random_frst_4(train_data,test_data,train_label,test_label ):
    print("------------random forest regression is starting-------------- ")
    regr =RandomForestRegressor( max_features="sqrt", random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/random_forest_with_different_parameters/max_features_sqrt.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

def random_frst_5(train_data,test_data,train_label,test_label ):
    print("------------random forest regression is starting-------------- ")
    regr =RandomForestRegressor(n_estimators=300, random_state=0)
    regr.fit(train_data, train_label)
    y_pred = regr.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt("bit_output/random_forest_with_different_parameters/n_estimators_300.csv",out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  

def xgb(train_data,test_data,train_label,test_label,f_name):
    xgb=XGBRegressor(booster="gblinear"  )
    xgb.fit(train_data,train_label)
    y_pred=xgb.predict(test_data)
    mse=metrics.mean_squared_error(test_label, y_pred)
    rmse=metrics.mean_squared_error(test_label, y_pred,squared=False)
    mae= metrics.mean_absolute_error(test_label,y_pred)
    r2=r2_score(test_label, y_pred)
    print('Mean squared error (MSE): %.3f ' % mse)  
    print('Root mean square error (RMSE) : %.3f '%rmse) 
    print('Mean Absolute Error (MAE): %.3f'%mae)
    print('R2 score : %.2f ' %r2 )
    acc=get_acc(test_label, y_pred)
    print("Accuracy is: %.3f" %acc)
    out=np.array([round(mse,3),round(rmse,3),round(mae,3),round(r2,3),round(acc,3)])
    np.savetxt(f_name,out.reshape(1,-1) , header="mse,rmse,mae,r2,accuracy", delimiter=",")  


random_frst_1(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
random_frst_2(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
random_frst_3(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
random_frst_4(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
random_frst_5(bit_train_data,bit_test_data,bit_train_label,bit_test_label)
xgb(bit_train_data,bit_test_data,bit_train_label,bit_test_label,"bit_output/XGBOOST.csv")


######################################################################
#                       adding new feature                           #
######################################################################
variation=[]

for i in range(1,len(bit_data)):
    if (bit_data.iloc[i-1,0]>bit_data.iloc[i,0]):
        variation.append(1)
    elif (bit_data.iloc[i-1,0]==bit_data.iloc[i,0]):
        variation.append(0)
    else:
      variation.append(-1)
variation.append(0)   
bit_data["variation"]=variation


scaler = MinMaxScaler()
scaler.fit(bit_data)
bit_data_normalized=scaler.transform(np.array(bit_data))
bit_data_normalized=pd.DataFrame(data=bit_data_normalized,columns=bit_data.columns)

bit_train_data=bit_data_normalized.iloc[train_data_idx:,:]
bit_train_label=bit_train_data["Price"].values
bit_train_data.drop("Price",axis=1,inplace=True)
bit_test_data=bit_data_normalized.iloc[:train_data_idx,:]
bit_test_label=bit_test_data["Price"].values
bit_test_data.drop("Price",axis=1,inplace=True)
bit_train_data=np.array(bit_train_data)
bit_test_data=np.array(bit_test_data)
bit_train_label=np.array(bit_train_label)
bit_test_label=np.array(bit_test_label)

xgb(bit_train_data,bit_test_data,bit_train_label,bit_test_label,"bit_output/XGBOOST_with_new_feature.csv")

 