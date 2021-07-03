import pandas as pd
import numpy as np


def cal_acc(y_pred, y):
    z = np.where((y * 1.05 > y_pred) & (y * 0.95 < y_pred), 1, 0)
    return np.count_nonzero(z == 1) / z.size


def return_dataset():
    df = pd.read_csv('bitcoinPricesData.csv')
    df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')

    for i in range(1, 101):
        df['last_price' + str(i)] = df['Price'].shift(periods=-1 * i, fill_value=0)

    df_train_vals = []
    df_all_vals = ['Price']
    for i in range(1, 101):
        df_train_vals.append('last_price' + str(i))
        df_all_vals.append('last_price' + str(i))

    df_test = df.iloc[:600, :]
    df_train = df.iloc[600:, :]

    y_test = df_test[['Price']].to_numpy().reshape((len(df_test),))
    y_tr = df_train[['Price']].to_numpy().reshape((len(df_train),))

    x_test = df_test[df_train_vals].to_numpy()
    x_tr = df_train[df_train_vals].to_numpy()
    
    return y_test, y_tr, x_test, x_tr
