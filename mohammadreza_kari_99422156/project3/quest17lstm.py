import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import tensorflow as tf


def return_accuracy(y_pred, y):
    z = np.where((y * 1.1 >= y_pred) & (y * 0.90 <= y_pred), 1, 0)
    return np.count_nonzero(z == 1) / z.size


df = pd.read_csv('bitcoin_data.csv')
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df['last_p'] = df['Price'].shift(periods=-1, fill_value=0)

df_train_vals = []
df_all_vals = ['Price']

df_test = df.iloc[:550, :]
df_train = df.iloc[550:3300, :]

y_test = df_test[['Price']].to_numpy().reshape((len(df_test),))
y_tr = df_train[['Price']].to_numpy().reshape((len(df_train),))

x_test = df_test[['last_p']].to_numpy().reshape((len(df_test),))
x_tr = df_train[['last_p']].to_numpy().reshape((len(df_train),))

y_test_cat = np.where(y_test - x_test < 0, 0, 1)
y_train_cat = np.where(y_tr - x_tr < 0, 0, 1)

x_test = np.reshape(x_test, (1, 1, 550))
x_tr = np.reshape(x_tr, (5, 1, 550))

y_test = np.reshape(y_test, (1, 1, 550))
y_tr = np.reshape(y_tr, (5, 1, 550))

y_test_cat = np.reshape(y_test_cat, (1, 1, 550))
y_train_cat = np.reshape(y_train_cat, (5, 1, 550))

### regressor LSTM


model = Sequential()
model.add(LSTM(550, input_shape=(1, 550), return_sequences=True))
model.add(Dense(550))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_tr, y_tr, epochs=10000, batch_size=1, verbose=2, validation_data=(x_test, y_test))


predict = model.predict(x_test)
predict = np.reshape(predict, (550, ))
y_test = np.reshape(y_test, (550, ))

print('LSTM accuracy:', return_accuracy(predict, y_test))
print('LSTM RMSE:', mean_squared_error(y_test, predict, squared=False))

### categorial LSTM


model_cat = Sequential()
model_cat.add(LSTM(550, input_shape=(1, 550), return_sequences=True))
model_cat.add(Dense(550))
model_cat.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
model_cat.fit(x_tr, y_train_cat, epochs=400, batch_size=1, verbose=2, validation_data=(x_test, y_test_cat))

predict = model_cat.predict(x_test)
predict = np.reshape(predict, (550, ))
predict = np.where(predict < 0.5, 1, 0)
y_test_cat = np.reshape(y_test_cat, (550, ))

print('categorial LSTM accuracy:', return_accuracy(predict, y_test_cat))

