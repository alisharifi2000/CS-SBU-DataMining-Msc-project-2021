import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import mean_squared_error


def return_accuracy(y_pred, y):
    z = np.where((y * 1.1 > y_pred) & (y * 0.90 < y_pred), 1, 0)
    return np.count_nonzero(z == 1) / z.size


df = pd.read_csv('bitcoin_data.csv')
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')


for i in range(1, 101):
    df['p'+str(i)] = df['Price'].shift(periods=-1 * i, fill_value=0)

df_train_vals = []
df_all_vals = ['Price']
for i in range(1, 101):
    df_train_vals.append('p'+str(i))
    df_all_vals.append('p' + str(i))

df_values = df[df_all_vals]


df_test = df.iloc[:550, :]
df_train = df.iloc[550:, :]

y_test = df_test[['Price']].to_numpy().reshape((len(df_test),))
y_tr = df_train[['Price']].to_numpy().reshape((len(df_train),))

x_test = df_test[df_train_vals].to_numpy().reshape((len(df_test), 10, 10, 1))
x_tr = df_train[df_train_vals].to_numpy().reshape((len(df_train), 10, 10, 1))


model = Sequential()
model.add(Conv2D(50, kernel_size=3, activation='softmax', input_shape=(10, 10, 1)))
model.add(Flatten())
model.add(Dense(20, activation='softmax'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_tr, y_tr, validation_data=(x_test, y_test), epochs=3)

predict_test = model.predict(x_test)
print(predict_test)
print(return_accuracy(predict_test, y_test))
print(mean_squared_error(y_test, predict_test, squared=False))

