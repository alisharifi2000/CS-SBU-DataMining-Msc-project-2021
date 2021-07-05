import pandas as pd
import numpy as np

df = pd.read_csv('./bitcoinPricesData.csv')
df = df.iloc[::-1,:]

df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
df = df[['Date', 'Price']]
df['Price'] = df['Price'].str.replace(',','').astype(float)
print(df)
