import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(0, 9, 1)
x = x.reshape(9, 1)
y = np.array([2, 1, 15, 4, 10, 3, 12, 8, 10])  # 양
y = y.reshape(9, 1)

xy = np.concatenate((x, y), axis=1)
df = pd.DataFrame(xy, columns=['hz', 'volumn'], )
#print(df.head())
up_value = 12
df1 = df.query('volumn > %d' %(up_value))
print(df1.head())
#print(df1['hz'])
#print(df1['hz'].values)

#input : 찾고싶은 array

def find_index(y, up_value):
    x_len = len(y)
    x = np.arange(0, x_len, 1)
    x = x.reshape(x_len, 1)
    xy = np.concatenate((x, y), axis=1)
    df = pd.DataFrame(xy, columns=['hz', 'volumn'], )
    df1 = df.query('volumn > upvalue')
    return df1['hz'].values


#y_peak = [10, 12, 15]

#y_peak1 = pd.Series(y_peak)
#y_peak2 = pd.Series(y)
#print(y_peak1.head())
#print(y_peak2.head())
#y_peak2.query()
# x_peak = y_peak2.where(y_peak2 == y_peak1)
# print(x_peak)
