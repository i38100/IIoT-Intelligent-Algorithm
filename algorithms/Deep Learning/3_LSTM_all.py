import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler


def feature_vector(df, feature, N):
    rows = df.shape[0]
    column_n = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    column_name = "{}_{}".format(feature, N)
    df[column_name] = column_n


# 读取csv文件删除不用的列
mycsv = pd.read_csv('../data/yiqing.csv')
mycsv = mycsv.drop(['date', 'tot_con', 'tot_sus', 'death', 'heal'], axis=1)
rows = mycsv.shape[0]  # 数据行数
column = mycsv.shape[1]  # 列数

# 数据清洗，删除中间异常行以及首位较小的值
del_row = [0, rows-1]
for row in range(1, rows - 1):
    if mycsv.iloc[row, 0] > (mycsv.iloc[row - 1, 0] + mycsv.iloc[row + 1, 0]):
        del_row.append(row)
    if mycsv.iloc[row, 0] < 30:
        del_row.append(row)
mycsv = mycsv.drop(del_row)
mycsv = mycsv[:].reset_index(drop=True)

# 重新组织数据
n = 2
for feature in ['new_con', 'new_sus']:
    for i in range(1, n):
        feature_vector(mycsv, feature, n-i)
mycsv = mycsv.drop(['new_sus'], axis=1)
mycsv = mycsv[n - 1:].reset_index(drop=True)
print(mycsv.head())

# 设置输入x与y，转化为浮点数且归一化
x = mycsv.drop('new_con', axis=1)
y = mycsv.drop(['new_con_1', 'new_sus_1'], axis=1)
datax = x.values
datay = y.values
# print("X: ", datax)
# print("Y: ", datay)
datax = datax.astype('float32')
datay = datay.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
datax = scaler.fit_transform(datax)
datay = scaler.fit_transform(datay)
# print("X: ", datax)
# print("Y: ", datay)
datax = np.array(datax)
datay = np.array(datay)
datax = np.reshape(datax, (datax.shape[0], datax.shape[1], 1))

# 搭建模型，编译拟合
model = Sequential()
model.add(LSTM(3, input_shape=(None, 1)))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
start_train = time.time()
model.fit(datax, datay, epochs=100, batch_size=1, verbose=2)
end_train = time.time()

# 对训练效果进行评价，计算训练误差，注意平移现象
start_test = time.time()
pred = model.predict(datax)
end_test = time.time()
pred = pred[1:]
datay = datay[:-1]
# 反归一化
pred = scaler.inverse_transform(pred)
datay = scaler.inverse_transform(datay)
err_rate = np.sum(np.abs(pred - datay) / (datay + pred)) / datay.shape[0]

# 输出结果
print("\n结果如下所示：")
print("训练的误差为: %f" % err_rate)
print("训练过程所需时间为: %f 秒" % (end_train - start_train))
print("测试过程所需时间为: %f 秒" % (end_test - start_test))
# 画图显示拟合效果
plt.figure(figsize=(10, 10))
plt.title("LSTM")
plt.plot(datay, marker='o', label='real data')
plt.plot(pred, marker='*', label='predict')
plt.legend()
plt.show()
