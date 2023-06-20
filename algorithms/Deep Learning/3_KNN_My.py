import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def feature_vector(df, feature, N):
    rows = df.shape[0]
    column_n = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    column_name = "{}_{}".format(feature, N)
    df[column_name] = column_n


def myKNN(x_train, y_train, x_test, k):
    # 临时数组，用于后面排序时交换，不然会打乱原来顺序
    tmp = np.zeros((y_train.shape[0], 1))
    for i in range(y_train.shape[0]):
        tmp[i] = y_train[i]
    # 计算欧式距离
    x_train_row = x_train.shape[0]
    x_train_col = x_train.shape[1]
    dist1 = np.tile(x_test, (x_train_row)).reshape((x_train_row, x_train_col)) - x_train[:x_train_row]
    dist2 = dist1 ** 2
    dist3 = dist2.sum(axis=1)
    dist4 = dist3 ** 0.5
    # 冒泡排序
    for i in range(len(dist4) - 1):
        for j in range(len(dist4) - i - 1):
            if dist4[j] > dist4[j + 1]:
                dist4[j], dist4[j + 1] = dist4[j + 1], dist4[j]
                tmp[j], tmp[j + 1] = tmp[j + 1], tmp[j]
    ans = 0
    for i in range(k):
        ans += tmp[i]
    return ans/k


# 读取csv文件
mycsv = pd.read_csv('../data/yiqing.csv')
print(mycsv.head())
# 画出所有属性的图像

# 删除不用的列并画图
mycsv = mycsv.drop(['date', 'tot_con', 'tot_sus', 'death', 'heal'], axis=1)
rows = mycsv.shape[0]  # 数据行数
column = mycsv.shape[1]  # 列数
# print(mycsv.head())

# 数据清洗，删除中间异常行以及首位较小的值，并画图像
del_row = [0, rows-1]
for row in range(1, rows - 1):
    if mycsv.iloc[row, 0] > (mycsv.iloc[row - 1, 0] + mycsv.iloc[row + 1, 0]):
        del_row.append(row)
    if mycsv.iloc[row, 0] < 30:
        del_row.append(row)
# print(del_row)
mycsv = mycsv.drop(del_row)
mycsv = mycsv[:].reset_index(drop=True)

# 重新组织表格，前x天数据作为X，今天新增确诊作为y
n = 5
for feature in ['new_sus']:
    for i in range(1, n):
        feature_vector(mycsv, feature, n-i)
mycsv = mycsv.drop(['new_sus'], axis=1)
mycsv = mycsv[n - 1:].reset_index(drop=True)
# print(mycsv.head())
x = np.array(mycsv.drop('new_con', axis=1))
y = np.array(mycsv['new_con'])

# 输入参数
k = int(input("请输入参数k的值: "))

# 十次随机取平均
t = 0
error = 0
plt.figure(figsize=(10, 10))
for j in range(1, 11):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=j)
    err = 0
    predict = []
    start_time = time.time()
    for i in range(x_test.shape[0]):
        pred = myKNN(x_train, y_train, x_test[i], k)
        predict.append(pred)
        err += np.abs(pred - y_test[i])/(pred + y_test[i])
    end_time = time.time()
    plt.subplot(2, 5, j)
    plt.title("KNN")
    plt.plot(y_test, marker='o', label='y_test')
    plt.plot(predict, marker='*', label='predict')
    plt.legend()
    t += (end_time - start_time)
    error += err/y_test.shape[0]
plt.show()

# 输出结果
print("测试结果的平均误差为: ", error/10)
print("整个过程所需平均时间为: ", t/10, "秒")
