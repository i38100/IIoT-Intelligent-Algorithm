import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def feature_vector(df, feature, N):
    rows = df.shape[0]
    column_n = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    column_name = "{}_{}".format(feature, N)
    df[column_name] = column_n


# 读取csv文件
mycsv = pd.read_csv('../data/yiqing.csv')
print(mycsv.head())
# 画出所有属性的图像
plt.title("raw_data")
plt.plot(mycsv.index, mycsv['new_con'], label='new_con')
plt.plot(mycsv.index, mycsv['new_sus'], label='new_sus')
plt.plot(mycsv.index, mycsv['tot_con'], label='tot_con')
plt.plot(mycsv.index, mycsv['tot_sus'], label='tot_sus')
plt.plot(mycsv.index, mycsv['death'], label='death')
plt.plot(mycsv.index, mycsv['heal'], label='heal')
plt.legend()
plt.show()

# 删除不用的列并画图
plt.title("only new_con and new_sus")
mycsv = mycsv.drop(['date', 'tot_con', 'tot_sus', 'death', 'heal'], axis=1)
plt.plot(mycsv.index, mycsv['new_con'], label='new_con')
plt.plot(mycsv.index, mycsv['new_sus'], label='new_sus')
plt.legend()
plt.show()
rows = mycsv.shape[0]  # 数据行数
column = mycsv.shape[1]  # 列数
print(mycsv.head())

# 数据清洗，删除中间异常行以及首位较小的值，并画图像
del_row = [0, rows-1]
for row in range(1, rows - 1):
    if mycsv.iloc[row, 0] > (mycsv.iloc[row - 1, 0] + mycsv.iloc[row + 1, 0]):
        del_row.append(row)
    if mycsv.iloc[row, 0] < 30:
        del_row.append(row)
print(del_row)
mycsv = mycsv.drop(del_row)
mycsv = mycsv[:].reset_index(drop=True)
plt.title("cleaned data")
plt.plot(mycsv.index, mycsv['new_con'], label='new_con')
plt.plot(mycsv.index, mycsv['new_sus'], label='new_sus')
plt.legend()
plt.show()

# 重新组织表格，前x天数据作为X，今天新增确诊作为y
n = 5
for feature in ['new_sus']:
    for i in range(1, n):
        feature_vector(mycsv, feature, n-i)
mycsv = mycsv.drop(['new_sus'], axis=1)
mycsv = mycsv[n-1:].reset_index(drop=True)
print(mycsv.head())
x = np.array(mycsv.drop('new_con', axis=1))
y = np.array(mycsv['new_con'])

# 十次随机取平均
t_train = 0
t_test = 0
error = 0
plt.figure(figsize=(10, 10))
for i in range(1, 11):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
    '''
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    '''
    # 创建线性回归对象并训练
    Lin = LinearRegression()
    start_train = time.time()
    Lin.fit(x_train, y_train)
    end_train = time.time()
    # 测试，累计结果、画图直观比较
    start_test = time.time()
    pred = Lin.predict(x_test)
    end_test = time.time()
    plt.subplot(2, 5, i)
    plt.title("linear regression")
    plt.plot(y_test, marker='o', label='y_test')
    plt.plot(pred, marker='*', label='predict')
    plt.legend()
    t_train += (end_train - start_train)
    t_test += (end_test - start_test)
    err_rate = np.sum(np.abs(pred - y_test) / (y_test + pred)) / y_test.shape[0]
    error += err_rate
plt.show()

# 输出结果
print("\n结果如下所示：")
print("测试结果的误差为: ", error/10)
print("训练过程所需时间为: ", t_train/10, "秒")
print("测试过程所需时间为: ", t_test/10, "秒")
