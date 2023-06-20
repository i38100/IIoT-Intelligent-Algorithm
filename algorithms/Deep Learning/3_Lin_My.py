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


# 损失函数
def costfunc(X, Y, theta):
    inner = np.power((X*theta.T)-Y, 2)
    return np.sum(inner)/(2*len(X))


# 梯度下降
def GD(X, Y, theta, alpha, iters):
    temp = np.mat(np.zeros(theta.shape))
    cost = np.zeros(iters)
    thetaCol = theta.shape[1]
    for i in range(iters):
        err = (X*theta.T-Y)
        for j in range(thetaCol):
            derivativeInner = np.multiply(err, X[:, j])
            temp[0, j] = theta[0, j] - (alpha*np.sum(derivativeInner)/len(X))
        theta = temp
        cost[i] = costfunc(X, Y, theta)
    return theta, cost


# 读取csv文件
mycsv = pd.read_csv('../data/yiqing.csv')
# print(mycsv.head())
# 画出所有属性的图像

# 删除不用的列
mycsv = mycsv.drop(['date', 'tot_con', 'tot_sus', 'death', 'heal'], axis=1)
rows = mycsv.shape[0]  # 数据行数
column = mycsv.shape[1]  # 列数

# 数据清洗，删除中间异常行以及首尾较小的值
del_row = [0, rows-1]
for row in range(1, rows - 1):
    if mycsv.iloc[row, 0] > (mycsv.iloc[row - 1, 0] + mycsv.iloc[row + 1, 0]):
        del_row.append(row)
    if mycsv.iloc[row, 0] < 30:
        del_row.append(row)
# print(del_row)
mycsv = mycsv.drop(del_row)
mycsv = mycsv[:].reset_index(drop=True)

# 重新组织表格，前4天数据作为x，今天新增确诊作为y
n = 5
for feature in ['new_sus']:
    for i in range(1, n):
        feature_vector(mycsv, feature, n-i)
mycsv = mycsv.drop(['new_sus'], axis=1)
mycsv = mycsv[n - 1:].reset_index(drop=True)
mycsv.insert(1, 'ones', 1)  # 第二列插入全1
# print(mycsv.head())

# 划分数据
cols = mycsv.shape[1]
X = mycsv.iloc[:, 1:cols]
Y = mycsv.iloc[:, 0:1]
X = np.mat(X.values)
Y = np.mat(Y.values)

# 归一化及数据划分
for i in range(1, 5):
    X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))

'''
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
'''
# 十次随机取平均
t_train = 0
t_test = 0
error = 0
plt.figure(figsize=(10, 10))
for i in range(1, 11):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
    # 初始化参数
    theta = np.mat([0, 0, 0, 0, 0])
    iters = 10000
    alpha = 0.004
    # 训练，画出损失值变化趋势
    start_train = time.time()
    Theta, cost = GD(x_train, y_train, theta, alpha, iters)
    end_train = time.time()
    print("theta值为：", Theta)
    '''
    plt.title("cost")
    plt.plot(cost, marker='*', label='cost')
    plt.legend()
    plt.show()
    '''
    # 测试，画图，计算正确率
    start_test = time.time()
    pred = x_test*Theta.T
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
print("测试结果的平均误差为: ", error/10)
print("训练过程所需平均时间为: ", t_train/10, "秒")
print("测试过程所需平均时间为: ", t_test/10, "秒")
