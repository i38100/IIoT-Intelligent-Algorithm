import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def feature_vector(df, feature, N):
    rows = df.shape[0]
    column_n = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    column_name = "{}_{}".format(feature, N)
    df[column_name] = column_n


mycsv = pd.read_csv('../data/yiqing.csv')
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
mycsv = mycsv.drop(del_row)
mycsv = mycsv[:].reset_index(drop=True)

n = 5
for feature in ['new_con', 'new_sus']:
    for i in range(1, n):
        feature_vector(mycsv, feature, n-i)
mycsv = mycsv.drop(['new_sus'], axis=1)
mycsv = mycsv[n-1:].reset_index(drop=True)
x = np.array(mycsv.drop('new_con', axis=1))
y = np.array(mycsv['new_con'])
# print(mycsv.head())

# 十次随机取平均
t_train = 0
t_test = 0
error = 0
plt.figure(figsize=(10, 10))
for j in range(1, 11):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=j)
    y_train = y_train.reshape([-1, 1])
    y_test = y_test.reshape([-1, 1])
    '''
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    '''

    # 定义三个隐层
    n1 = 60
    n2 = 100
    n3 = 50
    X = tf.placeholder(tf.float32, shape=[None, 8])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    w1 = tf.Variable(tf.random_normal([8, n1], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1), [n1])
    o1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    w2 = tf.Variable(tf.random_normal([n1, n2], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1), [n2])
    o2 = tf.nn.relu(tf.matmul(o1, w2) + b2)
    w3 = tf.Variable(tf.random_normal([n2, n3], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1), [n3])
    o3 = tf.nn.relu(tf.matmul(o2, w3) + b3)
    w4 = tf.Variable(tf.random_normal([n3, 1], stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1), [1])
    pred = tf.matmul(o3, w4) + b4

    # 定义损失函数以及误差
    loss = tf.reduce_sum(tf.pow(pred-Y, 2))/y_train.shape[0]
    err_train = tf.reduce_sum(tf.abs(tf.abs(pred)-Y)/(tf.abs(pred)+Y))/y_train.shape[0]
    err_test = tf.reduce_sum(tf.abs(tf.abs(pred)-Y)/(tf.abs(pred)+Y))/y_test.shape[0]

    # 训练并测试模型
    optimizer = tf.train.AdamOptimizer(0.004).minimize(loss)  # 0.001是learning rate
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # 训练
        print('\n开始第%d次运行中训练...' % j)
        start_train = time.time()
        for i in range(10000):
            train = sess.run([optimizer, loss, err_train], feed_dict={X: x_train, Y: y_train})
            if i % 1000 == 0:
                epoch = int(i/1000)
                # print("Epoch {0}: 损失函数值为{1}, 训练集上误差为{2}".format(epoch, train[1], train[2]))
        end_train = time.time()

        # 测试
        start_test = time.time()
        test = sess.run([err_test, pred], feed_dict={X: x_test, Y: y_test})
        end_test = time.time()
        plt.subplot(2, 5, j)
        plt.title("BP")
        plt.plot(y_test, marker='o', label='y_test')
        plt.plot(test[1], marker='*', label='predict')
        plt.legend()
        t_train += (end_train - start_train)
        t_test += (end_test - start_test)
        print('第%d次运行中测试的误差为%f' % (j, test[0]))
        error += test[0]


# 输出结果
print("\n结果如下所示：")
print("测试结果的平均误差为: ", error/10)
print("训练过程所需平均时间为: ", t_train/10, "秒")
print("测试过程所需平均时间为: ", t_test/10, "秒")
plt.show()
