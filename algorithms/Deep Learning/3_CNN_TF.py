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

for feature in ['new_con', 'new_sus']:
    for N in range(1, 5):
        feature_vector(mycsv, feature, N)
mycsv = mycsv.drop(['new_sus'], axis=1)
mycsv = mycsv[4:].reset_index(drop=True)
x = np.array(mycsv.drop('new_con', axis=1))
y = np.array(mycsv['new_con'])

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
    X = tf.placeholder(tf.float32, shape=[None, 8])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    X_data = tf.reshape(X, [-1, 8, 1])
    n1 = 10
    n2 = 10
    # 都是一维卷积
    # 第一个卷积层与池化层，大小为n1，方括号中是维度
    # 滤波器宽度，输入信号通道，输出信号通道，方括号中三维构成filter
    # 8*1->8*n1->4*n1
    con1_w = tf.Variable(tf.truncated_normal(shape=[2, 1, n1], stddev=0.1))  # 产生正态分布，shape是维度
    con1_b = tf.Variable(tf.constant(0.1, shape=[n1]))  # 全用0.1填充，shape是维度
    con1_o = tf.nn.relu(tf.nn.conv1d(X_data, con1_w, stride=1, padding='SAME') + con1_b)
    pool1_o = tf.nn.pool(con1_o, window_shape=[2], strides=[2], pooling_type="MAX", padding="SAME")
    # 第二个卷积层与池化层，大小为n2
    # 4*n1->4*n2->2*n2
    con2_w = tf.Variable(tf.truncated_normal(shape=[2, n1, n2], stddev=0.1))
    con2_b = tf.Variable(tf.constant(0.1, shape=[n2]))
    con2_o = tf.nn.relu(tf.nn.conv1d(pool1_o, con2_w, stride=1, padding='SAME') + con2_b)
    pool2_o = tf.nn.pool(con2_o, window_shape=[2], strides=[2], pooling_type="MAX", padding="SAME")
    # 第一个全连接层
    # 2*n2
    fc1_w = tf.Variable(tf.truncated_normal(shape=[2*n2, 10], stddev=0.1))
    fc1_b = tf.Variable(tf.constant(0.1, shape=[10]))
    pool2_flatten = tf.reshape(pool2_o, [-1, 2*n2])
    fc1_o = tf.nn.relu(tf.matmul(pool2_flatten, fc1_w) + fc1_b)
    # 第二个全连接层，大小为10
    fc2_w = tf.Variable(tf.truncated_normal(shape=[10, 1], stddev=0.1))
    fc2_b = tf.Variable(tf.constant(0.1, shape=[1]))
    pred = tf.matmul(fc1_o, fc2_w) + fc2_b  # 最终输出值

    # 定义损失函数以及误差
    loss = tf.reduce_sum(tf.pow(pred-Y, 2))/y_train.shape[0]
    err_train = tf.reduce_sum(tf.abs(tf.abs(pred)-Y)/(tf.abs(pred)+Y))/y_train.shape[0]
    err_test = tf.reduce_sum(tf.abs(tf.abs(pred)-Y)/(tf.abs(pred)+Y))/y_test.shape[0]

    # 训练并测试模型
    optimizer = tf.train.AdamOptimizer(0.012).minimize(loss)
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
        plt.title("CNN")
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
