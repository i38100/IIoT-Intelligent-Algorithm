import time
import numpy as np
import struct
from PIL import Image


# 读取训练数据
def get_data(file_name):
    file_handle = open(file_name, 'rb')
    file_buffer = file_handle.read()

    head = struct.unpack_from('>IIII', file_buffer, 0)
    offset = struct.calcsize('>IIII')
    data_num = head[1]
    row_num = head[2]
    col_num = head[3]

    data_size = row_num * col_num
    data_vec = np.empty((data_num, data_size))
    fmt = '>' + str(data_size) + 'B'
    for i in range(data_num):
        data_vec[i] = np.array(struct.unpack_from(fmt, file_buffer, offset))
        offset += struct.calcsize(fmt)
    return data_vec


# 读取训练标签
def get_label(file_name):
    file_handle = open(file_name, 'rb')
    file_buffer = file_handle.read()

    head = struct.unpack_from('>II', file_buffer, 0)
    offset = struct.calcsize('>II')
    label_num = head[1]

    fmt = '>' + str(label_num) + 'B'
    label_vec = np.array(struct.unpack_from(fmt, file_buffer, offset))
    return label_vec


# sigmoid函数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# sigmoid函数的导数
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


# 预处理自己简笔画
def pre_pic(pic_name):
    img = Image.open(pic_name)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert("L"))
    return im_arr


# 定义BP网络类
class BP(object):
    def __init__(self, sizes):
        # 例如sizes=[2,3,2]表示输入层有两个神经元，隐藏层有3个神经元，输出层有2个神经元
        # num_layers代表神经网络数目
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 初始输入层，随机产生每层中的y个神经元的biase值，标准正态分布
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # 向前传播，加权求和再加偏置
    def feedforward(self, x):
        a = x.reshape(x.shape[0], 1)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    # 后向传播误差
    def backprop(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        x1 = x.reshape(x.shape[0], 1)
        # 记录输入和输出
        Outs = [x]
        Ins = []
        for b, w in zip(self.biases, self.weights):
            In = np.dot(w, x1) + b
            Ins.append(In)
            x1 = sigmoid(In)
            Outs.append(x1)
        # 求输出层δ的值
        e = np.zeros((10, 1))
        e[y] = 1.0
        delta = (Outs[-1]-e) * sigmoid_prime(Ins[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, Outs[-2].transpose())
        # 求隐层δ值，利用k+1层的δ值来计算第k层的δ值
        for k in range(2, self.num_layers):
            In = Ins[-k]
            delta = np.dot(self.weights[-k+1].transpose(), delta) * sigmoid_prime(In)
            delta_b[-k] = delta
            delta_w[-k] = np.dot(delta, Outs[-k-1].reshape(1, Outs[-k-1].shape[0]))
        return (delta_b, delta_w)

    # 批量梯度下降
    def BGD(self, x_train, y_train, x_test, y_test, epochs, batch_size, eta):
        n_train = x_train.shape[0]
        # 分成小批，更新w和b
        for j in range(epochs):
            batches_x = [x_train[k:k+batch_size] for k in range(0, n_train, batch_size)]
            batches_y = [y_train[k:k+batch_size] for k in range(0, n_train, batch_size)]
            for (batch_x, batch_y) in zip(batches_x, batches_y):
                delta_b = [np.zeros(b.shape) for b in self.biases]
                delta_w = [np.zeros(w.shape) for w in self.weights]
                for (x, y) in zip(batch_x, batch_y):
                    tmp_b, tmp_w = self.backprop(x, y)
                    # 累加存储偏导值
                    delta_b = [db+tb for db, tb in zip(delta_b, tmp_b)]
                    delta_w = [dw+tw for dw, tw in zip(delta_w, tmp_w)]
                # 更新，勿忘除样本大小
                self.biases = [b-(eta/batch_x.shape[0])*db for b, db in zip(self.biases, delta_b)]
                self.weights = [w-(eta/batch_x.shape[0])*dw for w, dw in zip(self.weights, delta_w)]
            # 输出测试每轮结束后，神经网络的准确度
            acc_rate = self.evaluate(x_test, y_test)
            print("Epoch {0}: {1}{2}".format(j, '测试集上识别正确率为 ', acc_rate))

    # 评估在测试集的正确率
    def evaluate(self, x_test, y_test):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in zip(x_test, y_test)]
        res = sum(int(x == y) for (x, y) in results)
        return (res/y_test.shape[0])

    # 识别自己的简笔画
    def my_test(self):
        for i in range(10):
            im1 = pre_pic('../my_picture/%d.jpg' % i)
            im1 = im1.reshape((28*28))
            im1 = im1.astype('float32')/255
            label = np.argmax(self.feedforward(im1))
            if label == i:
                print('标签为 %d 的简笔画识别成功' % i)
            else:
                print('标签为 %d 的简笔画识别失败' % i)


if __name__ == "__main__":
    # 读取数据
    train_data = get_data("../data/train-images-idx3-ubyte")
    test_data = get_data("../data/t10k-images-idx3-ubyte")
    train_label = get_label("../data/train-labels-idx1-ubyte")
    test_label = get_label("../data/t10k-labels-idx1-ubyte")
    train_num = int(input("请输入训练数据的个数: "))
    test_num = int(input("请输入测试数据的个数: "))
    train_data = train_data[: train_num]
    test_data = test_data[: test_num]
    train_label = train_label[: train_num]
    test_label = test_label[: test_num]
    train_label = train_label.reshape(train_label.shape[0], 1)
    test_label = test_label.reshape(test_label.shape[0], 1)
    # 归一化
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255.0
    test_data /= 255.0
    # 定义网络实例并训练测试
    net = BP([784, 80, 80, 10])
    start_time = time.time()
    net.BGD(train_data, train_label, test_data, test_label, 10, 10, 3.0)
    end_time = time.time()
    # 输出结果并识别自己的简笔画
    print("\n整个过程所需时间为: %f 秒" % (end_time - start_time))
    print("\n开始识别自己的10幅简笔画...")
    net.my_test()
