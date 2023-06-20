import numpy as np
import struct
import time
# import matplotlib.pyplot as plt
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


# 返回投票结果
def myKNN(train_data, train_label, test_data, train_num, k):
    # 计算欧式距离
    train_data_row = train_num
    train_data_col = train_data.shape[1]
    dist1 = np.tile(test_data, (train_data_row)).reshape((train_data_row, train_data_col)) - train_data[:train_data_row]
    dist2 = dist1 ** 2
    dist3 = dist2.sum(axis=1)
    dist4 = dist3 ** 0.5

    dist_sort = dist4.argsort()
    label_count = np.zeros((10), np.int32)
    for i in range(k):
        valid_label = train_label[dist_sort[i]]
        label_count[valid_label] += 1
    return np.argmax(label_count)


def pre_pic(pic_name):
    img = Image.open(pic_name)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert("L"))
    return im_arr


def main():
    # 读取四个文件
    train_data = get_data("../data/train-images-idx3-ubyte")
    test_data = get_data("../data/t10k-images-idx3-ubyte")
    train_label = get_label("../data/train-labels-idx1-ubyte")
    test_label = get_label("../data/t10k-labels-idx1-ubyte")

    # 输入参数
    k = int(input("请输入参数k的值: "))
    train_num = int(input("请输入训练数据的个数: "))
    test_num = int(input("请输入测试数据的个数: "))

    # 执行主体过程
    acc = 0
    start_time = time.time()
    for i in range(test_num):
        result = myKNN(train_data, train_label, test_data[i], train_num, k)
        if result == test_label[i]:
            acc += 1.0
        train_data[train_num] = test_data[i]
        train_label[train_num] = test_label[i]
        train_num += 1
    end_time = time.time()
    acc_rate = acc / test_num

    print("\n结果如下所示:")
    print("k的值为: %d" % k)
    print("正确率为: %f" % acc_rate)
    print("整个过程所需时间为: %f 秒" % (end_time - start_time))

    # 识别自己的10幅简笔画
    print("\n开始识别自己的10幅简笔画...")
    for i in range(10):
        im1 = pre_pic('../my_picture/%d.jpg' % i)
        im1 = im1.reshape((1, 28*28))
        label = myKNN(train_data, train_label, im1, train_num, k)
        if label == i:
            print('标签为 %d 的简笔画识别成功' % i)
        else:
            print('标签为 %d 的简笔画识别失败' % i)


if __name__ == "__main__":
    main()
