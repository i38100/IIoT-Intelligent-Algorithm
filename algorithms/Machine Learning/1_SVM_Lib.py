import numpy as np
import struct
import time
from sklearn.svm import SVC
from PIL import Image
from sklearn import metrics


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

    # 输入训练及测试的数据个数，取出来并且归一化
    train_num = int(input("请输入训练数据的个数: "))
    test_num = int(input("请输入测试数据的个数: "))
    train_data = train_data[: train_num]
    test_data = test_data[: test_num]
    train_label = train_label[: train_num]
    test_label = test_label[: test_num]
    train_data = train_data/255.0
    test_data = test_data/255.0
    # 输入svm的参数
    c = float(input('请输入参数c的值：'))
    g = float(input('请输入参数gamma的值：'))
    # clf = SVC(probability=False, kernel="linear")
    clf = SVC(probability=False, kernel="rbf", C=c, gamma=g)

    # 训练过程
    start_train = time.time()
    clf.fit(train_data, train_label)
    end_train = time.time()

    # 测试过程
    start_test = time.time()
    predicted = clf.predict(test_data)
    end_test = time.time()

    # 输出结果
    print("\n结果如下所示：")
    print("训练过程所需时间为: %f 秒" % (end_train - start_train))
    print("测试过程所需时间为: %f 秒" % (end_test - start_test))
    print("测试结果的正确率为: %f" % metrics.accuracy_score(test_label, predicted))

    # 识别自己的10幅简笔画
    print("\n开始识别自己的10幅简笔画...")
    for i in range(10):
        im1 = pre_pic('../my_picture/%d.jpg' % i)
        im1 = im1.reshape((1, 28*28))
        im1 = im1.astype('float32')/255
        label = clf.predict(im1)
        if label == i:
            print('标签为 %d 的简笔画识别成功' % i)
        else:
            print('标签为 %d 的简笔画识别失败' % i)


if __name__ == '__main__':
    main()
