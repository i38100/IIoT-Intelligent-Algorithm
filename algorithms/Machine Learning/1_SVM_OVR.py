import numpy as np
import struct
import time
from PIL import Image


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


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


# 调整数值大于H或小雨L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 对于给定的alpha值，计算E值并返回
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 用来选择第二个alpha值并保证每次优化中采用最大步长
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0

    # 将输入的Ei在误差缓存中设置为已经计算好的
    oS.eCache[i] = [1, Ei]

    # 构建一个非零表
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        # 循环查找最大步长的Ej
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 如果是第一次循环，随机选择一个alpha值，并计算Ej
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 当alpha优化后更新误差缓存
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 核转换函数
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1.0*kTup[1]**2))
    return K


# 数据及参数存储
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


# 寻找决策边界的优化例程函数
def innerL(i, oS):
    Ei = calcEk(oS, i)

    # 根据误差Ei判断alpha是否可以被优化
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择第二个alpha值
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # 保证alpha值范围在0-C区间
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            return 0

        # 计算alpha[j]的最优修改量
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            return 0

        # 对i进行修改，修改量与j相同，但方向相反
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)

        # 计算并设置常数项b
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        # alpha被优化则返回1
        return 1
    else:
        return 0


# 完整的SMO算法
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 1.3)):
    # 初始化`optStruct`，将输入参数存入数据对象
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    # 初始化控制函数退出的变量
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 进入主体while循环
    # 设置多个循环退出条件，例如迭代达到最大次数或遍历集合后未修改任何alpha值时候退出
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历任何可能的alpha值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        else:
            # 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        # 在完整遍历和非边界值遍历之间来回切换
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
    return oS.b, oS.alphas


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
    # 输入svm参数
    c = float(input('请输入参数c的值：'))
    sigma = float(input('请输入参数sigma的值：'))

    # 初始化
    b_sv = [0]*10
    sv_index = [0]*10
    sv = [0]*10
    label_sv = [0]*10
    alpha_sv = [0]*10
    train_time = 0.0
    ktup = ('rbf', sigma)

    # 训练十个二分类器
    for i in range(10):
        # 将训练集划分两类，一对剩余
        index = np.concatenate((np.nonzero(train_label == i)[0], np.nonzero(train_label != i)[0]))  # 分成两类
        tmp_label = np.array([train_label[k] for k in index])
        traindata = np.array([train_data[k] for k in index])
        trainlabel = np.array([-1 if k != i else 1 for k in tmp_label])

        # 训练
        start_train = time.time()
        b, alphas = smoP(traindata, trainlabel, c, 0.0001, 10000, ktup)
        end_train = time.time()
        train_time += end_train - start_train

        # 记录该分类器参数
        data_mat = np.mat(traindata)
        label_mat = np.mat(trainlabel).transpose()
        b_sv[i] = b
        sv_index = np.nonzero(alphas.A > 0)[0]
        sv[i] = data_mat[sv_index]
        label_sv[i] = label_mat[sv_index]
        alpha_sv[i] = alphas[sv_index]

    # 每个测试数据在10个分类器上测试
    testdata = np.mat(test_data)
    testlabel = np.mat(test_label).transpose()
    m = testdata.shape[0]
    acc = 0
    start_test = time.time()
    for i in range(m):
        predict = [-1]*10
        for j in range(10):
            kernelEval = kernelTrans(sv[j], testdata[i], ktup)
            predict[j] = kernelEval.T * np.multiply(label_sv[j], alpha_sv[j]) + b_sv[j]
        if np.argmax(predict) == testlabel[i]:
            acc += 1
    end_test = time.time()
    acc_rate = acc / m

    # 输出结果
    print("\n训练过程所需时间为: %f 秒" % train_time)
    print("测试过程所需时间为: %f 秒" % (end_test - start_test))
    print("测试结果的正确率为: %f" % acc_rate)

    # 识别自己的简笔画
    print("\n开始识别自己的10幅简笔画...")
    for i in range(10):
        im1 = pre_pic('../my_picture/%d.jpg' % i)
        im1 = im1.reshape((1, 28*28))
        predict = [-1]*10
        for j in range(10):
            kernelEval = kernelTrans(sv[j], testdata[i], ktup)
            predict[j] = kernelEval.T * np.multiply(label_sv[j], alpha_sv[j]) + b_sv[j]
        if np.argmax(predict) == testlabel[i]:
            print('标签为 %d 的简笔画识别成功' % i)
        else:
            print('标签为 %d 的简笔画识别失败' % i)


if __name__ == '__main__':
    main()
