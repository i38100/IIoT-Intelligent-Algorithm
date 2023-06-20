import time
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import matplotlib.pyplot as plt


# 预处理自己的简笔画
def pre_pic(pic_name):
    img = Image.open(pic_name)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert("L"))
    return im_arr


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 归一化，且深度为1
train_images = train_images.reshape([-1, 28, 28, 1]) / 255.0
test_images = test_images.reshape([-1, 28, 28, 1]) / 255.0

# 创建模型
model = keras.models.Sequential()
# 卷积层 28*28*1->28*28*32
model.add(keras.layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding='same'))
# 最大池化 28*28*32->14*14*32
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
# 卷积层 14*14*32->14*14*64
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
# 最大池化 14*14*64->7*7*64
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
# flatten展开成一维3136
model.add(keras.layers.Flatten())
# 全连接层 3136->256
model.add(keras.layers.Dense(256, activation=tf.nn.relu))
# Dropout层
keras.layers.AlphaDropout(rate=0.5)
# 全连接层 256->10
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# 编译
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 输出网络框架
model.summary()

# 训练
print('\n开始训练...')
start_train = time.time()
model.fit(train_images, train_labels, epochs=5, validation_data=[test_images[:1000], test_labels[:1000]])
end_train = time.time()

# 评估
print('\n开始测试...')
start_test = time.time()
test_loss, test_acc = model.evaluate(test_images, test_labels)
end_test = time.time()

print('\n正确率为：', test_acc)
print("训练过程所需时间为: %f 秒" % (end_train - start_train))
print("测试过程所需时间为: %f 秒" % (end_test - start_test))

# 识别自己的10幅简笔画
print("\n开始识别自己的10幅简笔画...")
for i in range(10):
    im1 = pre_pic('../my_picture/%d.jpg' % i)
    im1 = im1.reshape([-1, 28, 28, 1]) / 255.0
    label = model.predict(im1)
    if np.argmax(label[0]) == i:
        print('标签为 %d 的简笔画识别成功' % i)
    else:
        print('标签为 %d 的简笔画识别失败' % i)
