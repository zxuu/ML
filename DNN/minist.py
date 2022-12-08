import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math
import torch

from Neural_Network.multilayer_perceptron import MultilayerPerceptron

data = pd.read_csv('../data/mnist-demo.csv')  # 10000*785

numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))
for plot_index in range(numbers_to_display):
    digit = data[plot_index:plot_index + 1].values
    digit_label = digit[0][0]
    digit_pixels = digit[0][1:]
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# 训练集&测试集（把图片拉成一维向量）
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
train_data = train_data.values
test_data = test_data.values
# 训练集数量
num_training_examples = 1000

# 第一列代表该数字是几
x_train = train_data[:num_training_examples, 1:]  # 1000*784
y_train = train_data[:num_training_examples, [0]]  # 1000*1

x_test = test_data[:, 1:]  # 2000*784
y_test = test_data[:, [0]]  # 2000*1

layers = [784, 25, 10]

normalize_data = True  # 归一化
max_iterations = 1000  # 迭代次数
alpha = 0.1  # 学习率

multilayer_perceptron = MultilayerPerceptron(x_train, y_train, layers, normalize_data)
(thetas, costs) = multilayer_perceptron.train(max_iterations, alpha)
plt.plot(range(len(costs)), costs)
plt.xlabel('Gradient steps')
plt.ylabel('costs')
plt.show()

y_train_predictions = multilayer_perceptron.predict(x_train)
y_test_predictions = multilayer_perceptron.predict(x_test)

train_p = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
test_p = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100

print('训练集准确率：', train_p)
print('测试集准确率：', test_p)
