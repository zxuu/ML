import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient


class MultilayerPerceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        '''
        初始参数
        :param data: 传进来数据值1*784
        :param labels: 标签（one-hot encoding 编码）
        :param layers: 784 25 10 各层神经元个数
        :param normalize_data: 标准化data
        '''
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed  # 1000*785
        self.labels = labels  # 1000*1
        self.layers = layers  # [784,25,10]
        self.normalize_data = normalize_data
        self.thetas = MultilayerPerceptron.theta_init(layers)  # {dict}

    @staticmethod
    def theta_init(layers):
        '''
        初始化各层神经元
        :param layers: [list] 各层神经元个数[784,25,10]
        :return:
        '''
        num_layers = len(layers)  # 几层隐含层，3
        thetas = {}
        for layer_index in range(num_layers - 1):  # 2次
            """
            会执行两次，得到两组参数矩阵：25*785， 10*26
            """
            in_count = layers[layer_index]  # 当前层的输入
            out_count = layers[layer_index + 1]  # 当前层的输出
            # 这里需要考虑偏置项，记住一点偏置的个数跟输出的结果是一致的
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05  # 随即得到初始化结果0~1之间，尽量小一点
        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        '''
        将矩阵拉长方便计算，并还原
        各层参数θ拉长成一长条，然后拼接在一起成为一个
        :param self:
        :param thetas: 权重参数θ
        :return: 一长条权重参数θ
        '''
        num_theta_layers = len(thetas)  # 权重的组数
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            unrolled_theta = np.hstack((unrolled_theta, thetas[theta_layer_index].flatten()))
        return unrolled_theta

    def train(self, max_iterations=1000, alpha=0.1):
        '''
        训练模块
        :param max_iterations: 迭代次数
        :param alpha: 学习率
        :return: self.thetas,cost_history：训练好的权重、损失值
        '''
        # 权重参数θ拉成一长条
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)
        # 进行梯度下降
        (optimized_theta, cost_history) = MultilayerPerceptron.gradient_descent(self.data, self.labels, unrolled_theta
                                                                                , self.layers, max_iterations, alpha)
        # 转参数换成矩阵形式
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta, self.layers)
        return self.thetas, cost_history

    @staticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations, alpha):
        '''
        梯度下降
            1、计算损失值
            2、计算梯度值
            3、进行梯度更新
        :param data: 数据
        :param labels: 标签
        :param unrolled_theta: 权重参数（一维长条）
        :param layers: 各层数值
        :param max_iterations: 迭代次数
        :param alpha: 学习率
        :return: optimized_theta,cost_history
        '''
        optimized_theta = unrolled_theta  # 优化后的参数θ
        cost_history = []  # 记录损失值。每次迭代记录一个损失值
        for _ in range(max_iterations):
            # 计算损失值
            cost = MultilayerPerceptron.cost_function(data
                                                      , labels
                                                      , MultilayerPerceptron.thetas_roll(optimized_theta, layers)
                                                      , layers)

            cost_history.append(cost)
            # 计算梯度
            theta_gradient = MultilayerPerceptron.gradient_step(data, labels, optimized_theta, layers)  # 梯度
            # 更新参数
            optimized_theta = optimized_theta - alpha * theta_gradient

        return optimized_theta, cost_history

    @staticmethod
    def thetas_roll(unrolled_theta, layers):
        '''
        将长条还原成各个矩阵
        :param unrolled_theta: 长条权重θ
        :param layers: [各个矩阵的数值784，25，,10]
        :return: theta, {[25×785],[10×26]}
        '''
        num_layers = len(layers)
        theta = {}
        unrolled_shift = 0  # 变换到哪个矩阵了
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]

            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height

            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume

            layer_theta_unrolled = unrolled_theta[start_index:end_index]
            theta[layer_index] = layer_theta_unrolled.reshape((thetas_height, thetas_width))

            unrolled_shift = end_index
        return theta

    @staticmethod
    def cost_function(data, labels, thetas, layers):
        '''
        计算损失
        :param data: 输入数据
        :param labels: 标签，0123456789
        :param thetas: 权重参数θ，矩阵形式：{[25×785],[10×26]}
        :param layers: [list]，[784,25,10]
        :return: 损失
        '''
        num_layers = len(layers)
        num_examples = data.shape[0]  # 样本个数
        num_labels = layers[-1]  # 标签个数（最后一层的个数）
        # 前向传播走一次
        predictions = MultilayerPerceptron.feedforward_propagation(data, thetas, layers)
        # 制作标签，每一个样本的标签都得是one-hot。label只是0123456789
        bitwise_labels = np.zeros((num_examples, num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1  # 对应位置赋值为1
        # 对的损失
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        # 错的损失
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 1]))
        # 总损失
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost)
        return cost

    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        '''
        进行前向传播
        :param thetas: 权重参数
        :param layers: 各层神经元个数[784,25,10]
        :return:
        '''
        num_layers = len(layers)
        num_examples = data.shape[0]  # 样本个数
        in_layer_activation = data  # 层输入
        # 逐层计算
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            # 正常计算完之后是num_example*25，但是要考虑到偏置项，再加一列1,变成num_example*26。每层都有偏置项！！!
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))  # 加在第一列
            in_layer_activation = out_layer_activation
        # 返回输出值结果,不要偏置项,[1000,10],10分类
        return in_layer_activation[:, 1:]  # 最后一层要去掉偏置项

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        '''
        y一步步的梯度下降
        :param data: 输入数据
        :param labels: 标签
        :param optimized_theta: 长条θ，unrolled
        :param layers: []
        :return: thetas_unrolled_gradients
        '''
        theta = MultilayerPerceptron.thetas_roll(optimized_theta, layers)  # 转换成矩阵形式的权重
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data, labels, theta, layers)
        # 拉成一条
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients

    @staticmethod
    def back_propagation(data, labels, thetas, layers):
        '''
        反向传播
        :param data: 数据
        :param labels: 标签
        :param theta: {[],[]},矩阵θ
        :param layers: [784,25,10]
        :return: deltas
        '''
        num_layers = len(layers)
        (num_examples, num_features) = data.shape
        num_label_types = layers[-1]  # 10，最后一层的神将元个数（要分类数）

        # 记录所有样本的梯度和
        deltas = {}
        # 初始化操作
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count + 1))  # 25*785  10*26

        for example_index in range(num_examples):  # 遍历每一个样本
            # 每一层的输出（没有加一列1）（不经过激活函数）
            layers_inputs = {}
            # 每一层的输出（经过激活函数之后的）
            layers_activation = data[example_index, :].reshape((num_features, 1))  # 785*1
            # 记录每一层的输出（经过激活函数之后的）

            # 记录每一层的激活之前的输出、激活之后的输出
            layers_activations = {}
            layers_activations[0] = layers_activation
            # 逐层计算(从后往前计算)
            for layer_index in range(num_layers - 1):
                # 得到当前权重参数值25*785  10*26
                layer_theta = thetas[layer_index]
                # 第一次得到25*1  第二次得到10*1
                layer_input = np.dot(layer_theta, layers_activation)
                # 更新layers_activation,还要加1扩列
                layers_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))
                # 记录
                layers_inputs[layer_index + 1] = layer_input
                # 记录
                layers_activations[layer_index + 1] = layers_activation
            # 最后一层的结果（去掉加的偏置项）
            output_layer_activation = layers_activation[1:, :]

            # 经过以上处理，记录了每一层的激活之前的输出、激活之后的输出

            delta = {}
            # 标签处理
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1  # one-hot编码
            # 计算输出层和真实值之间的差异
            delta[num_layers - 1] = output_layer_activation - bitwise_label

            # 遍历循环L L-1 L-2 ......
            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index + 1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array((1)), layer_input))

                # 按照公式进行计算(核心公式)
                # 当前层的权重×下一层的差值*当前层的sigmod()的导数
                delta[layer_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)
                # 过滤掉位置参数
                delta[layer_index] = delta[layer_index][1:, :]
            # 梯度
            for layer_index in range(num_layers - 1):
                layer_delta = np.dot(delta[layer_index + 1], layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta

        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)
        return deltas

    def predict(self, data):
        data_processed = prepare_for_training(data, normalize_data=True)[0]
        num_examples = data_processed.shape[0]
        predictions = MultilayerPerceptron.feedforward_propagation(data_processed, self.thetas, self.layers)
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))
