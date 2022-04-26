import  numpy as np
from utils.features import prepare_for_training    #导入预处理的一些步骤

class LinearRegression:

    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
        '''
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵θ
        :param data: 传进来的样本数据
        :param labels: 真实值
        :param polynomial_degree:
        :param sinusoid_degree:
        :param normalize_data: 是否标准化
        '''
        (data_processed,    #数据标准化的一个结果
         features_mean,    #mean值，下面一个是标准差
         features_deviation)=prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0,normalize_data=True)



        self.data=data_processed    #标准化处理，并且加一列1
        self.labels=labels
        self.features_mean=features_mean    #均值
        self.features_deviation=features_deviation    #方差
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree
        self.normalize_data=normalize_data

        num_features = self.data.shape[1]    #data是传进来的数据，.shape()获取数组的形状，行、列
        self.theta = np.zeros((num_features,1))    #将θ构建一个矩阵的形式，num_features行，1列
        print(data_processed)

    def train(self,alpha,num_iterations=500):
        '''
        训练函数，执行梯度下降
        :param alpha: 学习率
        :param num_iteration:迭代次数
        :return: θ值， 损失值
        '''
        cost_history = self.gradient_decent(alpha,num_iterations)
        return self.theta,cost_history

    def gradient_decent(self,alpha,num_iterations):
        '''
        实际迭代模块，会迭代num_iterations次
        梯度下降函数，完成函数的更新
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return:
        '''
        cost_history = []
        for _ in range(num_iterations):    #迭代500次
            self.gradient_step(alpha)    #更新一次参数
            cost_history.append(self.cost_function(self.data,self.labels))    #计算一次损失
        return cost_history

    def gradient_step(self,alpha):
        '''
        梯度下降参数更新计算方法，注意是矩阵运算
        :param alpha: 学习率
        :return:
        '''
        num_examples=self.data.shape[0]    #多少个样本
        prediction=LinearRegression.hypothesis(self.data,self.theta)    #预测值等于样本数据*参数θ
        delta = prediction - self.labels    #预测值与真实值的差值
        theta = self.theta
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T    #采用批量梯度下降
        self.theta = theta    #更新θ参数

    def cost_function(self,data,labels):
        '''
        损失计算方法
        :param data:
        :param labels:
        :return:
        '''
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels    #预测值减去真实值
        cost = (1/2)*np.dot(delta.T,delta)/num_examples   #平方差
        #print('cost:',cost)  [[4.8798887]]
        #print(cost[0][0])
        return cost[0][0]    #？？？？？

    @staticmethod    #使之成为静态方法
    def hypothesis(data,theta):
        '''
        计算预测值
        :param data:
        :param theta:
        :return:预测结果值（）
        '''
        prediction=np.dot(data,theta)    #124×2 * 2×1
        return prediction

    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]    #只要标准化后的样本数据
        return self.cost_function(data_processed,labels)    #返回损失值
    def predict(self,data):
        '''
        用训练好的参数模型，预测得到回归值结果
        :param data:
        :return: 预测值
        '''
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]    #只要标准化后的样本数据
        print(data_processed)
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        return predictions