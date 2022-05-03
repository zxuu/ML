import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid

class LogisticRegression:
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=False):
        '''
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        :param data: 输入样本数据
        :param labels: 样本标签（真实值）
        :param polynomial_degree: 特征变换
        :param sinusoid_degree: 特征变换
        :param normalize_data: 是否对输入数据进行标准化操作
        '''
        (data_processed,
         features_mean, 
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=False)
         
        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)    #['SETOSA','VERSICOLOR','VIRGINICA'] 就是三种花

        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        
        num_features = self.data.shape[1]    #样本特征数量（3，加了一列1）
        num_unique_labels = np.unique(labels).shape[0]    #花的种类数
        self.theta = np.zeros((num_unique_labels,num_features))    #（3*3）花的种类数3*特征数量3
        
    def train(self,max_iterations=1000):
        cost_histories = []    #记录损失值
        num_features = self.data.shape[1]    #样本属性量
        for label_index,unique_label in enumerate(self.unique_labels):
            '''
            对每一种花做一次二分类
            '''
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features,1))    #初始的参数矩阵
            current_lables = (self.labels == unique_label).astype(float)    #是当前花的标记为1，其他为0
            (current_theta,cost_history) = LogisticRegression.gradient_descent(self.data,current_lables,current_initial_theta,max_iterations)
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)
            
        return self.theta, cost_histories
            
    @staticmethod        
    def gradient_descent(data,labels,current_initial_theta,max_iterations):
        '''
        执行梯度下降
        :param data: 样本数据
        :param labels: 已经分类后的标签
        :param current_initial_theta: 对当前分类初始化的参数θ(3*1)
        :param max_iterations: 迭代次数
        :return: 优化后的参数θ，损失值列表
        '''
        cost_history = []    #记录损失值
        num_features = data.shape[1]    #样本特征量
        result = minimize(
            #要优化的目标：
            #lambda current_theta:LogisticRegression.cost_function(样本数据,标签(0,1),初始参数θ.reshape(num_features,1)),
            lambda current_theta:LogisticRegression.cost_function(data,labels,current_theta.reshape(num_features,1)),
            #初始化的权重参数
            current_initial_theta,
            #选择优化策略
            method = 'CG',
            # 梯度下降迭代计算公式
            #jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_initial_theta.reshape(num_features,1)),
            jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_theta.reshape(num_features,1)),
            # 记录结果
            callback = lambda current_theta:cost_history.append(LogisticRegression.cost_function(data,labels,current_theta.reshape((num_features,1)))),
            # 迭代次数  
            options={'maxiter': max_iterations}                                               
            )
        if not result.success:
            raise ArithmeticError('Can not minimize cost function'+result.message)
        optimized_theta = result.x.reshape(num_features,1)
        return optimized_theta,cost_history
        
    @staticmethod 
    def cost_function(data,labels,theat):
        '''
        计算损失
        :param data: 样本数据
        :param labels: 分类后的标签
        :param theat: 参数θ(3*1)
        :return: 损失值
        '''
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data,theat)
        y_is_set_cost = np.dot(labels[labels == 1].T,np.log(predictions[labels == 1]))
        y_is_not_set_cost = np.dot(1-labels[labels == 0].T,np.log(1-predictions[labels == 0]))
        cost = (-1/num_examples)*(y_is_set_cost+y_is_not_set_cost)   #损失函数
        return cost

    @staticmethod 
    def hypothesis(data,theat):
        '''
        预测值
        :param data: 样本数据
        :param theat: 参数θ(3*1)
        :return: 预测值（矩阵）,sigmod之后的
        '''
        predictions = sigmoid(np.dot(data,theat))    #[150*3]*[3*1]
        return  predictions
    
    @staticmethod     
    def gradient_step(data,labels,theta):
        '''
        梯度下降计算公式
        :param data: 样本数据
        :param labels: 标签
        :param theta: 参数θ
        :return: 梯度值（一维数组）
        '''
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data,theta)    #预测值
        label_diff = predictions- labels    #差值
        gradients = (1/num_examples)*np.dot(data.T,label_diff)
        
        return gradients.T.flatten()    #转换成一个一维数组
    
    def predict(self,data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree,self.normalize_data)[0]    #标准化数据
        prob = LogisticRegression.hypothesis(data_processed,self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)    #[1*150] 按行检索最大值，返回下标
        class_prediction = np.empty(max_prob_index.shape,dtype=object)
        for index,label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label
        return class_prediction.reshape((num_examples,1))
        
        
        
        
        
        
        
        
        
        


