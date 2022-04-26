import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('./data/world-happiness-report-2017.csv')

#得到训练和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

#查看数据
plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

#开始训练
num_iteration = 500    #迭代次数
learning_rate = 0.01    #学习率

linear_regression = LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(learning_rate,num_iteration)

print('开始时的损失：',cost_history[0])
print('训练后的损失：',cost_history[-1])
#print('history:',cost_history)
print('theta:',theta)

plt.plot(range(num_iteration),cost_history)
plt.xlabel('迭代')
plt.ylabel('损失')
plt.title('损失情况')
plt.show()

#开始测试
predictions_num = 100    #测试100个
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_predictions,y_predictions,'r',label='prediction')    #画回归线
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
