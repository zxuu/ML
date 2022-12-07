import torch
import torch.nn as nn
from torch.autograd import Variable

# y=x**2
# x=[3]
# x=Variable(torch.Tensor(x), requires_grad=True)
# y=x**2
# y.backward()
# print(x.grad)

x = [[0.1, 0.8, 1], [0.8, 0.2, 1]]
y = [[1], [0]]
w = [[0.1, 0.2, 0.3]]

x = Variable(torch.Tensor(x))
y = Variable(torch.Tensor(y))
w = Variable(torch.Tensor(w), requires_grad=True)  # 自动求导requires_grad=True

for i in range(1000):
    out = torch.mm(x, w.t())
    delta = (out - y)
    loss = delta[0] ** 2 + delta[1] ** 2
    print(loss)

    w.grad = torch.Tensor([[0, 0, 0]])  # 梯度清零 torch.Tensor()

    loss.backward()  # 反向传播过程中自动求导

    w.data -= w.grad * 0.01  # 必须这样写不可以w.data = w.data-w.grad * 0.01否则累加

print(torch.mm(x, w.t())) #输出预测结果
