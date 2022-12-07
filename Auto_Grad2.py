import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(torch.Tensor([[0.1, 0.8], [0.8, 0.2]]))
y = Variable(torch.Tensor([1, 0]))


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer = nn.Linear(2, 1)
        self.layer2 = nn.Linear(1, 2)
        self.layer3 = nn.Linear(2, 3)

    def forward(self, x):
        out = self.layer(x)
        out = torch.tanh(out)
        out = self.layer2(out)
        out = torch.tanh(out)
        out = self.layer3(out)
        return self.layer(x)


net = MyNet()
mls = nn.MSELoss()  # 均方差误差
opt = torch.optim.Adam(net.parameters(), lr=0.01)  # 优化器，随机梯度下降，学习率
print(net.parameters())

for i in range(10):
    out = net(x)  # 开始训练
    loss = mls(out, y)  # 计算损失
    print(loss)
    opt.zero_grad()
    loss.backward()  # 损失函数上的反向传播
    opt.step()

print(net(x))
