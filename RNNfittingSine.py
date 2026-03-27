# 导库 matplot
import matplotlib.pyplot as plt
# 导入 torch用于完成神经网络计算
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义超参数
# 输入层神经元：1
input_size = 1
# 隐藏层一个：大小：100个神经元
hidden_size = 1000
# 输出层：1
output_size = 1
# 训练轮数：1000
num_epochs = 1000
# 学习率：0.01
learning_rate = 0.01

# 生成训练数据
# 按步长0.01，生成0-2pi内的t值和对应的y值
t = np.arange(0, 2*np.pi, 0.01)
y = np.sin(t)
# 转换为PyTorch张量
t_tensor = torch.FloatTensor(t).view(-1, 1, 1).to(DEVICE)  # (序列长度, batch_size, input_size)
y_tensor = torch.FloatTensor(y).view(-1, 1, 1).to(DEVICE)

# 类：前向传播和反向传播
class RNN(nn.Module):
    # pytorch定义神经网络结构
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
    # 前向传播：输入input和上步记忆h_prev,输出预测值和更新的记忆
    def forward(self, x, h_prev):
        out, h = self.rnn(x, h_prev)
        out = self.fc(out)
        return out, h

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)  # (num_layers, batch_size, hidden_size)

# 实例化模型、损失函数和优化器
model = RNN(input_size, hidden_size, output_size)
model = model.to(DEVICE)
criterion = nn.MSELoss()
# 反向传播：利用pytorch自动微分机制，计算梯度并更新，损失函数使用均方误差，使用adam优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
# 主程序
# 开始训练，同时显示损失的变化曲线
losses = []
for epoch in range(num_epochs):
    # 初始化隐藏状态
    hidden = model.init_hidden().to('cuda:0')

    # 前向传播
    outputs, hidden = model(t_tensor, hidden)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
# 显示结果
# 使用训练好的模型进行预测
with torch.no_grad():
    hidden = model.init_hidden().to('cuda:0')
    predictions, _ = model(t_tensor, hidden)

plt.figure(figsize=(12, 6))
plt.plot(t, y, label='True sin(t)', linewidth=2)
plt.plot(t, predictions.cpu().numpy().flatten(), 'r--', label='Predicted sin(t)', linewidth=2)
plt.title('RNN Prediction vs True sin(t)')
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.legend()
plt.grid(True)
plt.show()
