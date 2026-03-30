import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ====================== 核心：类脑神经元层 ======================
class BioMLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.99, eta=0.01):
        super().__init__()
        # 控制sigmoid门控的可训练权重
        self.linear = nn.Linear(in_dim, out_dim)
        # 可训练参数：s_i对门控的调节系数
        self.gamma = nn.Parameter(torch.tensor(0.1))
        # 生物超参
        self.alpha = alpha
        self.eta = eta
        # 神经元内部置信度状态 s_i
        self.register_buffer("s_i", torch.zeros(out_dim))
        # 保存sigmoid门控输出
        self.gate_a = None

    def forward(self, x):
        # 1. 计算门控输入
        z = self.linear(x) + self.gamma * self.s_i
        # 2. sigmoid软门控
        self.gate_a = torch.sigmoid(z)
        # 3. 非inplace更新s_i，避免破坏计算图
        self.s_i = self.alpha * self.s_i + (1 - self.alpha) * self.gate_a.detach()
        return self.gate_a

    # 奖励直接修改s_i（纯状态更新，不参与梯度）
    def apply_reward(self, R):
        self.s_i = self.s_i + self.eta * R * self.gate_a.detach()

    def reset_state(self):
        self.s_i.zero_()

# ====================== 双层类脑MLP ======================
class BioSinNet(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.layer1 = BioMLPLayer(1, hidden_size)
        self.layer2 = BioMLPLayer(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_layer(x)

    def reset_all_states(self):
        self.layer1.reset_state()
        self.layer2.reset_state()

    def apply_reward(self, R):
        self.layer1.apply_reward(R)
        self.layer2.apply_reward(R)

# ====================== 损失函数 ======================
def compute_loss(y_pred, y_true, model, 
                 λ_stable=0.1, λ_sparse=1.2, λ_consist=0):
    # 主任务损失
    loss_task = nn.MSELoss()(y_pred, y_true)
    
    # 正则1：s_i稳定性
    loss_stable = λ_stable * (torch.mean(model.layer1.s_i**2) + torch.mean(model.layer2.s_i**2))
    # 正则2：稀疏性
    loss_sparse = λ_sparse * (torch.abs(model.layer1.s_i).mean() + torch.abs(model.layer2.s_i).mean())
    # 正则3：s_i与门控一致
    loss_consist = λ_consist * (
        ((model.layer1.s_i - model.layer1.gate_a)**2).mean() +
        ((model.layer2.s_i - model.layer2.gate_a)**2).mean()
    )
    
    total_loss = loss_task + loss_stable + loss_sparse + loss_consist
    return total_loss, loss_task

# ====================== 数据准备 ======================
x = np.linspace(0, 2 * np.pi, 1000, dtype=np.float32)
y = np.sin(x)+np.sin(3*x)+np.sin(5*x)+0.1*x**2



x_tensor = torch.from_numpy(x).unsqueeze(-1)
y_tensor = torch.from_numpy(y).unsqueeze(-1)

# ====================== 训练配置 ======================
model = BioSinNet(hidden_size=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 10000

loss_curve = []
task_loss_curve = []

# ====================== 训练循环 ======================
print("开始训练...")
for epoch in range(EPOCHS):
    model.reset_all_states()
    
    # 前向传播
    y_pred = model(x_tensor)
    # 计算损失
    total_loss, task_loss = compute_loss(y_pred, y_tensor, model)
    
    # 反向传播（先算梯度，再改状态）
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 奖励应用：放在梯度更新之后，彻底避免inplace冲突
    R = -task_loss.detach()
    model.apply_reward(R)
    
    # 记录日志
    loss_curve.append(total_loss.item())
    task_loss_curve.append(task_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Total Loss: {total_loss.item():.6f} | MSE Loss: {task_loss.item():.6f}")

# ====================== 结果可视化 ======================
print("\n训练完成，绘制结果...")
model.eval()
model.reset_all_states()
with torch.no_grad():
    y_pred = model(x_tensor).numpy().squeeze()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(x, y, label='真实 sin(x)', linewidth=2)
plt.plot(x, y_pred, label='BioMLP拟合结果', linestyle='--', linewidth=2)
plt.title('拟合效果')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(task_loss_curve, label='MSE Loss', color='orange')
plt.title('训练损失曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()