import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ====================== 普通双层MLP（对照基准）======================
class NormalMLP(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

# ====================== 数据：训练 0~2π，泛化测试 0~4π ======================
x_train = np.linspace(0, 2 * np.pi, 1000, dtype=np.float32)
y_train = np.sin(x_train)+np.sin(3*x_train)+np.sin(5*x_train)+0.1*x_train**2
x_train_tensor = torch.from_numpy(x_train).unsqueeze(-1)
y_train_tensor = torch.from_numpy(y_train).unsqueeze(-1)

x_test  = np.linspace(0, 4 * np.pi, 1000, dtype=np.float32)
y_test  = np.sin(x_test)+np.sin(3*x_test)+np.sin(5 *x_test )+0.1*x_test**2 
x_test_tensor  = torch.from_numpy(x_test).unsqueeze(-1)
y_test_tensor  = torch.from_numpy(y_test).unsqueeze(-1)

# ====================== 训练配置 ======================
model = NormalMLP(hidden_size=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
EPOCHS = 10000

loss_history = []

print("开始训练普通双层MLP...")
for epoch in range(EPOCHS):
    model.train()
    y_pred = model(x_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 15 == 0:
        print(f"Epoch {epoch+1:3d} | MSE Loss: {loss.item():.6f}")

# ====================== 泛化测试 ======================
model.eval()
with torch.no_grad():
    y_pred_test = model(x_test_tensor).numpy().squeeze()

# ====================== 绘图 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(14, 6))

plt.plot(x_test, y_test, label='真实 sin(x) 0~4π', linewidth=2.5)
plt.plot(x_test, y_pred_test, label='普通MLP预测（仅训练0~2π）', linestyle='--', linewidth=2.5)
plt.axvline(x=2*np.pi, color='red', linestyle=':', linewidth=2, label='训练边界 2π')

plt.title('普通双层MLP泛化能力测试', fontsize=14)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()