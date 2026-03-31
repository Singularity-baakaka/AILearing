import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text

# --------------------------
# 1. 生成近似线性的数据（房子大小 → 房价）
# --------------------------
np.random.seed(42)  # 固定随机结果，方便你复现
x = np.linspace(50, 150, 100)  # 房子大小：50~150平
y = 3 * x + np.random.randn(100) * 8  # 近似线性，带一点噪声（y≈3x）

# 把 x 变成矩阵格式（sklearn 要求）
X = x.reshape(-1, 1)

# --------------------------
# 2. 训练决策树回归
# --------------------------
# 决策树会：
# 1. 二分切割 x 区间
# 2. 每个区间输出 y 的均值
# 3. 用 MSE（方差）做分裂依据
# 4. 限制深度防止过拟合
# --------------------------
tree = DecisionTreeRegressor(
    max_depth=4,        # 树深度
    criterion="squared_error"  # 用均方误差（方差）分裂 → 你写的完全一致
)
tree.fit(X, y)

# --------------------------
# 3. 预测（看决策树如何分段输出均值）
# --------------------------
x_pred = np.linspace(50, 150, 100).reshape(-1, 1)
y_pred = tree.predict(x_pred)

# --------------------------
# 4. 画图看结果
# --------------------------
plt.scatter(x, y, label="原始数据（近似线性）", s=15)
plt.plot(x_pred, y_pred, color="red", linewidth=3, label="决策树拟合结果")
plt.xlabel("房子大小")
plt.ylabel("房价")
plt.legend()
plt.title("决策树回归：分段常数拟合线性数据")
plt.show()

# --------------------------
# 5. 打印决策规则 → 你能直接看到它怎么切区间、输出均值
# --------------------------
print("\n==== 决策树学习到的规则 ====")
print(export_text(tree))