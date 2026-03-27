import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ======================
# 1. 读取图片 → 矩阵 A
# ======================
img = Image.open("2.png").convert("L")  # 灰度图
A = np.array(img, dtype=np.float32) / 255.0              # 归一化到 0~1
print("A 的形状:", A.shape)  # 应该是 (256,256)

# ======================
# 2. 算 A^T A
# ======================
ATA = A.T @ A

# ======================
# 3. 求 A^TA 的特征值 & 特征向量 → V 和 σ
# ======================
eig_vals, V = np.linalg.eig(ATA)

# 特征值可能有微小虚部，去掉
eig_vals = np.real(eig_vals)
V = np.real(V)

# 奇异值 σ = sqrt(特征值)
sigmas = np.sqrt(np.maximum(eig_vals, 0))  # 防止负数

# 从大到小排序（关键！信息多的放前面）
idx = np.argsort(sigmas)[::-1]
sigmas = sigmas[idx]
V = V[:, idx]

# ======================
# 4. 用 A v_i = σ_i u_i 计算 U
# ======================
U = np.zeros_like(A)
for i in range(256):
    sigma = sigmas[i]
    vi = V[:, i:i+1]
    if sigma > 1e-6:
        Ui = A @ vi / sigma
    else:
        Ui = np.zeros((256,1))
    U[:, i:i+1] = Ui

# ======================
# 5. 只保留前 k 个奇异值（压缩）
# ======================
k = 4  # 只留前20个“信息最多”的方向
U_k = U[:, :k]
Sigma_k = np.diag(sigmas[:k])
V_k = V[:, :k]

# 重建图片
A_recon = U_k @ Sigma_k @ V_k.T
A_recon = np.clip(A_recon, 0, 1)

# ======================
# 6. 画图对比
# ======================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("原图")
plt.imshow(A, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"只用前{k}个奇异值")
plt.imshow(A_recon, cmap="gray")
plt.axis("off")

plt.show()

# ======================
# 7. 看奇异值分布：前面巨大，后面几乎为0
# ======================
plt.figure()
plt.plot(sigmas[:50])
plt.title("前50个奇异值大小")
plt.xlabel("i")
plt.ylabel("σ_i (信息量)")
plt.show()