import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# -------------------------- 1. 生成测试数据（线性不可分的月牙形） --------------------------
# 生成100个带噪声的月牙形数据
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
# 把标签从0/1换成我们熟悉的+1/-1，和之前的SVM理论完全对应
y = np.where(y == 0, -1, 1)

# 先看一下数据长什么样
plt.figure(figsize=(8, 5))
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='+1 类', s=50)
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', label='-1 类', s=50)
plt.legend(fontsize=12)
plt.title('测试数据：月牙形（线性不可分）', fontsize=14)
plt.show()


# -------------------------- 2. 定义画决策边界的工具函数 --------------------------
# 这个函数会画出：数据点、决策边界、支持向量
def plot_svm_boundary(model, X, y, title):
    # 生成网格，覆盖所有数据范围
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200)
    )
    
    # 预测网格中每个点的分类结果
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    
    # 画图
    plt.figure(figsize=(8, 5))
    # 画决策边界的填充色
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # 画黑色的决策边界线（f(x)=0）
    plt.contour(xx1, xx2, Z, levels=[0], linewidths=2, colors='black')
    # 画数据点
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='+1 类', s=50)
    plt.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', label='-1 类', s=50)
    # 画支持向量（黑圈标记，对应我们说的α>0的点）
    plt.scatter(
        model.support_vectors_[:, 0], model.support_vectors_[:, 1],
        s=150, linewidth=2, facecolors='none', edgecolors='black',
        label='支持向量'
    )
    plt.legend(fontsize=12)
    plt.title(title, fontsize=14)
    plt.show()


# -------------------------- 3. 线性SVM（硬间隔）效果 --------------------------
# 线性核，C设为极大值，代表硬间隔（不允许分错）
svm_linear = SVC(kernel='linear', C=1e10)
svm_linear.fit(X, y)

# 画图看效果
plot_svm_boundary(svm_linear, X, y, '线性SVM效果：无法分开非线性数据')


# -------------------------- 4. RBF核SVM效果 --------------------------
# RBF核，gamma就是我们之前说的γ，控制核的"宽窄"
svm_rbf = SVC(kernel='rbf', C=1e10, gamma=1)
svm_rbf.fit(X, y)

# 画图看效果
plot_svm_boundary(svm_rbf, X, y, 'RBF核SVM效果：完美拟合非线性边界')


# -------------------------- 5. 【可选】手写RBF核函数，彻底搞懂核的本质 --------------------------
# 手写RBF核函数，完全对应我们的公式：k(x,z) = exp(-γ||x-z||²)
def my_rbf_kernel(X1, X2, gamma=1):
    """
    输入两个样本矩阵X1、X2，输出核矩阵K
    K[i,j] = k(X1[i], X2[j])，也就是第i个和第j个样本的核值
    """
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # 完全按照RBF公式计算
            K[i, j] = np.exp(-gamma * np.sum((X1[i] - X2[j]) ** 2))
    return K

# 用我们自己写的核函数跑SVM
svm_my_rbf = SVC(kernel=lambda X1, X2: my_rbf_kernel(X1, X2, gamma=1), C=1e10)
svm_my_rbf.fit(X, y)

# 画图看效果，和官方RBF完全一致
plot_svm_boundary(svm_my_rbf, X, y, '手写RBF核SVM效果：和官方实现完全一致')