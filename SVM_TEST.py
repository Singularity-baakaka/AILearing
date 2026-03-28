import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 纯二维向量运算工具 (避开 np.cross 报错) --------------------------
def vec_sub(a, b):
    """向量减法：a - b"""
    return np.array([a[0] - b[0], a[1] - b[1]])

def vec_add(a, b):
    """向量加法：a + b"""
    return np.array([a[0] + b[0], a[1] + b[1]])

def vec_dot(a, b):
    """向量点积：a · b"""
    return a[0] * b[0] + a[1] * b[1]

def vec_perp(a):
    """二维向量垂直旋转90度 (逆时针)：(x,y) -> (-y, x)"""
    return np.array([-a[1], a[0]])

def vec_cross_z(a, b):
    """二维向量的叉积的z分量 (标量)：a × b = ax*by - ay*bx"""
    return a[0] * b[1] - a[1] * b[0]

def vec_len(a):
    """向量长度"""
    return np.sqrt(a[0]**2 + a[1]**2)

def vec_normalize(a):
    """单位化向量"""
    l = vec_len(a)
    return a / l if l > 1e-6 else np.array([1.0, 0.0])

# -------------------------- GJK 核心算法 --------------------------
def support(poly_A, poly_B, direction):
    """Minkowski 差支撑点：A - B 在 direction 方向上的最远点"""
    # A 在 direction 上的最远点
    idx_a = np.argmax([vec_dot(p, direction) for p in poly_A])
    # B 在 -direction 上的最远点
    idx_b = np.argmax([vec_dot(p, -direction) for p in poly_B])
    return vec_sub(poly_A[idx_a], poly_B[idx_b])

def contains_origin(simplex, direction):
    """
    判断单纯形是否包含原点，并更新搜索方向
    返回：(是否包含原点, 新的搜索方向, 新的单纯形)
    """
    a = simplex[-1]
    ao = vec_sub(np.zeros(2), a)  # 从a指向原点

    if len(simplex) == 3:
        # 三角形情况
        b = simplex[1]
        c = simplex[0]
        ab = vec_sub(b, a)
        ac = vec_sub(c, a)
        
        # 计算边的法向量
        ab_perp = vec_perp(ab)
        if vec_dot(ab_perp, ac) > 0:
            ab_perp = -ab_perp
            
        ac_perp = vec_perp(ac)
        if vec_dot(ac_perp, ab) > 0:
            ac_perp = -ac_perp

        if vec_dot(ab_perp, ao) > 0:
            # 原点在 AB 边外
            return False, ab_perp, [a, b]
        elif vec_dot(ac_perp, ao) > 0:
            # 原点在 AC 边外
            return False, ac_perp, [a, c]
        else:
            # 原点在三角形内！碰撞！
            return True, np.zeros(2), simplex

    else:
        # 线段情况
        b = simplex[0]
        ab = vec_sub(b, a)
        
        # 计算垂直于 ab 且指向原点的方向
        ab_perp = vec_perp(ab)
        if vec_dot(ab_perp, ao) < 0:
            ab_perp = -ab_perp
            
        return False, ab_perp, simplex

def gjk(poly_A, poly_B):
    """
    GJK 碰撞检测主算法
    返回：(是否碰撞, 最终单纯形)
    """
    # 初始方向：从 A 的中心指向 B 的中心
    center_A = np.mean(poly_A, axis=0)
    center_B = np.mean(poly_B, axis=0)
    direction = vec_sub(center_B, center_A)
    
    if vec_len(direction) < 1e-6:
        direction = np.array([1.0, 0.0])

    simplex = [support(poly_A, poly_B, direction)]
    direction = vec_sub(np.zeros(2), simplex[0])  # 指向原点

    while True:
        support_point = support(poly_A, poly_B, direction)
        
        # 检查支撑点是否在原点之后
        if vec_dot(support_point, direction) <= 0:
            # 没有穿过原点，不碰撞
            return False, simplex
        
        simplex.append(support_point)
        collision, direction, simplex = contains_origin(simplex, direction)
        
        if collision:
            return True, simplex

# -------------------------- EPA 算法 (计算最小平移向量 MTV) --------------------------
def epa(poly_A, poly_B, simplex):
    """
    扩展多面体算法 (EPA)：计算精确的最小平移向量
    """
    # 初始化 EPA 多边形为 GJK 结束时的单纯形
    polytope = simplex.copy()
    
    # 主循环：不断扩展直到找到最近边
    while True:
        # 1. 找到 polytope 中距离原点最近的边
        min_dist = float('inf')
        closest_normal = None
        closest_idx = -1
        
        n = len(polytope)
        for i in range(n):
            p1 = polytope[i]
            p2 = polytope[(i+1) % n]
            
            edge = vec_sub(p2, p1)
            normal = vec_perp(edge)
            normal = vec_normalize(normal)
            
            # 确保法向量指向外部
            if vec_dot(normal, p1) < 0:
                normal = -normal
            
            # 计算原点到这条边的距离
            dist = vec_dot(normal, p1)
            
            if dist < min_dist:
                min_dist = dist
                closest_normal = normal
                closest_idx = i
        
        # 2. 在这个法向量方向上找新的支撑点
        new_support = support(poly_A, poly_B, closest_normal)
        new_dist = vec_dot(new_support, closest_normal)
        
        # 3. 检查是否收敛
        if abs(new_dist - min_dist) < 1e-3:
            # 收敛了，返回 MTV
            # 注意：EPA 算出来的是 Minkowski 差里的向量，
            # 要让 A 分离，我们需要往相反方向移
            mtv = closest_normal * (min_dist + 0.02) # 加一点点余量保证分离
            return mtv
        else:
            # 没收敛，把新支撑点插入 polytope，继续扩展
            polytope.insert(closest_idx + 1, new_support)

# -------------------------- 可视化与测试 --------------------------
def plot_result(poly_A, poly_B, mtv):
    plt.figure(figsize=(12, 6))
    
    # 左图：原始重叠
    plt.subplot(1, 2, 1)
    plt.fill(poly_A[:, 0], poly_A[:, 1], color='royalblue', alpha=0.5, label='多边形A')
    plt.fill(poly_B[:, 0], poly_B[:, 1], color='orange', alpha=0.5, label='多边形B')
    
    # 闭合绘制
    def close_poly(p): return np.vstack([p, p[0]])
    plt.plot(close_poly(poly_A)[:, 0], close_poly(poly_A)[:, 1], color='darkblue', linewidth=2)
    plt.plot(close_poly(poly_B)[:, 0], close_poly(poly_B)[:, 1], color='darkorange', linewidth=2)
    
    plt.title('原始重叠 (GJK检测碰撞)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：平移后分离
    plt.subplot(1, 2, 2)
    poly_A_trans = poly_A + mtv
    plt.fill(poly_A_trans[:, 0], poly_A_trans[:, 1], color='royalblue', alpha=0.5, label='平移后A')
    plt.fill(poly_B[:, 0], poly_B[:, 1], color='orange', alpha=0.5, label='多边形B')
    
    plt.plot(close_poly(poly_A_trans)[:, 0], close_poly(poly_A_trans)[:, 1], color='darkblue', linewidth=2)
    plt.plot(close_poly(poly_B)[:, 0], close_poly(poly_B)[:, 1], color='darkorange', linewidth=2)
    
    # 绘制平移向量
    mean_A = np.mean(poly_A, axis=0)
    plt.arrow(mean_A[0], mean_A[1], mtv[0], mtv[1], 
              head_width=0.15, color='red', linewidth=2, label=f'平移向量 t={np.round(mtv,2)}')

    plt.title('平移后完全分离 (EPA计算MTV)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 测试1：重叠正方形
    print("=== 测试1：重叠正方形 ===")
    square_A = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    square_B = np.array([[1, 1], [1, 3], [3, 3], [3, 1]])
    
    is_collision, simplex = gjk(square_A, square_B)
    print(f"是否碰撞: {is_collision}")
    
    if is_collision:
        mtv = epa(square_A, square_B, simplex)
        print(f"最小平移向量: {np.round(mtv, 4)}")
        plot_result(square_A, square_B, mtv)

    # 测试2：旋转凸多边形
    print("\n=== 测试2：旋转凸多边形 ===")
    def make_rotated_rect(center, scale, angle):
        rect = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * scale
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return (R @ rect.T).T + center
    
    poly_A = make_rotated_rect((2, 3), 1.5, np.pi/6)
    poly_B = make_rotated_rect((3, 2), 1.2, -np.pi/4)
    
    is_collision2, simplex2 = gjk(poly_A, poly_B)
    print(f"是否碰撞: {is_collision2}")
    
    if is_collision2:
        mtv2 = epa(poly_A, poly_B, simplex2)
        print(f"最小平移向量: {np.round(mtv2, 4)}")
        plot_result(poly_A, poly_B, mtv2)