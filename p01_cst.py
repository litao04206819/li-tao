"""
CST (Class-Shape Transformation) 翼型参数化方法
==============================================
参考: Kulfan B.M., "Universal Parametric Geometry Representation Method,"
      Journal of Aircraft, Vol. 45, No. 1, 2008.

核心思想: y(ψ) = C(ψ) · S(ψ) + ψ · ΔzTE
  - C(ψ) = ψ^N1 · (1-ψ)^N2   类别函数（前缘圆+尾缘尖）
  - S(ψ) = Σ Ai · Bernstein_i  形状函数（Bernstein多项式加权）
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb

# ============================================================
# 核心函数
# ============================================================

def bernstein(n, k, psi):
    """第k个n阶Bernstein多项式基函数"""
    return comb(n, k) * psi**k * (1 - psi)**(n - k)


def cst_airfoil(A_upper, A_lower, n_pts=201, N1=0.5, N2=1.0, dz_te=0.0):
    """
    用CST方法生成翼型坐标

    参数:
        A_upper : array  上表面CST系数
        A_lower : array  下表面CST系数
        n_pts   : int    每面的离散点数
        N1, N2  : float  类别函数指数 (翼型默认0.5, 1.0)
        dz_te   : float  尾缘半厚度

    返回:
        x, y_upper, y_lower : 无量纲坐标
    """
    psi = np.linspace(0, 1, n_pts)

    # 类别函数 C(ψ) —— 保证前缘圆+尾缘尖
    C = psi**N1 * (1 - psi)**N2

    # 形状函数 S(ψ) —— Bernstein多项式加权和
    n_upper = len(A_upper) - 1
    n_lower = len(A_lower) - 1

    S_upper = sum(A_upper[k] * bernstein(n_upper, k, psi) for k in range(n_upper + 1))
    S_lower = sum(A_lower[k] * bernstein(n_lower, k, psi) for k in range(n_lower + 1))

    # 翼型坐标 = 类别函数 × 形状函数 + 尾缘修正
    y_upper = C * S_upper + psi * dz_te
    y_lower = C * S_lower - psi * dz_te

    return psi, y_upper, y_lower


def fit_cst(x_data, y_data, order=5, N1=0.5, N2=1.0):
    """
    对已知翼型坐标做最小二乘CST拟合

    参数:
        x_data, y_data : 翼型单面坐标 (无量纲, x从0到1)
        order          : Bernstein多项式阶数

    返回:
        A : CST系数数组
    """
    psi = x_data
    C = psi**N1 * (1 - psi)**N2

    # 构造Bernstein基函数矩阵
    B = np.zeros((len(psi), order + 1))
    for k in range(order + 1):
        B[:, k] = C * bernstein(order, k, psi)

    # 最小二乘求解: B @ A ≈ y_data
    A, _, _, _ = np.linalg.lstsq(B, y_data, rcond=None)
    return A


# ============================================================
# NACA 0012 坐标 (用于验证拟合)
# ============================================================

def naca0012(x):
    """NACA 0012 解析公式"""
    t = 0.12
    return 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                     + 0.2843*x**3 - 0.1015*x**4)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 示例1: 直接用CST系数生成翼型 ---
    # 这组系数近似于一个典型运输机翼型
    A_up = np.array([0.17, 0.16, 0.14, 0.16, 0.12, 0.10])
    A_lo = np.array([-0.17, -0.08, -0.06, -0.04, -0.02, -0.005])

    x, yu, yl = cst_airfoil(A_up, A_lo)

    ax = axes[0, 0]
    ax.fill_between(x, yu, yl, alpha=0.15, color='steelblue')
    ax.plot(x, yu, 'b-', lw=2, label='Upper')
    ax.plot(x, yl, 'r-', lw=2, label='Lower')
    ax.set_aspect('equal')
    ax.set_title('CST Direct Generation', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例2: 拟合NACA 0012 ---
    x_fit = np.linspace(0.001, 1, 200)  # 避开前缘奇点
    y_target = naca0012(x_fit)

    ax = axes[0, 1]
    for order in [3, 5, 7]:
        A_fitted = fit_cst(x_fit, y_target, order=order)
        x_gen, yu_gen, _ = cst_airfoil(A_fitted, -A_fitted)
        ax.plot(x_gen, yu_gen, lw=2, label=f'CST order={order}')

    ax.plot(x_fit, y_target, 'k--', lw=1.5, label='NACA 0012 exact')
    ax.set_aspect('equal')
    ax.set_title('CST Fitting NACA 0012 (Different Orders)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例3: 改变单个CST系数的效果 ---
    A_base = np.array([0.15, 0.15, 0.15, 0.15, 0.15])
    ax = axes[1, 0]

    for i in range(5):
        A_mod = A_base.copy()
        A_mod[i] += 0.08  # 增大第i个系数
        x_m, yu_m, _ = cst_airfoil(A_mod, -A_base)
        ax.plot(x_m, yu_m, lw=2, label=f'A[{i}] += 0.08')

    x_b, yu_b, _ = cst_airfoil(A_base, -A_base)
    ax.plot(x_b, yu_b, 'k--', lw=2, label='Baseline')
    ax.set_aspect('equal')
    ax.set_title('Effect of Each CST Coefficient', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 示例4: Bernstein基函数可视化 ---
    ax = axes[1, 1]
    psi = np.linspace(0, 1, 200)
    n = 5
    for k in range(n + 1):
        bk = np.array([bernstein(n, k, p) for p in psi])
        ax.plot(psi, bk, lw=2, label=f'B_{k},{n}')
    ax.set_title(f'Bernstein Basis Functions (n={n})', fontsize=13, fontweight='bold')
    ax.set_xlabel('ψ = x/c')
    ax.set_ylabel('B(ψ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/cst_demo.png', dpi=150)
    plt.close()
    print("CST demo saved.")
