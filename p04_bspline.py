"""
B样条 / NURBS 翼型参数化方法
==============================
参考: Piegl L. & Tiller W., "The NURBS Book," 2nd ed., Springer, 1997.

核心思想: 用控制点定义参数曲线 C(u) = Σ P_i · N_{i,p}(u)
         移动控制点改变翼型形状, 天然光滑, 局部可控
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# B样条核心实现
# ============================================================

def find_span(n, p, u, U):
    """查找参数u所在的节点区间 (二分搜索)"""
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def basis_funs(i, u, p, U):
    """计算所有非零的p阶B样条基函数 (de Boor-Cox递推)"""
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def bspline_curve(P, p, U, n_pts=201):
    """
    计算B样条曲线

    参数:
        P     : (n+1, 2) 控制点坐标
        p     : int      阶数
        U     : array    节点向量
        n_pts : int      输出点数

    返回:
        curve : (n_pts, 2) 曲线坐标
    """
    n = len(P) - 1
    curve = np.zeros((n_pts, 2))
    u_vals = np.linspace(U[p], U[n + 1] - 1e-10, n_pts)

    for idx, u in enumerate(u_vals):
        span = find_span(n, p, u, U)
        N = basis_funs(span, u, p, U)
        for j in range(p + 1):
            curve[idx] += N[j] * P[span - p + j]
    return curve


def uniform_knot_vector(n, p):
    """生成均匀节点向量 (clamped: 首尾重复p+1次)"""
    m = n + p + 1
    U = np.zeros(m + 1)
    for i in range(m + 1):
        if i <= p:
            U[i] = 0.0
        elif i >= m - p:
            U[i] = 1.0
        else:
            U[i] = (i - p) / (m - 2 * p)
    return U


def bspline_airfoil(ctrl_upper, ctrl_lower, p=3, n_pts=201):
    """
    用B样条生成翼型

    参数:
        ctrl_upper : (m, 2) 上表面控制点 [必须从(0,0)开始, (1,~0)结束]
        ctrl_lower : (m, 2) 下表面控制点
        p          : 阶数 (通常3=三次)

    返回:
        x_up, y_up, x_lo, y_lo
    """
    n_up = len(ctrl_upper) - 1
    n_lo = len(ctrl_lower) - 1

    U_up = uniform_knot_vector(n_up, p)
    U_lo = uniform_knot_vector(n_lo, p)

    curve_up = bspline_curve(ctrl_upper, p, U_up, n_pts)
    curve_lo = bspline_curve(ctrl_lower, p, U_lo, n_pts)

    return curve_up[:, 0], curve_up[:, 1], curve_lo[:, 0], curve_lo[:, 1]


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 示例1: 基本B样条翼型 ---
    ctrl_up = np.array([
        [0.00, 0.000],
        [0.00, 0.030],  # 控制前缘曲率
        [0.10, 0.065],
        [0.30, 0.065],
        [0.60, 0.045],
        [0.85, 0.018],
        [1.00, 0.001],
    ])
    ctrl_lo = np.array([
        [0.00,  0.000],
        [0.00, -0.025],
        [0.10, -0.050],
        [0.30, -0.048],
        [0.60, -0.025],
        [0.85, -0.010],
        [1.00, -0.001],
    ])

    xu, yu, xl, yl = bspline_airfoil(ctrl_up, ctrl_lo, p=3)

    ax = axes[0, 0]
    ax.fill_between(xu, yu, np.interp(xu, xl, yl), alpha=0.15, color='steelblue')
    ax.plot(xu, yu, 'b-', lw=2, label='Upper surface')
    ax.plot(xl, yl, 'r-', lw=2, label='Lower surface')
    ax.plot(ctrl_up[:, 0], ctrl_up[:, 1], 'b^--', ms=8, alpha=0.6, label='Ctrl pts (upper)')
    ax.plot(ctrl_lo[:, 0], ctrl_lo[:, 1], 'rv--', ms=8, alpha=0.6, label='Ctrl pts (lower)')
    ax.set_aspect('equal')
    ax.set_title('B-Spline Airfoil (p=3)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 示例2: 移动单个控制点的效果 ---
    ax = axes[0, 1]
    ax.plot(xu, yu, 'k--', lw=1.5, label='Original')

    for delta_y, color in [(0.02, 'blue'), (0.04, 'green'), (-0.02, 'red')]:
        ctrl_mod = ctrl_up.copy()
        ctrl_mod[3, 1] += delta_y  # 移动第3个控制点
        xu_m, yu_m, _, _ = bspline_airfoil(ctrl_mod, ctrl_lo, p=3)
        ax.plot(xu_m, yu_m, color=color, lw=2,
                label=f'P[3].y += {delta_y:+.3f}')
        ax.plot(ctrl_mod[3, 0], ctrl_mod[3, 1], 'o', color=color, ms=10)

    ax.set_aspect('equal')
    ax.set_title('Moving Control Point P[3]', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例3: 不同阶数p的影响 ---
    ax = axes[1, 0]
    for p in [2, 3, 4, 5]:
        try:
            xu_p, yu_p, _, _ = bspline_airfoil(ctrl_up, ctrl_lo, p=p)
            ax.plot(xu_p, yu_p, lw=2, label=f'p = {p}')
        except Exception:
            pass

    ax.plot(ctrl_up[:, 0], ctrl_up[:, 1], 'ks--', ms=6, alpha=0.4, label='Ctrl pts')
    ax.set_aspect('equal')
    ax.set_title('Effect of B-Spline Degree p', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例4: B样条基函数可视化 ---
    ax = axes[1, 1]
    n_ctrl = 7
    p = 3
    U = uniform_knot_vector(n_ctrl - 1, p)
    u_plot = np.linspace(0, 1 - 1e-10, 500)

    for i in range(n_ctrl):
        N_vals = np.zeros(len(u_plot))
        for idx, u in enumerate(u_plot):
            span = find_span(n_ctrl - 1, p, u, U)
            N = basis_funs(span, u, p, U)
            for j in range(p + 1):
                if span - p + j == i:
                    N_vals[idx] = N[j]
        ax.plot(u_plot, N_vals, lw=2, label=f'N_{i},{p}')

    ax.set_title(f'B-Spline Basis Functions (n={n_ctrl-1}, p={p})', fontsize=13, fontweight='bold')
    ax.set_xlabel('u')
    ax.set_ylabel('N(u)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/bspline_demo.png', dpi=150)
    plt.close()
    print("B-Spline demo saved.")
