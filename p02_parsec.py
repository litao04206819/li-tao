"""
PARSEC 翼型参数化方法
=====================
参考: Sobieczky H., "Parametric Airfoils and Wings,"
      Notes on Numerical Fluid Mechanics, Vol. 68, Vieweg, 1999.

核心思想: 用12个有明确几何含义的参数（前缘半径、最大厚度位置等）
         直接定义翼型, 每面用6阶多项式 z = Σ a_k · ψ^(k-1/2)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 核心函数
# ============================================================

def parsec_airfoil(params, n_pts=201):
    """
    PARSEC参数化生成翼型

    参数 (dict):
        r_le    : 前缘半径
        x_up    : 上表面最高点弦向位置
        z_up    : 上表面最高点高度 (正值)
        zxx_up  : 上表面最高点处曲率 (负值表示下凸)
        x_lo    : 下表面最低点弦向位置
        z_lo    : 下表面最低点深度 (负值)
        zxx_lo  : 下表面最低点处曲率 (正值表示上凸)
        z_te    : 尾缘偏移
        dz_te   : 尾缘半厚度
        alpha_te: 尾缘角 (rad)
        beta_te : 尾缘方向角 (rad)

    返回:
        x, y_upper, y_lower
    """
    r_le   = params['r_le']
    x_up   = params['x_up']
    z_up   = params['z_up']
    zxx_up = params['zxx_up']
    x_lo   = params['x_lo']
    z_lo   = params['z_lo']
    zxx_lo = params['zxx_lo']
    z_te   = params['z_te']
    dz_te  = params['dz_te']
    alpha_te = params['alpha_te']
    beta_te  = params['beta_te']

    # --- 构造上表面6×6线性方程组 ---
    # 基函数: ψ^(1/2), ψ^(3/2), ψ^(5/2), ψ^(7/2), ψ^(9/2), ψ^(11/2)
    # 约束条件:
    #   1. z(1) = z_te + dz_te           (尾缘高度)
    #   2. z(x_up) = z_up                (最高点高度)
    #   3. z'(x_up) = 0                  (最高点斜率=0)
    #   4. z''(x_up) = zxx_up            (最高点曲率)
    #   5. z'(1) = tan(alpha_te - beta_te/2)  (尾缘角)
    #   6. sqrt(2*r_le) 前缘半径约束

    # 指数序列
    exp = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

    def build_system(x_ext, z_ext, zxx_ext, is_upper):
        """为上表面或下表面构造线性方程组"""
        A = np.zeros((6, 6))
        b = np.zeros(6)

        # 约束1: z(1) = z_te ± dz_te
        for j in range(6):
            A[0, j] = 1.0  # 1^exp[j] = 1
        b[0] = z_te + dz_te if is_upper else z_te - dz_te

        # 约束2: z(x_ext) = z_ext
        for j in range(6):
            A[1, j] = x_ext**exp[j]
        b[1] = z_ext

        # 约束3: z'(x_ext) = 0
        for j in range(6):
            A[2, j] = exp[j] * x_ext**(exp[j] - 1)
        b[2] = 0.0

        # 约束4: z''(x_ext) = zxx_ext
        for j in range(6):
            A[3, j] = exp[j] * (exp[j] - 1) * x_ext**(exp[j] - 2)
        b[3] = zxx_ext

        # 约束5: z'(1) = 尾缘斜率
        for j in range(6):
            A[4, j] = exp[j]  # exp[j] * 1^(exp[j]-1)
        if is_upper:
            b[4] = np.tan(alpha_te - beta_te / 2)
        else:
            b[4] = np.tan(alpha_te + beta_te / 2)

        # 约束6: 前缘半径  →  a1 = ±sqrt(2*r_le)
        A[5, 0] = 1.0
        b[5] = np.sqrt(2 * r_le) if is_upper else -np.sqrt(2 * r_le)

        return A, b

    # 求解上表面
    A_up, b_up = build_system(x_up, z_up, zxx_up, True)
    a_upper = np.linalg.solve(A_up, b_up)

    # 求解下表面
    A_lo, b_lo = build_system(x_lo, z_lo, zxx_lo, False)
    a_lower = np.linalg.solve(A_lo, b_lo)

    # --- 生成坐标 ---
    x = np.linspace(0, 1, n_pts)
    y_upper = np.zeros_like(x)
    y_lower = np.zeros_like(x)

    for j in range(6):
        # 避免 0^负数
        mask = x > 0
        y_upper[mask] += a_upper[j] * x[mask]**exp[j]
        y_lower[mask] += a_lower[j] * x[mask]**exp[j]

    return x, y_upper, y_lower


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 示例1: 典型对称翼型 ---
    params_sym = {
        'r_le': 0.0080,    # 前缘半径
        'x_up': 0.30,      # 最大厚度位置 30%弦长
        'z_up': 0.06,      # 最大半厚度 6%
        'zxx_up': -0.40,   # 上表面曲率
        'x_lo': 0.30,
        'z_lo': -0.06,
        'zxx_lo': 0.40,
        'z_te': 0.0,
        'dz_te': 0.0,
        'alpha_te': 0.0,
        'beta_te': 0.16,
    }
    x, yu, yl = parsec_airfoil(params_sym)

    ax = axes[0, 0]
    ax.fill_between(x, yu, yl, alpha=0.15, color='steelblue')
    ax.plot(x, yu, 'b-', lw=2)
    ax.plot(x, yl, 'r-', lw=2)
    ax.set_aspect('equal')
    ax.set_title('Symmetric Airfoil (PARSEC)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.grid(True, alpha=0.3)

    # --- 示例2: 有弯度的翼型 ---
    params_camb = {
        'r_le': 0.0100,
        'x_up': 0.35,
        'z_up': 0.08,       # 上面更高
        'zxx_up': -0.45,
        'x_lo': 0.25,
        'z_lo': -0.03,      # 下面较浅
        'zxx_lo': 0.30,
        'z_te': 0.002,
        'dz_te': 0.001,
        'alpha_te': -0.02,
        'beta_te': 0.12,
    }
    x, yu, yl = parsec_airfoil(params_camb)

    ax = axes[0, 1]
    ax.fill_between(x, yu, yl, alpha=0.15, color='coral')
    ax.plot(x, yu, 'b-', lw=2)
    ax.plot(x, yl, 'r-', lw=2)
    ax.set_aspect('equal')
    ax.set_title('Cambered Airfoil (PARSEC)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.grid(True, alpha=0.3)

    # --- 示例3: 改变最大厚度位置 x_up 的效果 ---
    ax = axes[1, 0]
    for x_up_val in [0.20, 0.30, 0.40, 0.50]:
        p = params_sym.copy()
        p['x_up'] = x_up_val
        p['x_lo'] = x_up_val
        x, yu, yl = parsec_airfoil(p)
        ax.plot(x, yu, lw=2, label=f'x_up = {x_up_val:.2f}')
        ax.plot(x, yl, lw=2, color=ax.get_lines()[-1].get_color())

    ax.set_aspect('equal')
    ax.set_title('Effect of Max-Thickness Position', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例4: 改变前缘半径 r_le 的效果 ---
    ax = axes[1, 1]
    for r_le_val in [0.004, 0.008, 0.015, 0.025]:
        p = params_sym.copy()
        p['r_le'] = r_le_val
        x, yu, yl = parsec_airfoil(p)
        ax.plot(x, yu, lw=2, label=f'r_LE = {r_le_val:.3f}')
        ax.plot(x, yl, lw=2, color=ax.get_lines()[-1].get_color())

    ax.set_xlim(-0.02, 0.25)
    ax.set_ylim(-0.08, 0.10)
    ax.set_aspect('equal')
    ax.set_title('Effect of Leading-Edge Radius (Zoomed)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/parsec_demo.png', dpi=150)
    plt.close()
    print("PARSEC demo saved.")
