"""
Hicks-Henne 凸包函数翼型参数化方法
====================================
参考: Hicks R.M. & Henne P.A., "Wing Design by Numerical Optimization,"
      Journal of Aircraft, Vol. 15, No. 7, 1978.

核心思想: 在基准翼型上叠加一组凸包(Bump)函数
         y(ψ) = y_base(ψ) + Σ b_i · f_i(ψ)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 核心函数
# ============================================================

def hicks_henne_bump(psi, t_i, e_i=3.0):
    """
    单个Hicks-Henne凸包函数

    参数:
        psi : array  弦向坐标 [0, 1]
        t_i : float  凸包峰值位置 (0, 1)
        e_i : float  凸包宽度指数 (越大越窄, 通常取2~4)

    返回:
        f   : array  凸包函数值
    """
    # 避免 log(0)
    psi_safe = np.clip(psi, 1e-10, 1 - 1e-10)
    exponent = np.log(0.5) / np.log(t_i)
    f = np.sin(np.pi * psi_safe**exponent)**e_i
    return f


def hicks_henne_deform(x_base, y_base, bump_params, e_i=3.0):
    """
    用Hicks-Henne凸包变形翼型

    参数:
        x_base, y_base : 基准翼型坐标
        bump_params    : list of (t_i, b_i) 元组
                         t_i = 峰值位置, b_i = 幅值(正=往外鼓, 负=往里凹)
        e_i            : 凸包宽度指数

    返回:
        y_new : 变形后的y坐标
    """
    y_new = y_base.copy()
    for t_i, b_i in bump_params:
        y_new += b_i * hicks_henne_bump(x_base, t_i, e_i)
    return y_new


def naca0012_upper(x):
    """NACA 0012 上表面"""
    t = 0.12
    return 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                + 0.2843*x**3 - 0.1015*x**4)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.linspace(0, 1, 301)

    # --- 示例1: 各凸包函数的形状 ---
    ax = axes[0, 0]
    positions = [0.1, 0.25, 0.5, 0.75, 0.9]
    for ti in positions:
        f = hicks_henne_bump(x, ti, e_i=3.0)
        ax.plot(x, f, lw=2, label=f't = {ti}')
    ax.set_title('Hicks-Henne Bump Functions (e=3)', fontsize=13, fontweight='bold')
    ax.set_xlabel('ψ = x/c')
    ax.set_ylabel('f(ψ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例2: 宽度指数e的影响 ---
    ax = axes[0, 1]
    for ei in [1, 2, 3, 5, 8]:
        f = hicks_henne_bump(x, 0.5, e_i=ei)
        ax.plot(x, f, lw=2, label=f'e = {ei}')
    ax.set_title('Effect of Width Exponent e (t=0.5)', fontsize=13, fontweight='bold')
    ax.set_xlabel('ψ = x/c')
    ax.set_ylabel('f(ψ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例3: 在NACA0012上叠加凸包 ---
    y_base = naca0012_upper(x)

    ax = axes[1, 0]
    ax.plot(x, y_base, 'k--', lw=2, label='NACA 0012 (baseline)')

    # 情况A: 在前缘加正凸包（增厚前缘）
    bumps_A = [(0.15, 0.010), (0.25, 0.008)]
    y_A = hicks_henne_deform(x, y_base, bumps_A)
    ax.plot(x, y_A, 'b-', lw=2, label='+ front bumps')

    # 情况B: 在后部加负凸包（减薄后缘）
    bumps_B = [(0.65, -0.005), (0.80, -0.008)]
    y_B = hicks_henne_deform(x, y_base, bumps_B)
    ax.plot(x, y_B, 'r-', lw=2, label='- rear bumps')

    ax.set_aspect('equal')
    ax.set_title('Deforming NACA 0012 with Bumps', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例4: 用8个凸包做"均匀覆盖"参数化 ---
    ax = axes[1, 1]
    ax.plot(x, y_base, 'k--', lw=2, label='Baseline')

    np.random.seed(42)
    n_bumps = 8
    t_positions = np.linspace(0.05, 0.95, n_bumps)
    amplitudes = 0.006 * (np.random.rand(n_bumps) - 0.5)  # 随机小扰动

    bumps = list(zip(t_positions, amplitudes))
    y_rand = hicks_henne_deform(x, y_base, bumps)
    ax.plot(x, y_rand, 'b-', lw=2, label=f'{n_bumps} random bumps')

    # 显示各凸包的位置和极性
    for ti, bi in bumps:
        color = 'green' if bi > 0 else 'red'
        ax.annotate('', xy=(ti, y_base[np.argmin(np.abs(x-ti))]),
                    xytext=(ti, y_base[np.argmin(np.abs(x-ti))] + bi*5),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.set_aspect('equal')
    ax.set_title(f'Random Perturbation with {n_bumps} Bumps', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/hicks_henne_demo.png', dpi=150)
    plt.close()
    print("Hicks-Henne demo saved.")
