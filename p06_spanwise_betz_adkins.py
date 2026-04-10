"""
桨叶展向分布参数化 + Betz-Prandtl 最优分布
============================================
参考:
  - Ganguli R. & Chopra I., J. Aircraft, 1995 (展向B样条)
  - Betz A., 1919; Prandtl L., 叶尖损失修正
  - Adkins C.N. & Liebeck R.H., J. Propulsion and Power, 1994

核心思想: 在展向若干控制截面定义弦长/扭转, B样条插值;
         Betz-Prandtl分布提供理论最优参考。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ============================================================
# 展向B样条/样条插值
# ============================================================

def spanwise_distribution(r_ctrl, val_ctrl, n_pts=101, r_range=(0.15, 1.0)):
    """
    用三次样条插值展向分布

    参数:
        r_ctrl  : array  控制截面径向位置 r/R
        val_ctrl: array  控制截面上的值 (弦长、扭转等)
        n_pts   : int    输出点数
        r_range : tuple  径向范围

    返回:
        r_out, val_out
    """
    cs = CubicSpline(r_ctrl, val_ctrl, bc_type='natural')
    r_out = np.linspace(r_range[0], r_range[1], n_pts)
    val_out = cs(r_out)
    return r_out, val_out


# ============================================================
# Prandtl 叶尖损失因子
# ============================================================

def prandtl_tip_loss(r_R, B, phi_tip, R=1.0):
    """
    Prandtl 叶尖损失修正因子 F

    参数:
        r_R     : array  r/R
        B       : int    叶片数
        phi_tip : float  叶尖处入流角 (rad)
        R       : float  半径

    返回:
        F : array  损失因子 (0~1)
    """
    r = r_R * R
    exponent = -B * (R - r) / (2 * r * np.sin(phi_tip) + 1e-10)
    F = (2 / np.pi) * np.arccos(np.clip(np.exp(exponent), -1, 1))
    return F


# ============================================================
# Betz 最优环量分布 (简化模型)
# ============================================================

def betz_optimal_circulation(r_R, V_inf, omega, B, R=1.0):
    """
    Betz最优环量分布 (含Prandtl修正)

    简化假设: 轻载条件, 无粘

    参数:
        r_R   : array  r/R
        V_inf : float  前进速度 (m/s)
        omega : float  角速度 (rad/s)
        B     : int    叶片数
        R     : float  半径 (m)

    返回:
        Gamma : 环量分布
        phi   : 入流角分布
        F     : Prandtl损失因子
    """
    r = r_R * R
    # 入流角 (无诱导速度的几何角)
    phi = np.arctan2(V_inf, omega * r)
    phi_tip = np.arctan2(V_inf, omega * R)

    # Prandtl修正
    F = prandtl_tip_loss(r_R, B, phi_tip, R)

    # 简化的Betz环量 (正比于 r·F·sin(phi))
    # 精确形式需要Goldstein因子, 这里用Prandtl近似
    Gamma = 2 * np.pi * r * V_inf * np.sin(phi) * np.cos(phi) * F / B

    return Gamma, phi, F


# ============================================================
# Adkins-Liebeck 反设计
# ============================================================

def adkins_liebeck_design(r_R, V_inf, omega, T_required, B,
                           Cl_design, Cd_design, R=1.0):
    """
    Adkins-Liebeck 最小诱导损失螺旋桨设计

    参数:
        r_R        : array  r/R (不含叶根和叶尖)
        V_inf      : float  前进速度 (m/s)
        omega      : float  角速度 (rad/s)
        T_required : float  所需推力 (N)
        B          : int    叶片数
        Cl_design  : float  设计升力系数
        Cd_design  : float  对应阻力系数
        R          : float  半径 (m)

    返回:
        chord  : 弦长分布 (m)
        beta   : 桨距角分布 (deg)
        alpha  : 设计攻角 (deg)
        phi    : 入流角 (deg)
    """
    rho = 1.225  # 空气密度
    r = r_R * R
    lambda_r = omega * r / V_inf  # 局部速比

    # 迭代求解入流因子
    phi_tip = np.arctan2(V_inf, omega * R)
    F = prandtl_tip_loss(r_R, B, phi_tip, R)

    # 初始入流角估计
    phi = np.arctan2(V_inf, omega * r)

    # 简单迭代 (3次通常收敛)
    for _ in range(10):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # 轴向和周向诱导因子
        # 简化: 假设 Cl >> Cd
        a = 1 / (1 + 4 * F * sin_phi**2 / (B * Cl_design * cos_phi / (2*np.pi*r) + 1e-10) + 1)
        a = np.clip(a, 0, 0.4)

        a_prime = 1 / (4 * F * cos_phi / (B * Cl_design / (2*np.pi*r) + 1e-10) - 1 + 1e-10)
        a_prime = np.clip(a_prime, 0, 0.4)

        # 更新入流角
        phi = np.arctan2(V_inf * (1 + a), omega * r * (1 - a_prime))

    # 合成来流速度
    W = np.sqrt((V_inf * (1 + a))**2 + (omega * r * (1 - a_prime))**2)

    # 弦长 (由升力需求确定)
    # 推力微元: dT = 0.5 * rho * W^2 * c * Cl * cos(phi) * dr - drag部分
    # 简化: 均匀推力分布
    chord = 8 * np.pi * r * F * np.sin(phi) / (B * Cl_design) * \
            (np.cos(phi) - lambda_r * np.sin(phi)) / \
            (np.sin(phi) + lambda_r * np.cos(phi) + 1e-10)
    chord = np.abs(chord)

    # 限制弦长在合理范围
    chord = np.clip(chord, 0.01 * R, 0.5 * R)

    # 桨距角 = 入流角 + 设计攻角
    alpha_design = np.degrees(np.arctan2(Cl_design, 2 * np.pi))  # 薄翼理论估计攻角
    beta = np.degrees(phi) + alpha_design

    return chord, beta, np.full_like(r_R, alpha_design), np.degrees(phi)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    r_R = np.linspace(0.15, 0.98, 200)

    # ==================== 展向分布参数化 ====================

    # --- 示例1: 弦长分布 ---
    ax = axes[0, 0]
    r_ctrl = np.array([0.15, 0.30, 0.50, 0.70, 0.85, 0.98])

    chord_sets = {
        'Rect-like':   [0.08, 0.10, 0.10, 0.10, 0.09, 0.04],
        'Tapered':     [0.12, 0.11, 0.09, 0.07, 0.05, 0.02],
        'High-perf':   [0.06, 0.12, 0.11, 0.08, 0.05, 0.01],
    }
    for name, vals in chord_sets.items():
        r, c = spanwise_distribution(r_ctrl, vals)
        ax.plot(r, c, lw=2.5, label=name)
        ax.plot(r_ctrl, vals, 'o', ms=7, color=ax.get_lines()[-1].get_color())

    ax.set_title('Chord Distribution c(r/R) — Spline Interpolation',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('c/R')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例2: 扭转角分布 ---
    ax = axes[0, 1]
    twist_sets = {
        'Linear':    [30, 24, 18, 12, 8, 5],
        'Ideal':     [35, 22, 14, 9, 6, 3],
        'Washout':   [25, 20, 15, 10, 6, 2],
    }
    for name, vals in twist_sets.items():
        r, th = spanwise_distribution(r_ctrl, vals)
        ax.plot(r, th, lw=2.5, label=name)
        ax.plot(r_ctrl, vals, 's', ms=7, color=ax.get_lines()[-1].get_color())

    ax.set_title('Twist Distribution θ(r/R) — Spline Interpolation',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('θ (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ==================== Betz-Prandtl ====================

    # --- 示例3: Prandtl叶尖损失因子 ---
    ax = axes[1, 0]
    for B_val in [2, 3, 4, 6]:
        F = prandtl_tip_loss(r_R, B_val, phi_tip=np.radians(10))
        ax.plot(r_R, F, lw=2.5, label=f'B = {B_val}')

    ax.set_title('Prandtl Tip-Loss Factor F(r/R)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('F')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # --- 示例4: 最优环量分布 ---
    ax = axes[1, 1]
    V_inf = 15.0   # m/s
    omega = 200.0  # rad/s
    B = 3

    Gamma, phi, F = betz_optimal_circulation(r_R, V_inf, omega, B)
    Gamma_norm = Gamma / np.max(Gamma)

    ax.plot(r_R, Gamma_norm, 'b-', lw=2.5, label='Betz-Prandtl optimal')
    ax.plot(r_R, F, 'g--', lw=2, label='Prandtl factor F')
    ax.set_title('Betz Optimal Circulation Distribution',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Γ / Γ_max')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ==================== Adkins-Liebeck ====================

    # --- 示例5: 设计弦长和桨距角 ---
    r_design = np.linspace(0.20, 0.95, 100)
    chord, beta, alpha, phi_deg = adkins_liebeck_design(
        r_design, V_inf=20, omega=300, T_required=50,
        B=3, Cl_design=0.6, Cd_design=0.012, R=0.15
    )

    ax = axes[2, 0]
    ax.plot(r_design, chord * 1000, 'b-', lw=2.5)
    ax.set_title('Adkins-Liebeck Design: Chord Distribution',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Chord (mm)')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(r_design, beta, 'r-', lw=2.5, label='β (pitch angle)')
    ax.plot(r_design, phi_deg, 'b--', lw=2, label='φ (inflow angle)')
    ax.set_title('Adkins-Liebeck Design: Pitch & Inflow Angle',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Angle (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/spanwise_betz_adkins_demo.png', dpi=150)
    plt.close()
    print("Spanwise + Betz-Prandtl + Adkins-Liebeck demo saved.")
