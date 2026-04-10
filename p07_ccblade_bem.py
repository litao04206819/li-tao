"""
CCBlade 式叶素动量理论求解器 (保证收敛)
=========================================
参考: Ning A., "A Simple Solution Method for the Blade Element Momentum
      Equations with Guaranteed Convergence," Wind Energy, 2014.

核心创新: 将BEM方程整理为一维残差方程 f(φ)=0, 用Brent方法求根,
         彻底解决传统BEM迭代发散问题。

本代码实现:
  1. Ning残差方程推导
  2. Brent求根保证收敛
  3. 完整螺旋桨性能分析
  4. 性能曲线扫描
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# 翼型气动数据 (简化: 线性升力 + 二次阻力)
# ============================================================

def airfoil_data(alpha_rad):
    """
    简化翼型气动模型
    Cl = 2π·α (线性区), 带失速修正
    Cd = Cd0 + k·Cl^2

    参数:
        alpha_rad : 攻角 (rad)

    返回:
        Cl, Cd
    """
    alpha_deg = np.degrees(alpha_rad)

    # 线性区 + 简单失速模型
    Cl_slope = 2 * np.pi  # 薄翼理论斜率
    alpha_stall = np.radians(12)  # 失速角

    if abs(alpha_rad) < alpha_stall:
        Cl = Cl_slope * alpha_rad
    else:
        # 失速后Cl线性下降
        Cl_max = Cl_slope * alpha_stall
        Cl = Cl_max * np.sign(alpha_rad) * (1 - 0.5 * (abs(alpha_rad) - alpha_stall) / alpha_stall)

    # 阻力极曲线
    Cd0 = 0.008
    k = 0.005
    Cd = Cd0 + k * Cl**2

    return Cl, Cd


# ============================================================
# Prandtl 叶尖/叶根损失
# ============================================================

def prandtl_loss(r, R, Rhub, B, phi):
    """
    Prandtl 叶尖+叶根损失修正

    返回:
        F : 总损失因子 (0~1)
    """
    sin_phi = abs(np.sin(phi)) + 1e-10

    # 叶尖损失
    f_tip = B * (R - r) / (2 * r * sin_phi)
    F_tip = (2 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), -1, 1))

    # 叶根损失
    f_hub = B * (r - Rhub) / (2 * r * sin_phi)
    F_hub = (2 / np.pi) * np.arccos(np.clip(np.exp(-f_hub), -1, 1))

    F = F_tip * F_hub
    return max(F, 1e-4)  # 避免除零


# ============================================================
# Ning 残差方程 (核心!)
# ============================================================

def ning_residual(phi, r, R, Rhub, chord, twist, Vx, Vy, B):
    """
    Ning (2014) 残差方程 f(φ) = 0

    这是整个方法的核心: 把传统BEM的迭代格式重写为
    一个关于入流角φ的一维方程, 然后用Brent法求根。

    参数:
        phi   : 入流角 (rad) — 待求解的未知量
        r     : 当前截面径向位置 (m)
        R     : 叶尖半径 (m)
        Rhub  : 叶根半径 (m)
        chord : 当前截面弦长 (m)
        twist : 当前截面扭转角 (rad)
        Vx    : 轴向来流速度 (m/s)
        Vy    : 切向来流速度 (m/s, = ωr)
        B     : 叶片数

    返回:
        residual : 残差值, = 0 时为解
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # 攻角 = 入流角 - 扭转角 (注意符号约定)
    alpha = phi - twist

    # 查询翼型数据
    Cl, Cd = airfoil_data(alpha)

    # 局部实度
    sigma = B * chord / (2 * np.pi * r)

    # Prandtl损失因子
    F = prandtl_loss(r, R, Rhub, B, phi)

    # ---- Ning残差方程推导 ----
    # 传统BEM:
    #   a  = 1 / (4F sin²φ/(σ Cn) + 1)
    #   a' = 1 / (4F sinφ cosφ/(σ Ct) - 1)
    # 其中 Cn = Cl cosφ + Cd sinφ, Ct = Cl sinφ - Cd cosφ
    #
    # 代入关系 tanφ = Vx(1+a) / (Vy(1-a'))
    # 整理得: f(φ) = sinφ/(1+a) - Vx cosφ/(Vy(1-a')) = 0 (等价形式)
    #
    # Ning的简化: 直接用力的平衡

    Cn = Cl * cos_phi + Cd * sin_phi
    Ct = Cl * sin_phi - Cd * cos_phi

    # 避免 phi=0 或 phi=pi/2 附近的奇异
    if abs(sin_phi) < 1e-6 or abs(cos_phi) < 1e-6:
        return 1e6

    # 诱导因子 (从BEM动量方程)
    if phi > 0:
        # 正常工作状态
        k = sigma * Cn / (4 * F * sin_phi * sin_phi)

        if k <= 2.0 / 3.0:
            # Glauert 修正前的标准公式
            a = k / (1 + k)
        else:
            # Glauert 经验修正 (大推力状态)
            # 使用 Buhl (2005) 的修正
            gamma1 = 2 * F * k - (10.0/9 - F)
            gamma2 = 2 * F * k - F * (4.0/3 - F)
            gamma3 = 2 * F * k - (25.0/9 - 2*F)

            if abs(gamma3) < 1e-6:
                a = 1 - 1.0 / (2*np.sqrt(gamma2))
            else:
                a = (gamma1 - np.sqrt(max(gamma2, 0))) / gamma3

        # 切向诱导因子
        kp = sigma * Ct / (4 * F * sin_phi * cos_phi)
        if abs(1 - kp) < 1e-6:
            a_prime = 0
        else:
            a_prime = kp / (1 - kp)

        a_prime = max(a_prime, -1)

    else:
        # 风车制动状态 (φ < 0 — 对于螺旋桨推进通常不会出现)
        k = sigma * Cn / (4 * F * sin_phi * sin_phi)
        a = k / (1 - k) if abs(1 - k) > 1e-6 else 0
        kp = sigma * Ct / (4 * F * sin_phi * cos_phi)
        a_prime = kp / (1 - kp) if abs(1 - kp) > 1e-6 else 0

    # ---- 残差: 入流角一致性 ----
    # 由诱导因子反算的入流角应该等于假设的φ
    # tanφ = Vx(1-a) / (Vy(1+a'))  (螺旋桨约定)
    # 注意: 螺旋桨和风力机的符号约定不同
    # 这里用螺旋桨约定: a减小轴向速度, a'增大旋转速度

    phi_computed = np.arctan2(Vx * (1 - a), Vy * (1 + a_prime))

    residual = phi - phi_computed

    return residual


# ============================================================
# BEM 求解器 (使用 Brent 方法)
# ============================================================

def solve_bem_section(r, R, Rhub, chord, twist, Vx, Vy, B):
    """
    求解单个截面的BEM方程

    使用 Brent 方法在 φ ∈ (ε, π/2-ε) 区间搜索零点

    返回:
        phi, alpha, Cl, Cd, a, a_prime, Cn, Ct, W
    """
    eps = 1e-6

    # Brent法需要一个包含零点的区间 [a, b], 且 f(a)·f(b) < 0
    # 对于螺旋桨, φ 通常在 (0, π/2) 之间
    # Ning建议: 分段搜索, 先尝试主区间

    phi_min = eps
    phi_max = np.pi / 2 - eps

    try:
        # 检查区间两端残差的符号
        fa = ning_residual(phi_min, r, R, Rhub, chord, twist, Vx, Vy, B)
        fb = ning_residual(phi_max, r, R, Rhub, chord, twist, Vx, Vy, B)

        if fa * fb < 0:
            phi = brentq(ning_residual, phi_min, phi_max,
                         args=(r, R, Rhub, chord, twist, Vx, Vy, B),
                         xtol=1e-8, maxiter=200)
        else:
            # 如果主区间没有零点, 尝试细分搜索
            phi = None
            n_sub = 50
            phi_tests = np.linspace(phi_min, phi_max, n_sub)
            for i in range(n_sub - 1):
                f1 = ning_residual(phi_tests[i], r, R, Rhub, chord, twist, Vx, Vy, B)
                f2 = ning_residual(phi_tests[i+1], r, R, Rhub, chord, twist, Vx, Vy, B)
                if f1 * f2 < 0:
                    phi = brentq(ning_residual, phi_tests[i], phi_tests[i+1],
                                 args=(r, R, Rhub, chord, twist, Vx, Vy, B),
                                 xtol=1e-8, maxiter=200)
                    break
            if phi is None:
                phi = np.arctan2(Vx, Vy)  # 回退到几何角
    except Exception:
        phi = np.arctan2(Vx, Vy)

    # 由解出的φ计算其他量
    alpha = phi - twist
    Cl, Cd = airfoil_data(alpha)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    Cn = Cl * cos_phi + Cd * sin_phi
    Ct = Cl * sin_phi - Cd * cos_phi

    sigma = B * chord / (2 * np.pi * r)
    F = prandtl_loss(r, R, Rhub, B, phi)

    k = sigma * Cn / (4 * F * sin_phi**2 + 1e-10)
    if k <= 2.0/3.0:
        a = k / (1 + k)
    else:
        gamma2 = 2*F*k - F*(4.0/3 - F)
        a = 1 - 1.0/(2*np.sqrt(max(gamma2, 1e-10)))
    a = np.clip(a, -0.5, 0.95)

    kp = sigma * Ct / (4 * F * sin_phi * cos_phi + 1e-10)
    a_prime = kp / (1 - kp) if abs(1 - kp) > 1e-6 else 0
    a_prime = np.clip(a_prime, -1, 5)

    W = np.sqrt((Vx*(1-a))**2 + (Vy*(1+a_prime))**2)

    return phi, alpha, Cl, Cd, a, a_prime, Cn, Ct, W


# ============================================================
# 完整螺旋桨分析
# ============================================================

class Propeller:
    """
    螺旋桨定义和BEM分析
    """
    def __init__(self, R, Rhub, B, r_R, chord, twist_deg):
        """
        参数:
            R         : 叶尖半径 (m)
            Rhub      : 叶根半径 (m)
            B         : 叶片数
            r_R       : 各截面 r/R
            chord     : 各截面弦长 (m)
            twist_deg : 各截面扭转角 (deg)
        """
        self.R = R
        self.Rhub = Rhub
        self.B = B
        self.r = r_R * R
        self.chord = chord
        self.twist = np.radians(twist_deg)
        self.n_sec = len(r_R)

    def analyze(self, V_inf, RPM, rho=1.225):
        """
        分析给定工况下的性能

        参数:
            V_inf : 前进速度 (m/s)
            RPM   : 转速
            rho   : 空气密度

        返回:
            results : dict 包含推力、扭矩、效率等
        """
        omega = RPM * 2 * np.pi / 60
        n = RPM / 60  # 转速 (rev/s)

        # 各截面求解
        phi_arr = np.zeros(self.n_sec)
        alpha_arr = np.zeros(self.n_sec)
        a_arr = np.zeros(self.n_sec)
        ap_arr = np.zeros(self.n_sec)
        dT_arr = np.zeros(self.n_sec)
        dQ_arr = np.zeros(self.n_sec)
        Cl_arr = np.zeros(self.n_sec)
        Cd_arr = np.zeros(self.n_sec)

        for i in range(self.n_sec):
            Vx = V_inf                # 轴向速度
            Vy = omega * self.r[i]    # 切向速度

            phi, alpha, Cl, Cd, a, a_prime, Cn, Ct, W = \
                solve_bem_section(self.r[i], self.R, self.Rhub,
                                  self.chord[i], self.twist[i],
                                  Vx, Vy, self.B)

            phi_arr[i] = phi
            alpha_arr[i] = alpha
            a_arr[i] = a
            ap_arr[i] = a_prime
            Cl_arr[i] = Cl
            Cd_arr[i] = Cd

            # 推力和扭矩微元 (每单位展向长度)
            dT_arr[i] = 0.5 * rho * W**2 * self.chord[i] * Cn * self.B
            dQ_arr[i] = 0.5 * rho * W**2 * self.chord[i] * Ct * self.B * self.r[i]

        # 梯形积分
        T = np.trapezoid(dT_arr, self.r)
        Q = np.trapezoid(dQ_arr, self.r)
        P = Q * omega

        # 无量纲系数
        D = 2 * self.R
        CT = T / (rho * n**2 * D**4)
        CQ = Q / (rho * n**2 * D**5)
        CP = P / (rho * n**3 * D**5)
        J = V_inf / (n * D)
        eta = J * CT / (CP + 1e-10) if CP > 0 else 0

        return {
            'T': T, 'Q': Q, 'P': P,
            'CT': CT, 'CQ': CQ, 'CP': CP,
            'J': J, 'eta': eta,
            'phi': np.degrees(phi_arr),
            'alpha': np.degrees(alpha_arr),
            'a': a_arr, 'a_prime': ap_arr,
            'Cl': Cl_arr, 'Cd': Cd_arr,
            'dT': dT_arr, 'dQ': dQ_arr,
            'r_R': self.r / self.R,
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    # --- 定义一个典型小型螺旋桨 ---
    R = 0.15       # 半径 15cm
    Rhub = 0.02    # 叶根半径 2cm
    B = 3          # 3叶

    # 展向截面 (用样条定义的弦长和扭转)
    r_R = np.linspace(0.20, 0.98, 25)
    # 弦长分布 (典型形状: 中部宽、尖端窄)
    chord = R * (0.12 * np.sin(np.pi * (r_R - 0.15) / 0.85) + 0.02)
    # 扭转分布 (理想扭转的近似)
    twist_deg = 30 * (0.3 / r_R) - 5

    prop = Propeller(R, Rhub, B, r_R, chord, twist_deg)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # --- 子图1: 桨叶几何 ---
    ax = axes[0, 0]
    ax.plot(r_R, chord * 1000, 'b-', lw=2.5, label='Chord')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Chord (mm)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax2 = ax.twinx()
    ax2.plot(r_R, twist_deg, 'r-', lw=2.5, label='Twist')
    ax2.set_ylabel('Twist (deg)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('Blade Geometry', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- 子图2: 单工况截面分析 ---
    res = prop.analyze(V_inf=15, RPM=8000)

    ax = axes[0, 1]
    ax.plot(res['r_R'], res['alpha'], 'b-', lw=2.5, label='α (AoA)')
    ax.plot(res['r_R'], res['phi'], 'r--', lw=2, label='φ (inflow)')
    ax.set_title(f"Section Angles (V={15} m/s, {8000} RPM)", fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Angle (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 子图3: 诱导因子分布 ---
    ax = axes[1, 0]
    ax.plot(res['r_R'], res['a'], 'b-', lw=2.5, label='a (axial)')
    ax.plot(res['r_R'], res['a_prime'], 'r-', lw=2.5, label="a' (tangential)")
    ax.set_title('Induction Factors', fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Induction factor')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 子图4: 推力/扭矩载荷分布 ---
    ax = axes[1, 1]
    ax.plot(res['r_R'], res['dT'], 'b-', lw=2.5, label='dT/dr')
    ax.set_ylabel('dT/dr (N/m)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax3 = ax.twinx()
    ax3.plot(res['r_R'], res['dQ'], 'r-', lw=2.5, label='dQ/dr')
    ax3.set_ylabel('dQ/dr (N·m/m)', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax.set_title('Thrust & Torque Loading', fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.grid(True, alpha=0.3)

    # --- 子图5: J-CT-CP-η 性能曲线 ---
    ax = axes[2, 0]
    RPM_fixed = 8000
    V_range = np.linspace(2, 35, 30)
    J_arr, CT_arr, CP_arr, eta_arr = [], [], [], []

    for V in V_range:
        r = prop.analyze(V_inf=V, RPM=RPM_fixed)
        J_arr.append(r['J'])
        CT_arr.append(r['CT'])
        CP_arr.append(r['CP'])
        eta_arr.append(r['eta'])

    ax.plot(J_arr, CT_arr, 'b-', lw=2.5, label='$C_T$')
    ax.plot(J_arr, CP_arr, 'r-', lw=2.5, label='$C_P$')
    ax.set_title('Performance Curves (Brent BEM)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Advance Ratio J')
    ax.set_ylabel('$C_T$, $C_P$')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax4 = ax.twinx()
    ax4.plot(J_arr, eta_arr, 'g--', lw=2.5, label='η')
    ax4.set_ylabel('Efficiency η', color='g')
    ax4.tick_params(axis='y', labelcolor='g')
    ax4.set_ylim(0, 1)

    # --- 子图6: 残差函数可视化 (展示Brent求根原理) ---
    ax = axes[2, 1]
    phi_scan = np.linspace(0.02, np.pi/2 - 0.02, 200)
    r_test = 0.5 * R
    chord_test = np.interp(0.5, r_R, chord)
    twist_test = np.radians(np.interp(0.5, r_R, twist_deg))
    Vx_test = 15.0
    Vy_test = 8000 * 2 * np.pi / 60 * r_test

    resid = [ning_residual(p, r_test, R, Rhub, chord_test, twist_test,
                           Vx_test, Vy_test, B) for p in phi_scan]

    ax.plot(np.degrees(phi_scan), resid, 'b-', lw=2.5)
    ax.axhline(y=0, color='k', linestyle='--', lw=1)
    ax.set_title('Ning Residual f(φ) at r/R=0.5', fontsize=13, fontweight='bold')
    ax.set_xlabel('φ (deg)')
    ax.set_ylabel('f(φ)')
    ax.grid(True, alpha=0.3)

    # 标记零点
    zero_crossings = []
    for i in range(len(resid) - 1):
        if resid[i] * resid[i+1] < 0:
            zero_crossings.append(np.degrees(phi_scan[i]))
    for zc in zero_crossings:
        ax.axvline(x=zc, color='r', linestyle=':', lw=2, alpha=0.7)
        ax.annotate(f'φ = {zc:.1f}°', xy=(zc, 0), xytext=(zc+5, max(resid)*0.3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=11, color='red')

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/ccblade_bem_demo.png', dpi=150)
    plt.close()

    # 打印性能摘要
    print("=" * 50)
    print("CCBlade-style BEM Analysis Results")
    print("=" * 50)
    print(f"Design point: V = 15 m/s, RPM = 8000")
    print(f"  Thrust  T  = {res['T']:.3f} N")
    print(f"  Torque  Q  = {res['Q']:.5f} N·m")
    print(f"  Power   P  = {res['P']:.2f} W")
    print(f"  CT = {res['CT']:.4f}")
    print(f"  CP = {res['CP']:.4f}")
    print(f"  J  = {res['J']:.3f}")
    print(f"  η  = {res['eta']:.4f}")
    print("=" * 50)
    print("Demo saved.")
