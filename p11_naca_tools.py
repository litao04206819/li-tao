"""
NACA 翼型族生成器 + 坐标处理工具
==================================
覆盖:
  - NACA 4位翼型 (如 NACA 2412)
  - NACA 5位翼型 (如 NACA 23012)
  - 翼型坐标读写 (Selig/Lednicer格式)
  - 翼型几何分析 (厚度、弯度、前缘半径)
  - 翼型坐标插值/重采样
  - CST逆向拟合

参考: Abbott I.H. & von Doenhoff A.E., "Theory of Wing Sections,"
      Dover, 1959. (NACA翼型族定义)
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb

# ============================================================
# NACA 4位翼型
# ============================================================

def naca4(designation, n_pts=101, cosine_spacing=True):
    """
    生成NACA 4位翼型坐标

    designation: str, 如 '2412'
      第1位: 最大弯度 m (占弦长百分比)
      第2位: 最大弯度位置 p (占弦长十分位)
      第3-4位: 最大厚度 t (占弦长百分比)

    返回: x, y_upper, y_lower, y_camber
    """
    m = int(designation[0]) / 100.0    # 最大弯度
    p = int(designation[1]) / 10.0     # 最大弯度位置
    t = int(designation[2:4]) / 100.0  # 最大厚度

    if cosine_spacing:
        # 半余弦分布: 前缘密后缘疏
        beta = np.linspace(0, np.pi, n_pts)
        x = 0.5 * (1 - np.cos(beta))
    else:
        x = np.linspace(0, 1, n_pts)

    # 厚度分布 (对称翼型部分)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                   + 0.2843*x**3 - 0.1015*x**4)

    # 弯度线
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)  # 弯度线斜率

    if m > 0 and p > 0:
        front = x <= p
        rear = x > p
        yc[front] = m / p**2 * (2*p*x[front] - x[front]**2)
        yc[rear] = m / (1-p)**2 * ((1-2*p) + 2*p*x[rear] - x[rear]**2)
        dyc[front] = 2*m / p**2 * (p - x[front])
        dyc[rear] = 2*m / (1-p)**2 * (p - x[rear])

    # 上下表面 (垂直于弯度线方向)
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl, x, yc


# ============================================================
# NACA 5位翼型
# ============================================================

def naca5(designation, n_pts=101, cosine_spacing=True):
    """
    生成NACA 5位翼型坐标

    designation: str, 如 '23012'
      第1位 × 3/2: 设计升力系数 Cl_i (十分位)
      第2-3位: 最大弯度位置 (弦长百分比的一半)
      第4-5位: 最大厚度 (弦长百分比)

    仅实现标准(非reflex)弯度线
    """
    Cl_design = int(designation[0]) * (3.0 / 20.0)
    p = int(designation[1:3]) / 200.0  # 最大弯度位置
    t = int(designation[3:5]) / 100.0

    if cosine_spacing:
        beta = np.linspace(0, np.pi, n_pts)
        x = 0.5 * (1 - np.cos(beta))
    else:
        x = np.linspace(0, 1, n_pts)

    # 厚度分布
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                   + 0.2843*x**3 - 0.1015*x**4)

    # 5位系列弯度线系数 (标准型)
    # 使用近似的 r, k1 值
    naca5_data = {
        210: (0.0580, 361.4),
        220: (0.1260, 51.64),
        230: (0.2025, 15.957),
        240: (0.2900, 6.643),
        250: (0.3910, 3.230),
    }

    key = int(designation[0:3])
    if key in naca5_data:
        r, k1 = naca5_data[key]
    else:
        r, k1 = 0.2025, 15.957  # 默认23xxx

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    front = x <= r
    rear = x > r

    yc[front] = k1/6 * (x[front]**3 - 3*r*x[front]**2 + r**2*(3-r)*x[front])
    yc[rear] = k1*r**3/6 * (1 - x[rear])

    dyc[front] = k1/6 * (3*x[front]**2 - 6*r*x[front] + r**2*(3-r))
    dyc[rear] = -k1*r**3/6

    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl, x, yc


# ============================================================
# 翼型坐标读写工具
# ============================================================

def read_airfoil_dat(filename):
    """
    读取Selig格式翼型坐标文件

    Selig格式: 从尾缘→上表面→前缘→下表面→尾缘
    返回: x_upper, y_upper, x_lower, y_lower (各从前缘到尾缘)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    name = lines[0].strip()
    coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    coords = np.array(coords)

    # 找前缘 (x最小的点)
    le_idx = np.argmin(coords[:, 0])

    # 上表面: 从前缘到尾缘 (Selig格式中上表面在前半段, 需要反转)
    x_upper = coords[:le_idx+1, 0][::-1]
    y_upper = coords[:le_idx+1, 1][::-1]

    # 下表面: 从前缘到尾缘
    x_lower = coords[le_idx:, 0]
    y_lower = coords[le_idx:, 1]

    return x_upper, y_upper, x_lower, y_lower, name


def write_airfoil_dat(filename, x_upper, y_upper, x_lower, y_lower, name="Airfoil"):
    """写出Selig格式翼型坐标文件"""
    with open(filename, 'w') as f:
        f.write(f"{name}\n")
        # 上表面: 尾缘→前缘
        for i in range(len(x_upper)-1, -1, -1):
            f.write(f"  {x_upper[i]:.6f}  {y_upper[i]:.6f}\n")
        # 下表面: 前缘→尾缘 (跳过前缘重复点)
        for i in range(1, len(x_lower)):
            f.write(f"  {x_lower[i]:.6f}  {y_lower[i]:.6f}\n")


# ============================================================
# 翼型几何分析
# ============================================================

def analyze_airfoil(x_upper, y_upper, x_lower, y_lower):
    """
    分析翼型几何特征

    返回:
        dict: 厚度、弯度、前缘半径等
    """
    # 在公共x坐标上插值
    x_common = np.linspace(0.001, 1, 500)
    yu_interp = np.interp(x_common, x_upper, y_upper)
    yl_interp = np.interp(x_common, x_lower, y_lower)

    # 厚度分布
    thickness = yu_interp - yl_interp
    max_t = thickness.max()
    max_t_pos = x_common[np.argmax(thickness)]

    # 弯度线
    camber = 0.5 * (yu_interp + yl_interp)
    max_camber = np.max(np.abs(camber))
    max_camber_pos = x_common[np.argmax(np.abs(camber))]

    # 前缘半径 (由前几个点的二次拟合估算)
    n_le = min(20, len(x_upper) // 4)
    x_le = x_upper[:n_le]
    y_le_up = y_upper[:n_le]
    # 拟合 y^2 = 2*r_le*x 在前缘附近
    if len(x_le) > 3 and x_le[1] > 0:
        coef = np.polyfit(x_le[1:n_le], y_le_up[1:n_le]**2, 1)
        r_le = coef[0] / 2
    else:
        r_le = 0

    # 尾缘厚度
    te_thickness = abs(y_upper[-1] - y_lower[-1])

    # 尾缘角
    if len(x_upper) > 2:
        dy_up = y_upper[-1] - y_upper[-3]
        dx_up = x_upper[-1] - x_upper[-3]
        dy_lo = y_lower[-1] - y_lower[-3]
        dx_lo = x_lower[-1] - x_lower[-3]
        te_angle = np.degrees(np.arctan2(dy_up, dx_up) - np.arctan2(dy_lo, dx_lo))
    else:
        te_angle = 0

    return {
        'max_thickness': max_t,
        'max_thickness_pos': max_t_pos,
        'max_camber': max_camber,
        'max_camber_pos': max_camber_pos,
        'le_radius': abs(r_le),
        'te_thickness': te_thickness,
        'te_angle': te_angle,
    }


# ============================================================
# CST 逆向拟合
# ============================================================

def fit_cst_from_coords(x_upper, y_upper, x_lower, y_lower, order=6):
    """
    从翼型坐标逆向拟合CST系数

    返回:
        A_upper, A_lower, fit_error_upper, fit_error_lower
    """
    def _fit_one_side(x_data, y_data, n_order, N1=0.5, N2=1.0):
        # 排除端点0和1处的奇异
        mask = (x_data > 1e-4) & (x_data < 0.999)
        x = x_data[mask]
        y = y_data[mask]

        C = x**N1 * (1 - x)**N2
        B = np.zeros((len(x), n_order + 1))
        for k in range(n_order + 1):
            B[:, k] = C * comb(n_order, k) * x**k * (1-x)**(n_order-k)

        A, _, _, _ = np.linalg.lstsq(B, y, rcond=None)

        # 拟合误差
        y_fit = B @ A
        error = np.sqrt(np.mean((y - y_fit)**2))
        return A, error

    A_up, err_up = _fit_one_side(x_upper, y_upper, order)
    A_lo, err_lo = _fit_one_side(x_lower, y_lower, order)

    return A_up, A_lo, err_up, err_lo


# ============================================================
# 翼型重采样/插值
# ============================================================

def resample_airfoil(x_in, y_in, n_new=101, cosine=True):
    """
    对翼型单面坐标重采样

    cosine=True: 前缘密后缘疏 (适合CFD网格)
    """
    if cosine:
        beta = np.linspace(0, np.pi, n_new)
        x_new = 0.5 * (1 - np.cos(beta))
    else:
        x_new = np.linspace(0, 1, n_new)

    # 确保输入单调递增
    sort_idx = np.argsort(x_in)
    x_sorted = x_in[sort_idx]
    y_sorted = y_in[sort_idx]

    # 去除重复x
    unique_mask = np.diff(x_sorted, prepend=-1) > 1e-10
    x_sorted = x_sorted[unique_mask]
    y_sorted = y_sorted[unique_mask]

    y_new = np.interp(x_new, x_sorted, y_sorted)
    return x_new, y_new


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))

    # --- 示例1: NACA 4位系列 ---
    ax = axes[0, 0]
    airfoils_4 = ['0012', '2412', '4412', '6412']
    for name in airfoils_4:
        xu, yu, xl, yl, xc, yc = naca4(name)
        ax.plot(xu, yu, lw=2, label=f'NACA {name}')
        ax.plot(xl, yl, lw=2, color=ax.get_lines()[-1].get_color())
    ax.set_aspect('equal')
    ax.set_title('NACA 4-Digit Series', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- 示例2: NACA 5位系列 ---
    ax = axes[0, 1]
    airfoils_5 = ['23012', '23015', '23018', '23021']
    for name in airfoils_5:
        xu, yu, xl, yl, xc, yc = naca5(name)
        ax.plot(xu, yu, lw=2, label=f'NACA {name}')
        ax.plot(xl, yl, lw=2, color=ax.get_lines()[-1].get_color())
    ax.set_aspect('equal')
    ax.set_title('NACA 5-Digit Series', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- 示例3: 弯度线对比 ---
    ax = axes[1, 0]
    for name in ['0012', '2412', '4412', '6412']:
        _, _, _, _, xc, yc = naca4(name)
        ax.plot(xc, yc, lw=2.5, label=f'NACA {name}')
    ax.set_title('Camber Lines (4-Digit)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c'); ax.set_ylabel('y_c/c')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- 示例4: 几何分析 ---
    ax = axes[1, 1]
    analysis_names = ['0012', '2412', '4415', '6418']
    bar_data = {'t/c': [], 'camber': [], 'r_LE×100': []}
    for name in analysis_names:
        xu, yu, xl, yl, _, _ = naca4(name)
        info = analyze_airfoil(xu, yu, xl, yl)
        bar_data['t/c'].append(info['max_thickness'] * 100)
        bar_data['camber'].append(info['max_camber'] * 100)
        bar_data['r_LE×100'].append(info['le_radius'] * 100)

    x_bar = np.arange(len(analysis_names))
    w = 0.25
    for i, (key, vals) in enumerate(bar_data.items()):
        ax.bar(x_bar + i*w, vals, w, label=key)
    ax.set_xticks(x_bar + w)
    ax.set_xticklabels([f'NACA {n}' for n in analysis_names])
    ax.set_title('Geometric Analysis', fontsize=13, fontweight='bold')
    ax.set_ylabel('% chord')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- 示例5: CST逆向拟合 ---
    ax = axes[2, 0]
    xu_orig, yu_orig, xl_orig, yl_orig, _, _ = naca4('2412', n_pts=201)
    ax.plot(xu_orig, yu_orig, 'k-', lw=3, alpha=0.3, label='NACA 2412 original')
    ax.plot(xl_orig, yl_orig, 'k-', lw=3, alpha=0.3)

    for order in [4, 6, 8]:
        A_up, A_lo, err_up, err_lo = fit_cst_from_coords(
            xu_orig, yu_orig, xl_orig, yl_orig, order=order)

        # 重建
        psi = np.linspace(0, 1, 201)
        C = psi**0.5 * (1-psi)**1.0
        S_up = sum(A_up[k] * comb(order, k) * psi**k * (1-psi)**(order-k)
                   for k in range(order+1))
        S_lo = sum(A_lo[k] * comb(order, k) * psi**k * (1-psi)**(order-k)
                   for k in range(order+1))
        yu_fit = C * S_up
        yl_fit = C * S_lo

        ax.plot(psi, yu_fit, '--', lw=2,
                label=f'CST n={order} (err={err_up:.1e})')
        ax.plot(psi, yl_fit, '--', lw=2, color=ax.get_lines()[-1].get_color())

    ax.set_aspect('equal')
    ax.set_title('CST Inverse Fitting of NACA 2412', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # --- 示例6: 余弦 vs 均匀采样 ---
    ax = axes[2, 1]
    xu_cos, yu_cos, _, _, _, _ = naca4('4412', n_pts=31, cosine_spacing=True)
    xu_uni, yu_uni, _, _, _, _ = naca4('4412', n_pts=31, cosine_spacing=False)
    xu_ref, yu_ref, _, _, _, _ = naca4('4412', n_pts=201)

    ax.plot(xu_ref, yu_ref, 'k-', lw=1, alpha=0.3, label='Reference (201 pts)')
    ax.plot(xu_cos, yu_cos, 'bo-', ms=5, lw=1.5, label='Cosine (31 pts)')
    ax.plot(xu_uni, yu_uni, 'rs-', ms=5, lw=1.5, label='Uniform (31 pts)')
    ax.set_aspect('equal')
    ax.set_title('Cosine vs Uniform Spacing', fontsize=13, fontweight='bold')
    ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 放大前缘
    ax_inset = ax.inset_axes([0.35, 0.15, 0.4, 0.5])
    ax_inset.plot(xu_ref, yu_ref, 'k-', lw=1, alpha=0.3)
    ax_inset.plot(xu_cos, yu_cos, 'bo-', ms=5, lw=1.5)
    ax_inset.plot(xu_uni, yu_uni, 'rs-', ms=5, lw=1.5)
    ax_inset.set_xlim(-0.01, 0.15)
    ax_inset.set_ylim(0.01, 0.10)
    ax_inset.set_aspect('equal')
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_title('Leading edge zoom', fontsize=8)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/naca_tools_demo.png', dpi=150)
    plt.close()

    # 导出示例坐标文件
    xu, yu, xl, yl, _, _ = naca4('2412', n_pts=101)
    write_airfoil_dat('/home/claude/parametric_codes/python/naca2412.dat',
                      xu, yu, xl, yl, 'NACA 2412')
    print("NACA tools demo saved.")
    print("Exported: naca2412.dat")
