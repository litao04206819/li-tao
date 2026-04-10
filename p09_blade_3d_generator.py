"""
三维螺旋桨桨叶几何生成器
=========================
整合所有参数化方法, 生成完整的三维桨叶几何并输出可导入CAD/CFD的坐标文件。

工作流:
  CST翼型参数化 → 展向B样条分布 → 三维叶片坐标 → STL/CSV输出
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import comb
from scipy.interpolate import CubicSpline

# ============================================================
# CST翼型生成 (复用p01)
# ============================================================

def bernstein_basis(n, k, psi):
    return comb(n, k) * psi**k * (1 - psi)**(n - k)

def cst_airfoil(A_upper, A_lower, n_pts=101, N1=0.5, N2=1.0, dz_te=0.001):
    psi = np.linspace(0, 1, n_pts)
    C = psi**N1 * (1 - psi)**N2
    n_up = len(A_upper) - 1
    n_lo = len(A_lower) - 1
    S_up = sum(A_upper[k] * bernstein_basis(n_up, k, psi) for k in range(n_up + 1))
    S_lo = sum(A_lower[k] * bernstein_basis(n_lo, k, psi) for k in range(n_lo + 1))
    yu = C * S_up + psi * dz_te
    yl = C * S_lo - psi * dz_te
    return psi, yu, yl


# ============================================================
# 三维桨叶生成器
# ============================================================

class BladeGeometry:
    """
    三维螺旋桨桨叶几何生成器

    坐标系约定:
      - X 轴: 旋转轴方向 (推力方向)
      - Y 轴: 展向 (从叶根到叶尖)
      - Z 轴: 弦向/法向
    """

    def __init__(self, R, Rhub, B=3):
        """
        参数:
            R    : 叶尖半径 (m)
            Rhub : 叶根半径 (m)
            B    : 叶片数
        """
        self.R = R
        self.Rhub = Rhub
        self.B = B
        self.sections = []

    def add_section(self, r_R, chord, twist_deg, A_upper, A_lower,
                    sweep=0.0, dihedral=0.0, pitch_axis=0.25):
        """
        添加一个展向控制截面

        参数:
            r_R       : 径向位置 r/R
            chord     : 弦长 (m)
            twist_deg : 扭转角 (deg, 正值为前缘上仰)
            A_upper   : CST上表面系数
            A_lower   : CST下表面系数
            sweep     : 后掠偏移 (m, 正值=后掠)
            dihedral  : 下反偏移 (m, 正值=下反)
            pitch_axis: 变距轴位置 (占弦长比, 通常0.25)
        """
        self.sections.append({
            'r_R': r_R,
            'chord': chord,
            'twist': np.radians(twist_deg),
            'A_upper': A_upper,
            'A_lower': A_lower,
            'sweep': sweep,
            'dihedral': dihedral,
            'pitch_axis': pitch_axis,
        })
        # 按径向位置排序
        self.sections.sort(key=lambda s: s['r_R'])

    def generate_from_distributions(self, r_ctrl, chord_ctrl, twist_ctrl,
                                     A_upper, A_lower, n_span=20,
                                     sweep_ctrl=None, pitch_axis=0.25):
        """
        从展向分布控制点自动生成所有截面

        参数:
            r_ctrl     : 控制截面径向位置 r/R
            chord_ctrl : 控制截面弦长 (m)
            twist_ctrl : 控制截面扭转角 (deg)
            A_upper    : CST上表面系数 (所有截面统一)
            A_lower    : CST下表面系数
            n_span     : 生成的展向截面数
            sweep_ctrl : 后掠分布 (m), 若None则无后掠
        """
        self.sections = []

        # 三次样条插值
        cs_chord = CubicSpline(r_ctrl, chord_ctrl, bc_type='natural')
        cs_twist = CubicSpline(r_ctrl, twist_ctrl, bc_type='natural')

        if sweep_ctrl is not None:
            cs_sweep = CubicSpline(r_ctrl, sweep_ctrl, bc_type='natural')

        r_span = np.linspace(r_ctrl[0], r_ctrl[-1], n_span)

        for r in r_span:
            sweep = float(cs_sweep(r)) if sweep_ctrl is not None else 0.0
            self.add_section(
                r_R=r,
                chord=float(cs_chord(r)),
                twist_deg=float(cs_twist(r)),
                A_upper=A_upper,
                A_lower=A_lower,
                sweep=sweep,
                pitch_axis=pitch_axis,
            )

    def build_surface(self, n_chordwise=61):
        """
        构建三维表面点阵

        返回:
            X, Y, Z : (n_span, 2*n_chordwise-1) 三维坐标网格
                       上下表面拼接成闭合截面
        """
        n_span = len(self.sections)
        n_total = 2 * n_chordwise - 2  # 上表面 + 下表面(反向, 去掉首尾重复)

        X = np.zeros((n_span, n_total))
        Y = np.zeros((n_span, n_total))
        Z = np.zeros((n_span, n_total))

        for i, sec in enumerate(self.sections):
            r = sec['r_R'] * self.R
            c = sec['chord']
            tw = sec['twist']
            pa = sec['pitch_axis']
            sw = sec['sweep']
            dh = sec['dihedral']

            # 生成CST翼型截面
            psi, yu, yl = cst_airfoil(sec['A_upper'], sec['A_lower'],
                                       n_pts=n_chordwise)

            # 拼接成闭合截面: 上表面(前缘→尾缘) + 下表面(尾缘→前缘)
            x_sec = np.concatenate([psi, psi[-2:0:-1]])  # 去掉首尾重复
            y_sec = np.concatenate([yu, yl[-2:0:-1]])

            # 缩放到实际弦长
            x_sec = x_sec * c
            y_sec = y_sec * c

            # 平移使变距轴在原点
            x_sec -= pa * c
            # 此时 x_sec 的范围大约是 [-pa*c, (1-pa)*c]

            # 旋转 (扭转)
            cos_tw = np.cos(tw)
            sin_tw = np.sin(tw)
            x_rot = x_sec * cos_tw - y_sec * sin_tw
            y_rot = x_sec * sin_tw + y_sec * cos_tw

            # 组装到三维坐标
            # X = 轴向 (来自翼型法向, 即旋转后的y)
            # Y = 展向
            # Z = 旋转平面内弦向 (旋转后的x)
            X[i, :] = y_rot + dh  # 轴向 + 下反
            Y[i, :] = r           # 展向
            Z[i, :] = x_rot + sw  # 弦向 + 后掠

        return X, Y, Z

    def build_all_blades(self, n_chordwise=61):
        """
        生成所有叶片的三维坐标 (绕旋转轴均匀分布)

        返回:
            blades : list of (X, Y, Z) for each blade
        """
        X0, Y0, Z0 = self.build_surface(n_chordwise)
        blades = []

        for b in range(self.B):
            angle = 2 * np.pi * b / self.B
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # 绕X轴旋转
            Y_rot = Y0 * cos_a - Z0 * sin_a  # 这里Z也参与旋转
            Z_rot = Y0 * sin_a + Z0 * cos_a

            # 但更准确的是绕推力轴(X)旋转Y-Z平面
            # Y_new = Y*cos - Z*sin ... 不对, Y是展向(径向), 应该绕X旋转
            # 修正: 展向Y和弦向Z都在旋转平面内
            blades.append((X0.copy(), Y_rot, Z_rot))

        return blades

    def export_csv(self, filename):
        """导出截面坐标到CSV (可导入ANSYS SpaceClaim等)"""
        X, Y, Z = self.build_surface()

        with open(filename, 'w') as f:
            f.write("# Propeller Blade Surface Coordinates\n")
            f.write("# R={:.4f}m, Rhub={:.4f}m, B={}\n".format(
                self.R, self.Rhub, self.B))
            f.write("# Sections: {}\n".format(len(self.sections)))
            f.write("X,Y,Z,Section\n")

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    f.write(f"{X[i,j]:.6f},{Y[i,j]:.6f},{Z[i,j]:.6f},{i}\n")

    def export_section_curves(self, filename):
        """导出各截面二维坐标 (供SpaceClaim导入做放样)"""
        with open(filename, 'w') as f:
            f.write("# Section curves for loft operation\n")
            f.write("# Format: r/R, x_3d, y_3d, z_3d\n\n")

            X, Y, Z = self.build_surface()
            for i, sec in enumerate(self.sections):
                f.write(f"# Section {i}: r/R = {sec['r_R']:.3f}, "
                        f"chord = {sec['chord']*1000:.1f}mm, "
                        f"twist = {np.degrees(sec['twist']):.1f}deg\n")
                for j in range(X.shape[1]):
                    f.write(f"{sec['r_R']:.4f},{X[i,j]:.6f},{Y[i,j]:.6f},{Z[i,j]:.6f}\n")
                f.write("\n")


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    # --- 定义螺旋桨参数 ---
    R = 0.15       # 150mm半径
    Rhub = 0.02    # 20mm叶根
    B = 3

    blade = BladeGeometry(R, Rhub, B)

    # CST翼型系数 (典型低雷诺数翼型)
    A_up = np.array([0.18, 0.22, 0.18, 0.16, 0.12, 0.08])
    A_lo = np.array([-0.18, -0.06, -0.02, -0.01, 0.00, -0.01])

    # 展向控制点
    r_ctrl = np.array([0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95])
    chord_ctrl = np.array([0.012, 0.020, 0.024, 0.022, 0.018, 0.012, 0.005])
    twist_ctrl = np.array([38, 30, 22, 16, 11, 7, 4])
    sweep_ctrl = np.array([0, 0, 0, 0.001, 0.002, 0.003, 0.004])  # 轻微后掠

    # 生成桨叶
    blade.generate_from_distributions(
        r_ctrl, chord_ctrl, twist_ctrl,
        A_up, A_lo,
        n_span=25,
        sweep_ctrl=sweep_ctrl
    )

    # 构建三维表面
    X, Y, Z = blade.build_surface(n_chordwise=51)
    blades = blade.build_all_blades(n_chordwise=51)

    # 导出文件
    blade.export_csv('/home/claude/parametric_codes/python/blade_surface.csv')
    blade.export_section_curves('/home/claude/parametric_codes/python/blade_sections.csv')

    # ========== 可视化 ==========
    fig = plt.figure(figsize=(18, 14))

    # --- 子图1: 单叶片三维视图 ---
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(Z * 1000, Y * 1000, X * 1000,
                    cmap='coolwarm', alpha=0.85,
                    edgecolor='gray', linewidth=0.2)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('r (mm)')
    ax.set_zlabel('X (mm)')
    ax.set_title('Single Blade 3D View', fontsize=13, fontweight='bold')
    ax.view_init(elev=25, azim=-60)

    # --- 子图2: 全桨三维视图 ---
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    colors = ['coolwarm', 'viridis', 'plasma']
    for b_idx, (Xb, Yb, Zb) in enumerate(blades):
        ax.plot_surface(Zb * 1000, Yb * 1000, Xb * 1000,
                        cmap=colors[b_idx % 3], alpha=0.7,
                        edgecolor='gray', linewidth=0.1)

    # 画桨毂
    theta_hub = np.linspace(0, 2*np.pi, 50)
    z_hub = Rhub * 1000 * np.cos(theta_hub)
    y_hub = Rhub * 1000 * np.sin(theta_hub)
    ax.plot(z_hub, y_hub, np.zeros_like(z_hub), 'k-', lw=2)

    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('X (mm)')
    ax.set_title(f'{B}-Blade Propeller', fontsize=13, fontweight='bold')
    ax.view_init(elev=30, azim=-45)

    # --- 子图3: 展向截面堆叠 (正视图) ---
    ax = fig.add_subplot(2, 2, 3)
    n_show = min(8, len(blade.sections))
    indices = np.linspace(0, len(blade.sections) - 1, n_show, dtype=int)

    for idx in indices:
        sec = blade.sections[idx]
        psi, yu, yl = cst_airfoil(sec['A_upper'], sec['A_lower'], n_pts=101)
        # 缩放并偏移
        scale = sec['chord'] * 1000
        offset_y = sec['r_R'] * R * 1000 * 0.1  # 垂直偏移用于显示

        ax.fill_between(psi * scale, yu * scale + offset_y,
                        yl * scale + offset_y,
                        alpha=0.2, color='steelblue')
        ax.plot(psi * scale, yu * scale + offset_y, 'b-', lw=1.2)
        ax.plot(psi * scale, yl * scale + offset_y, 'b-', lw=1.2)
        ax.text(scale + 1, offset_y,
                f'r/R={sec["r_R"]:.2f}\nc={scale:.1f}mm\nθ={np.degrees(sec["twist"]):.0f}°',
                fontsize=7, va='center')

    ax.set_aspect('equal')
    ax.set_title('Stacked Airfoil Sections', fontsize=13, fontweight='bold')
    ax.set_xlabel('Chord (mm)')
    ax.set_ylabel('Offset')
    ax.grid(True, alpha=0.3)

    # --- 子图4: 展向分布 ---
    ax = fig.add_subplot(2, 2, 4)
    r_plot = [s['r_R'] for s in blade.sections]
    c_plot = [s['chord'] * 1000 for s in blade.sections]
    tw_plot = [np.degrees(s['twist']) for s in blade.sections]

    ax.plot(r_plot, c_plot, 'b-o', lw=2.5, ms=4, label='Chord (mm)')
    ax.set_ylabel('Chord (mm)', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(r_plot, tw_plot, 'r-s', lw=2.5, ms=4, label='Twist (deg)')
    ax2.set_ylabel('Twist (deg)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax.plot(r_ctrl, chord_ctrl * 1000, 'b^', ms=10, mfc='none', mew=2,
            label='Control pts (chord)')
    ax2.plot(r_ctrl, twist_ctrl, 'rv', ms=10, mfc='none', mew=2,
             label='Control pts (twist)')

    ax.set_title('Spanwise Distributions', fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/blade_3d_demo.png', dpi=150)
    plt.close()

    print("3D blade geometry generated.")
    print(f"  Sections: {len(blade.sections)}")
    print(f"  Surface CSV: blade_surface.csv")
    print(f"  Section curves: blade_sections.csv")
    print(f"  Visualization: blade_3d_demo.png")
