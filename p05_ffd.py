"""
自由形变 (FFD, Free-Form Deformation) 方法
===========================================
参考: Sederberg T.W. & Parry S.R., "Free-Form Deformation of Solid Geometric Models,"
      ACM SIGGRAPH, 1986.
      Kenway G.K.W. & Martins J.R.R.A., AIAA Journal, 2016.

核心思想: 不直接参数化几何, 而是参数化几何所在空间的变形。
         将物体嵌入控制体, 移动控制点 → 空间变形 → 物体跟随变形。
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb

# ============================================================
# 核心函数 (2D FFD)
# ============================================================

def bernstein_poly(n, i, t):
    """n阶第i个Bernstein多项式"""
    return comb(n, i) * t**i * (1 - t)**(n - i)


class FFD2D:
    """
    二维FFD控制体

    控制点网格: (l+1) x (m+1)
    局部坐标 (s, t) ∈ [0,1] x [0,1]
    """

    def __init__(self, x_range, y_range, nx, ny):
        """
        初始化FFD控制体

        参数:
            x_range : (xmin, xmax) 控制体x范围
            y_range : (ymin, ymax) 控制体y范围
            nx, ny  : 每个方向的控制点数
        """
        self.x_range = x_range
        self.y_range = y_range
        self.nx = nx
        self.ny = ny

        # 生成初始控制点网格
        xs = np.linspace(x_range[0], x_range[1], nx)
        ys = np.linspace(y_range[0], y_range[1], ny)
        self.P0 = np.zeros((nx, ny, 2))  # 初始位置
        self.P = np.zeros((nx, ny, 2))   # 当前位置(含变形)
        for i in range(nx):
            for j in range(ny):
                self.P0[i, j] = [xs[i], ys[j]]
                self.P[i, j] = [xs[i], ys[j]]

    def world_to_local(self, pts):
        """世界坐标 → 局部坐标 (s, t)"""
        s = (pts[:, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        t = (pts[:, 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
        return s, t

    def deform(self, pts):
        """
        对一组点施加FFD变形

        参数:
            pts : (N, 2) 原始坐标

        返回:
            pts_new : (N, 2) 变形后坐标
        """
        s, t = self.world_to_local(pts)
        pts_new = np.zeros_like(pts)
        l = self.nx - 1  # Bernstein阶数
        m = self.ny - 1

        for k in range(len(pts)):
            for i in range(self.nx):
                Bi = bernstein_poly(l, i, s[k])
                for j in range(self.ny):
                    Bj = bernstein_poly(m, j, t[k])
                    pts_new[k] += Bi * Bj * self.P[i, j]
        return pts_new

    def move_control_point(self, i, j, dx, dy):
        """移动控制点(i,j)"""
        self.P[i, j, 0] += dx
        self.P[i, j, 1] += dy

    def plot(self, ax, original_pts=None, deformed_pts=None):
        """可视化FFD控制体和变形结果"""
        # 画控制网格
        for i in range(self.nx):
            ax.plot(self.P[i, :, 0], self.P[i, :, 1],
                    'g-o', ms=6, alpha=0.5, lw=1)
        for j in range(self.ny):
            ax.plot(self.P[:, j, 0], self.P[:, j, 1],
                    'g-o', ms=6, alpha=0.5, lw=1)

        if original_pts is not None:
            ax.plot(original_pts[:, 0], original_pts[:, 1],
                    'b--', lw=1.5, alpha=0.5, label='Original')
        if deformed_pts is not None:
            ax.plot(deformed_pts[:, 0], deformed_pts[:, 1],
                    'r-', lw=2.5, label='Deformed')


def naca0012_coords(n_pts=201):
    """生成NACA 0012 翼型坐标 (上+下表面)"""
    x = np.linspace(0, 1, n_pts)
    t = 0.12
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1015*x**4)
    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1], -yt[::-1]])
    return np.vstack([upper, lower])


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    airfoil = naca0012_coords(101)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 示例1: 增大厚度（上下控制点外移） ---
    ax = axes[0, 0]
    ffd = FFD2D((-0.05, 1.05), (-0.12, 0.12), 5, 3)
    # 把中间行上移、下移 → 增厚
    for i in range(5):
        ffd.move_control_point(i, 2, 0, 0.03)   # 上排外移
        ffd.move_control_point(i, 0, 0, -0.03)  # 下排外移

    deformed = ffd.deform(airfoil)
    ffd.plot(ax, airfoil, deformed)
    ax.set_aspect('equal')
    ax.set_title('FFD: Thickness Increase', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例2: 弯度变化（整体上移） ---
    ax = axes[0, 1]
    ffd = FFD2D((-0.05, 1.05), (-0.12, 0.12), 5, 3)
    # 中段控制点全部上移 → 增加弯度
    for j in range(3):
        ffd.move_control_point(1, j, 0, 0.02)
        ffd.move_control_point(2, j, 0, 0.035)
        ffd.move_control_point(3, j, 0, 0.02)

    deformed = ffd.deform(airfoil)
    ffd.plot(ax, airfoil, deformed)
    ax.set_aspect('equal')
    ax.set_title('FFD: Camber Change', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例3: 局部变形（后缘修型） ---
    ax = axes[1, 0]
    ffd = FFD2D((-0.05, 1.05), (-0.12, 0.12), 7, 3)
    # 只移动后缘附近的控制点
    ffd.move_control_point(5, 2, 0, 0.02)
    ffd.move_control_point(6, 2, 0, -0.01)
    ffd.move_control_point(5, 0, 0, 0.01)
    ffd.move_control_point(6, 1, 0.02, 0)

    deformed = ffd.deform(airfoil)
    ffd.plot(ax, airfoil, deformed)
    ax.set_aspect('equal')
    ax.set_title('FFD: Local Trailing-Edge Modification (7×3)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 示例4: 扭转变形（模拟桨叶扭转） ---
    ax = axes[1, 1]
    ffd = FFD2D((-0.05, 1.05), (-0.12, 0.12), 5, 3)
    # 后缘控制点整体旋转 → 模拟翼型扭转
    twist_angle = np.radians(5)  # 5度扭转
    for j in range(3):
        # 后缘向下旋转
        dy = -0.5 * np.sin(twist_angle) * 0.5
        ffd.move_control_point(3, j, 0, -dy * (j - 1) * 0.5)
        ffd.move_control_point(4, j, 0, -dy * (j - 1))

    deformed = ffd.deform(airfoil)
    ffd.plot(ax, airfoil, deformed)
    ax.set_aspect('equal')
    ax.set_title('FFD: Twist Deformation', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/ffd_demo.png', dpi=150)
    plt.close()
    print("FFD demo saved.")
