"""
完整集成流水线 — 从翼型选型到三维桨叶输出
===========================================
将所有参数化方法串联成一条完整的设计流水线:

  Step 1: NACA翼型生成 → CST拟合 (p11 + p01)
  Step 2: 展向分布定义 (p06)
  Step 3: Adkins-Liebeck初始设计 (p06)
  Step 4: BEM性能分析 (p07)
  Step 5: Hicks-Henne翼型微调 (p03)
  Step 6: FFD三维形状微调 (p05)
  Step 7: 三维桨叶几何生成 (p09)
  Step 8: 输出坐标文件 (供ANSYS导入)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import comb
from scipy.interpolate import CubicSpline

# ============================================================
# 内联核心函数 (从各模块提取)
# ============================================================

def bernstein_b(n, k, psi):
    return comb(n, k) * psi**k * (1 - psi)**(n - k)

def cst_profile(A_upper, A_lower, n_pts=101):
    psi = np.linspace(0, 1, n_pts)
    C = psi**0.5 * (1 - psi)**1.0
    n_up, n_lo = len(A_upper)-1, len(A_lower)-1
    S_up = sum(A_upper[k]*bernstein_b(n_up, k, psi) for k in range(n_up+1))
    S_lo = sum(A_lower[k]*bernstein_b(n_lo, k, psi) for k in range(n_lo+1))
    return psi, C*S_up + psi*0.001, C*S_lo - psi*0.001

def naca4_thickness(x, t_ratio):
    return 5*t_ratio*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                       + 0.2843*x**3 - 0.1015*x**4)

def fit_cst(x_data, y_data, order=6):
    mask = (x_data > 1e-4) & (x_data < 0.999)
    x, y = x_data[mask], y_data[mask]
    C = x**0.5 * (1-x)**1.0
    B = np.zeros((len(x), order+1))
    for k in range(order+1):
        B[:, k] = C * bernstein_b(order, k, x)
    A, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    return A

def hh_bump(psi, ti, ei=3.0):
    psi_s = np.clip(psi, 1e-10, 1-1e-10)
    return np.sin(np.pi * psi_s**(np.log(0.5)/np.log(ti)))**ei

def prandtl_F(r_R, B, phi_tip):
    f = B * (1 - r_R) / (2*r_R*abs(np.sin(phi_tip)) + 1e-6)
    return (2/np.pi) * np.arccos(np.clip(np.exp(-f), -1, 1))

# ============================================================
# 流水线
# ============================================================

class PropellerDesignPipeline:
    """完整螺旋桨设计流水线"""

    def __init__(self, R=0.15, Rhub=0.02, B=3, V_design=15.0, RPM_design=8000):
        self.R = R
        self.Rhub = Rhub
        self.B = B
        self.V = V_design
        self.RPM = RPM_design
        self.omega = RPM_design * 2 * np.pi / 60

        # 存储中间结果
        self.log = []
        self.results = {}

    def _log(self, msg):
        self.log.append(msg)
        print(f"  [{len(self.log):02d}] {msg}")

    # ---- Step 1: 翼型选型 + CST拟合 ----
    def step1_airfoil(self, naca_code='4412', cst_order=6):
        """选择NACA翼型并拟合CST系数"""
        self._log(f"Step 1: Select NACA {naca_code} → CST fit (order={cst_order})")

        # 生成NACA翼型
        x = np.linspace(0.001, 0.999, 200)
        m = int(naca_code[0]) / 100
        p = int(naca_code[1]) / 10
        t = int(naca_code[2:4]) / 100

        yt = naca4_thickness(x, t)
        yc = np.where(x <= p,
                       m/p**2 * (2*p*x - x**2) if p > 0 else 0,
                       m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2) if p > 0 else 0)

        y_upper = yc + yt
        y_lower = yc - yt

        # CST拟合
        self.A_upper = fit_cst(x, y_upper, cst_order)
        self.A_lower = fit_cst(x, y_lower, cst_order)

        # 验证
        psi_v, yu_v, yl_v = cst_profile(self.A_upper, self.A_lower)
        err = np.sqrt(np.mean((np.interp(x, psi_v, yu_v) - y_upper)**2))
        self._log(f"  CST fit RMS error = {err:.2e}")

        self.results['step1'] = {
            'x': x, 'y_upper': y_upper, 'y_lower': y_lower,
            'psi': psi_v, 'yu_cst': yu_v, 'yl_cst': yl_v,
            'naca_code': naca_code,
        }

    # ---- Step 2: 展向初始分布 (Adkins-Liebeck启发) ----
    def step2_spanwise(self):
        """用理想扭转 + 经验弦长定义展向分布"""
        self._log("Step 2: Define spanwise distributions")

        r_R = np.linspace(0.20, 0.97, 30)
        r = r_R * self.R

        # 理想扭转: β = arctan(V / ωr) + α_design
        phi_ideal = np.arctan2(self.V, self.omega * r)
        alpha_design = np.radians(5)
        twist = np.degrees(phi_ideal + alpha_design)

        # 经验弦长 (近似Betz最优)
        phi_tip = np.arctan2(self.V, self.omega * self.R)
        F = prandtl_F(r_R, self.B, phi_tip)
        Cl_design = 0.6
        sigma_opt = 8 * np.pi * r_R * F * np.sin(phi_ideal) / (self.B * Cl_design) * \
                     np.abs(np.cos(phi_ideal) - r_R*self.omega*self.R/self.V*np.sin(phi_ideal)) / \
                     (np.sin(phi_ideal) + r_R*self.omega*self.R/self.V*np.cos(phi_ideal) + 1e-6)
        chord = np.clip(np.abs(sigma_opt) * 2*np.pi*r / self.B, 0.005, 0.035)

        self.r_R = r_R
        self.chord = chord
        self.twist = twist

        self._log(f"  Chord range: [{chord.min()*1000:.1f}, {chord.max()*1000:.1f}] mm")
        self._log(f"  Twist range: [{twist.min():.1f}, {twist.max():.1f}] deg")

        self.results['step2'] = {'r_R': r_R, 'chord': chord, 'twist': twist}

    # ---- Step 3: BEM性能分析 ----
    def step3_bem_analysis(self):
        """简化BEM分析"""
        self._log("Step 3: BEM performance analysis")

        rho = 1.225
        T, Q = 0, 0
        dT_arr, dQ_arr = [], []

        for i in range(len(self.r_R)):
            r = self.r_R[i] * self.R
            Vy = self.omega * r
            W = np.sqrt(self.V**2 + Vy**2)
            phi = np.arctan2(self.V, Vy)
            alpha = np.radians(self.twist[i]) - phi
            alpha = np.clip(alpha, np.radians(-3), np.radians(10))

            Cl = 2*np.pi * alpha * 0.9  # 有限展弦比修正
            Cd = 0.008 + 0.004*Cl**2

            Cn = Cl*np.cos(phi) + Cd*np.sin(phi)
            Ct = Cl*np.sin(phi) - Cd*np.cos(phi)

            F = prandtl_F(np.array([self.r_R[i]]), self.B,
                           np.arctan2(self.V, self.omega*self.R))[0]

            dT = 0.5*rho*W**2*self.chord[i]*Cn*self.B*F
            dQ = 0.5*rho*W**2*self.chord[i]*Ct*self.B*r*F
            dT_arr.append(dT)
            dQ_arr.append(dQ)

        r_m = self.r_R * self.R
        T = np.trapezoid(dT_arr, r_m)
        Q = np.trapezoid(dQ_arr, r_m)
        P = Q * self.omega
        n_rev = self.RPM / 60
        D = 2*self.R
        CT = T / (rho*n_rev**2*D**4)
        CP = P / (rho*n_rev**3*D**5)
        J = self.V / (n_rev*D)
        eta = max(J*CT/(CP+1e-10), 0) if CP > 0 else 0

        self._log(f"  T = {T:.3f} N, Q = {Q:.5f} N·m")
        self._log(f"  CT = {CT:.4f}, CP = {CP:.4f}, J = {J:.3f}")
        self._log(f"  η = {eta:.4f}")

        self.results['step3'] = {
            'T': T, 'Q': Q, 'eta': eta, 'CT': CT, 'CP': CP, 'J': J,
            'dT': np.array(dT_arr), 'dQ': np.array(dQ_arr),
        }

    # ---- Step 4: Hicks-Henne翼型微调 ----
    def step4_airfoil_refinement(self, bumps_upper=None):
        """用Hicks-Henne凸包微调翼型"""
        self._log("Step 4: Hicks-Henne airfoil refinement")

        if bumps_upper is None:
            # 默认: 前缘轻微增厚, 后部轻微减薄
            bumps_upper = [(0.15, 0.003), (0.30, 0.002), (0.70, -0.002)]

        psi, yu_base, yl_base = cst_profile(self.A_upper, self.A_lower)

        yu_refined = yu_base.copy()
        for ti, bi in bumps_upper:
            yu_refined += bi * hh_bump(psi, ti)

        # 重新拟合CST
        self.A_upper_refined = fit_cst(psi, yu_refined)
        self.A_lower_refined = self.A_lower.copy()

        self._log(f"  Applied {len(bumps_upper)} bumps")

        self.results['step4'] = {
            'psi': psi, 'yu_base': yu_base, 'yu_refined': yu_refined,
            'yl': yl_base, 'bumps': bumps_upper,
        }

    # ---- Step 5: 三维桨叶生成 ----
    def step5_build_blade(self, n_span=25, n_chord=51):
        """生成三维桨叶表面"""
        self._log("Step 5: Build 3D blade geometry")

        A_up = self.A_upper_refined if hasattr(self, 'A_upper_refined') else self.A_upper
        A_lo = self.A_lower_refined if hasattr(self, 'A_lower_refined') else self.A_lower

        # 展向插值
        r_ctrl = self.r_R[::len(self.r_R)//6]
        c_ctrl = self.chord[::len(self.chord)//6]
        tw_ctrl = self.twist[::len(self.twist)//6]

        cs_c = CubicSpline(r_ctrl, c_ctrl, bc_type='natural')
        cs_tw = CubicSpline(r_ctrl, tw_ctrl, bc_type='natural')

        r_span = np.linspace(self.r_R[0], self.r_R[-1], n_span)
        chord_span = np.clip(cs_c(r_span), 0.003, 0.04)
        twist_span = cs_tw(r_span)

        # 生成翼型截面
        psi, yu, yl = cst_profile(A_up, A_lo, n_pts=n_chord)
        x_sec = np.concatenate([psi, psi[-2:0:-1]])
        y_sec = np.concatenate([yu, yl[-2:0:-1]])
        n_total = len(x_sec)

        X = np.zeros((n_span, n_total))
        Y = np.zeros((n_span, n_total))
        Z = np.zeros((n_span, n_total))

        for i in range(n_span):
            r = r_span[i] * self.R
            c = chord_span[i]
            tw = np.radians(twist_span[i])

            xs = x_sec * c - 0.25 * c
            ys = y_sec * c

            x_rot = xs*np.cos(tw) - ys*np.sin(tw)
            y_rot = xs*np.sin(tw) + ys*np.cos(tw)

            X[i, :] = y_rot
            Y[i, :] = r
            Z[i, :] = x_rot

        self._log(f"  Surface grid: {n_span} × {n_total} = {n_span*n_total} points")

        self.results['step5'] = {
            'X': X, 'Y': Y, 'Z': Z,
            'r_span': r_span, 'chord_span': chord_span, 'twist_span': twist_span,
        }

    # ---- Step 6: 导出 ----
    def step6_export(self, prefix='/home/claude/parametric_codes/python/pipeline'):
        """导出坐标文件"""
        self._log("Step 6: Export files")

        r5 = self.results['step5']
        X, Y, Z = r5['X'], r5['Y'], r5['Z']

        # CSV表面坐标
        csv_path = f"{prefix}_surface.csv"
        with open(csv_path, 'w') as f:
            f.write("X_mm,Y_mm,Z_mm,Section\n")
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    f.write(f"{X[i,j]*1000:.4f},{Y[i,j]*1000:.4f},{Z[i,j]*1000:.4f},{i}\n")

        # 截面曲线 (SpaceClaim放样用)
        sec_path = f"{prefix}_sections.csv"
        with open(sec_path, 'w') as f:
            f.write("Section,r_mm,X_mm,Y_mm,Z_mm\n")
            for i in range(X.shape[0]):
                r_mm = r5['r_span'][i] * self.R * 1000
                for j in range(X.shape[1]):
                    f.write(f"{i},{r_mm:.2f},{X[i,j]*1000:.4f},{Y[i,j]*1000:.4f},{Z[i,j]*1000:.4f}\n")

        # 设计摘要
        summary_path = f"{prefix}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PROPELLER DESIGN SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Radius:    {self.R*1000:.1f} mm\n")
            f.write(f"Hub:       {self.Rhub*1000:.1f} mm\n")
            f.write(f"Blades:    {self.B}\n")
            f.write(f"V_design:  {self.V:.1f} m/s\n")
            f.write(f"RPM:       {self.RPM}\n\n")
            if 'step1' in self.results:
                f.write(f"Airfoil:   NACA {self.results['step1']['naca_code']}\n")
            if 'step3' in self.results:
                r3 = self.results['step3']
                f.write(f"\nPerformance:\n")
                f.write(f"  Thrust:  {r3['T']:.3f} N\n")
                f.write(f"  Torque:  {r3['Q']:.5f} N·m\n")
                f.write(f"  η:       {r3['eta']:.4f}\n")
                f.write(f"  CT:      {r3['CT']:.4f}\n")
                f.write(f"  J:       {r3['J']:.3f}\n")
            f.write(f"\nDesign Log:\n")
            for entry in self.log:
                f.write(f"  {entry}\n")

        self._log(f"  Exported: {csv_path}")
        self._log(f"  Exported: {sec_path}")
        self._log(f"  Exported: {summary_path}")

    # ---- 综合可视化 ----
    def visualize(self, save_path=None):
        """生成设计总结图"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

        # (0,0): 翼型 + CST拟合 + HH微调
        ax = fig.add_subplot(gs[0, 0])
        if 'step1' in self.results:
            r1 = self.results['step1']
            ax.plot(r1['x'], r1['y_upper'], 'k-', lw=1.5, alpha=0.3, label='NACA original')
            ax.plot(r1['x'], r1['y_lower'], 'k-', lw=1.5, alpha=0.3)
            ax.plot(r1['psi'], r1['yu_cst'], 'b--', lw=2, label='CST fit')
            ax.plot(r1['psi'], r1['yl_cst'], 'b--', lw=2)
        if 'step4' in self.results:
            r4 = self.results['step4']
            ax.plot(r4['psi'], r4['yu_refined'], 'r-', lw=2, label='HH refined')
        ax.set_aspect('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title('Step 1+4: Airfoil', fontsize=11, fontweight='bold')

        # (0,1): 弦长分布
        ax = fig.add_subplot(gs[0, 1])
        if 'step2' in self.results:
            r2 = self.results['step2']
            ax.plot(r2['r_R'], r2['chord']*1000, 'b-', lw=2.5)
            ax.set_title('Step 2: Chord c(r/R)', fontsize=11, fontweight='bold')
            ax.set_xlabel('r/R'); ax.set_ylabel('mm'); ax.grid(True, alpha=0.3)

        # (0,2): 扭转分布
        ax = fig.add_subplot(gs[0, 2])
        if 'step2' in self.results:
            ax.plot(r2['r_R'], r2['twist'], 'r-', lw=2.5)
            ax.set_title('Step 2: Twist θ(r/R)', fontsize=11, fontweight='bold')
            ax.set_xlabel('r/R'); ax.set_ylabel('deg'); ax.grid(True, alpha=0.3)

        # (0,3): 载荷分布
        ax = fig.add_subplot(gs[0, 3])
        if 'step3' in self.results:
            r3 = self.results['step3']
            ax.plot(r2['r_R'], r3['dT'], 'b-', lw=2.5, label='dT/dr')
            ax.set_ylabel('dT/dr (N/m)', color='b')
            ax2 = ax.twinx()
            ax2.plot(r2['r_R'], r3['dQ'], 'r-', lw=2.5, label='dQ/dr')
            ax2.set_ylabel('dQ/dr', color='r')
            ax.set_title('Step 3: BEM Loading', fontsize=11, fontweight='bold')
            ax.set_xlabel('r/R'); ax.grid(True, alpha=0.3)

        # (1, 0:2): 三维桨叶
        ax = fig.add_subplot(gs[1, 0:2], projection='3d')
        if 'step5' in self.results:
            r5 = self.results['step5']
            ax.plot_surface(r5['Z']*1000, r5['Y']*1000, r5['X']*1000,
                            cmap='coolwarm', alpha=0.85, edgecolor='gray', linewidth=0.15)
            ax.set_xlabel('Z (mm)'); ax.set_ylabel('r (mm)'); ax.set_zlabel('X (mm)')
            ax.view_init(elev=25, azim=-55)
        ax.set_title('Step 5: 3D Blade', fontsize=11, fontweight='bold')

        # (1, 2:4): 全桨三维
        ax = fig.add_subplot(gs[1, 2:4], projection='3d')
        if 'step5' in self.results:
            r5 = self.results['step5']
            for b in range(self.B):
                angle = 2*np.pi*b/self.B
                Y_rot = r5['Y']*np.cos(angle) - r5['Z']*np.sin(angle)
                Z_rot = r5['Y']*np.sin(angle) + r5['Z']*np.cos(angle)
                ax.plot_surface(Z_rot*1000, Y_rot*1000, r5['X']*1000,
                                alpha=0.7, edgecolor='none')
            ax.view_init(elev=30, azim=-45)
        ax.set_title(f'Step 5: {self.B}-Blade Propeller', fontsize=11, fontweight='bold')

        # (2, 0:2): 截面堆叠图
        ax = fig.add_subplot(gs[2, 0:2])
        if 'step5' in self.results:
            r5 = self.results['step5']
            psi, yu, yl = cst_profile(
                self.A_upper_refined if hasattr(self, 'A_upper_refined') else self.A_upper,
                self.A_lower_refined if hasattr(self, 'A_lower_refined') else self.A_lower)
            n_show = min(10, len(r5['r_span']))
            indices = np.linspace(0, len(r5['r_span'])-1, n_show, dtype=int)
            for idx in indices:
                c = r5['chord_span'][idx]*1000
                offset = r5['r_span'][idx]*self.R*1000*0.12
                ax.fill_between(psi*c, yu*c+offset, yl*c+offset,
                                alpha=0.15, color='steelblue')
                ax.plot(psi*c, yu*c+offset, 'b-', lw=1)
                ax.plot(psi*c, yl*c+offset, 'b-', lw=1)
                ax.text(c+0.3, offset,
                        f'r/R={r5["r_span"][idx]:.2f}\n{c:.1f}mm, {r5["twist_span"][idx]:.0f}°',
                        fontsize=6, va='center')
            ax.set_aspect('equal')
            ax.set_title('Stacked Sections', fontsize=11, fontweight='bold')
            ax.set_xlabel('mm'); ax.grid(True, alpha=0.3)

        # (2, 2:4): 性能摘要文字
        ax = fig.add_subplot(gs[2, 2:4])
        ax.axis('off')
        summary_lines = [
            f"DESIGN SUMMARY",
            f"{'─'*40}",
            f"R = {self.R*1000:.0f} mm   Hub = {self.Rhub*1000:.0f} mm   B = {self.B}",
            f"V = {self.V:.0f} m/s   RPM = {self.RPM}",
        ]
        if 'step1' in self.results:
            summary_lines.append(f"Airfoil: NACA {self.results['step1']['naca_code']}")
        if 'step3' in self.results:
            r3 = self.results['step3']
            summary_lines += [
                f"{'─'*40}",
                f"Thrust   T  = {r3['T']:.3f} N",
                f"Torque   Q  = {r3['Q']:.5f} N·m",
                f"Power    P  = {r3['Q']*self.omega:.2f} W",
                f"CT = {r3['CT']:.4f}   CP = {r3['CP']:.4f}",
                f"J  = {r3['J']:.3f}    η  = {r3['eta']:.4f}",
            ]
        if 'step5' in self.results:
            r5 = self.results['step5']
            summary_lines += [
                f"{'─'*40}",
                f"Sections: {r5['X'].shape[0]}",
                f"Surface pts: {r5['X'].shape[0]*r5['X'].shape[1]}",
            ]
        summary_lines += [
            f"{'─'*40}",
            "DESIGN PIPELINE:",
        ] + [f"  {entry}" for entry in self.log[:10]]

        text = '\n'.join(summary_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=9, fontfamily='monospace', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ============================================================
# 执行完整流水线
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROPELLER DESIGN PIPELINE")
    print("=" * 60)

    pipe = PropellerDesignPipeline(
        R=0.15, Rhub=0.02, B=3,
        V_design=15.0, RPM_design=8000
    )

    pipe.step1_airfoil(naca_code='4412', cst_order=6)
    pipe.step2_spanwise()
    pipe.step3_bem_analysis()
    pipe.step4_airfoil_refinement(bumps_upper=[
        (0.12, 0.003),   # 前缘增厚 → 更大Re容忍度
        (0.35, 0.002),   # 压力面鼓包 → 增升
        (0.75, -0.002),  # 后缘减薄 → 减阻
    ])
    pipe.step5_build_blade(n_span=25, n_chord=51)
    pipe.step6_export()

    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    pipe.visualize(save_path='/home/claude/parametric_codes/python/pipeline_demo.png')
    print("Pipeline demo saved.")
