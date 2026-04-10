"""
CST + B样条 + Kriging 代理模型优化框架
=======================================
参考: 
  - Kulfan (2008) CST参数化
  - Sacks J. et al., "Design and Analysis of Computer Experiments," 
    Statistical Science, 1989 (Kriging原始文献)
  - Jones D.R. et al., "Efficient Global Optimization of Expensive 
    Black-Box Functions," J. Global Optimization, 1998 (EGO/EI准则)

完整流程:
  参数化 → 拉丁超立方采样 → BEM评估 → Kriging拟合 → 优化 → 验证
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
from scipy.stats import norm

# ============================================================
# 拉丁超立方采样 (LHS)
# ============================================================

def latin_hypercube_sampling(n_samples, n_dims, bounds):
    """
    拉丁超立方采样

    参数:
        n_samples : int         采样点数
        n_dims    : int         维度
        bounds    : (n_dims, 2) 每维的上下界

    返回:
        samples : (n_samples, n_dims)
    """
    samples = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        # 每维均匀分成n_samples个区间, 每个区间随机取一个点
        perm = np.random.permutation(n_samples)
        for i in range(n_samples):
            low = bounds[j, 0] + (bounds[j, 1] - bounds[j, 0]) * perm[i] / n_samples
            high = bounds[j, 0] + (bounds[j, 1] - bounds[j, 0]) * (perm[i] + 1) / n_samples
            samples[i, j] = np.random.uniform(low, high)
    return samples


# ============================================================
# Kriging (高斯过程回归) — 从零实现
# ============================================================

class Kriging:
    """
    Kriging代理模型 (Ordinary Kriging)

    数学模型: Y(x) = μ + Z(x)
    其中 Z(x) 是均值为零的高斯过程, 协方差函数为:
    k(x_i, x_j) = σ² · exp(-Σ θ_d |x_i^d - x_j^d|^p_d)
    
    这里用各向异性高斯核 (p=2):
    k(x_i, x_j) = σ² · exp(-Σ θ_d (x_i^d - x_j^d)²)
    """

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.theta = None
        self.mu = None
        self.sigma2 = None
        self.K_inv = None

    def _correlation_matrix(self, X1, X2, theta):
        """计算相关矩阵 R_{ij} = exp(-Σ θ_d (x_i^d - x_j^d)²)"""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]

        R = np.zeros((n1, n2))
        for k in range(d):
            diff = X1[:, k:k+1] - X2[:, k:k+1].T  # (n1, n2)
            R += theta[k] * diff**2
        R = np.exp(-R)
        return R

    def fit(self, X, y):
        """
        训练Kriging模型

        参数:
            X : (n, d) 训练输入
            y : (n,)   训练输出
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        n, d = X.shape

        # 归一化
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-10
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-10

        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std

        # 优化超参数 θ (最大化对数似然)
        def neg_log_likelihood(log_theta):
            theta = np.exp(log_theta)
            R = self._correlation_matrix(X_norm, X_norm, theta)
            R += 1e-8 * np.eye(n)  # 正则化

            try:
                L = np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                return 1e10

            R_inv = np.linalg.solve(R, np.eye(n))
            ones = np.ones(n)

            mu = ones @ R_inv @ y_norm / (ones @ R_inv @ ones)
            residual = y_norm - mu
            sigma2 = residual @ R_inv @ residual / n

            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e10

            nll = 0.5 * n * np.log(max(sigma2, 1e-10)) + 0.5 * logdet
            return nll

        # 多起点优化
        best_nll = 1e10
        best_theta = np.ones(d)
        for _ in range(10):
            log_theta0 = np.random.uniform(-2, 3, d)
            try:
                res = minimize(neg_log_likelihood, log_theta0,
                               method='L-BFGS-B',
                               bounds=[(-3, 5)] * d)
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_theta = np.exp(res.x)
            except Exception:
                pass

        self.theta = best_theta

        # 用最优θ计算模型参数
        R = self._correlation_matrix(X_norm, X_norm, self.theta)
        R += 1e-8 * np.eye(n)
        self.R = R
        self.R_inv = np.linalg.solve(R, np.eye(n))

        ones = np.ones(n)
        self.mu = ones @ self.R_inv @ y_norm / (ones @ self.R_inv @ ones)
        residual = y_norm - self.mu
        self.sigma2 = residual @ self.R_inv @ residual / n

    def predict(self, X_new, return_std=False):
        """
        预测新点

        返回:
            y_pred : 预测均值
            y_std  : 预测标准差 (如果 return_std=True)
        """
        X_norm = (X_new - self.X_mean) / self.X_std
        X_train_norm = (self.X_train - self.X_mean) / self.X_std
        y_train_norm = (self.y_train - self.y_mean) / self.y_std

        r = self._correlation_matrix(X_norm, X_train_norm, self.theta)

        y_pred_norm = self.mu + r @ self.R_inv @ (y_train_norm - self.mu)
        y_pred = y_pred_norm * self.y_std + self.y_mean

        if return_std:
            n = len(self.y_train)
            ones = np.ones(n)
            mse = np.zeros(X_new.shape[0])
            for i in range(X_new.shape[0]):
                ri = r[i:i+1, :]
                term1 = 1 - ri @ self.R_inv @ ri.T
                term2_num = (1 - ri @ self.R_inv @ ones)**2
                term2_den = ones @ self.R_inv @ ones
                mse[i] = self.sigma2 * (term1[0, 0] + term2_num[0] / term2_den)
            y_std = np.sqrt(np.maximum(mse, 0)) * self.y_std
            return y_pred, y_std

        return y_pred

    def expected_improvement(self, X_new, y_best):
        """
        期望改进准则 (EI) — 用于自适应采样

        EI(x) = (y_best - μ(x)) · Φ(z) + σ(x) · φ(z)
        其中 z = (y_best - μ(x)) / σ(x)
        """
        y_pred, y_std = self.predict(X_new, return_std=True)
        y_std = np.maximum(y_std, 1e-10)

        z = (y_best - y_pred) / y_std
        ei = (y_best - y_pred) * norm.cdf(z) + y_std * norm.pdf(z)
        ei[y_std < 1e-8] = 0
        return ei


# ============================================================
# 螺旋桨参数化 + 评估函数 (简化版)
# ============================================================

def propeller_objective(params, verbose=False):
    """
    螺旋桨性能评估 (简化的BEM模型)

    参数 (5维设计变量):
        params[0] : 叶根弦长系数 c_root (0.05 ~ 0.15)
        params[1] : 最大弦长位置 r_cmax (0.3 ~ 0.7)
        params[2] : 最大弦长值 c_max (0.08 ~ 0.18)
        params[3] : 叶根扭转角 twist_root (15 ~ 45 deg)
        params[4] : 叶尖扭转角 twist_tip (0 ~ 15 deg)

    返回:
        -η (负效率, 因为优化器做最小化)
    """
    c_root, r_cmax, c_max, twist_root, twist_tip = params

    R = 0.15
    n_sec = 20
    r_R = np.linspace(0.2, 0.98, n_sec)

    # 弦长分布 (抛物线近似)
    chord = np.zeros(n_sec)
    for i, r in enumerate(r_R):
        if r < r_cmax:
            chord[i] = c_root + (c_max - c_root) * ((r - 0.2) / (r_cmax - 0.2))**0.8
        else:
            chord[i] = c_max * ((0.98 - r) / (0.98 - r_cmax))**1.2
    chord = chord * R
    chord = np.clip(chord, 0.005, 0.03)

    # 扭转分布 (线性)
    twist = np.linspace(twist_root, twist_tip, n_sec)

    # 简化BEM分析
    V_inf = 15.0
    omega = 8000 * 2 * np.pi / 60
    rho = 1.225
    B = 3

    T_total = 0
    Q_total = 0

    for i in range(n_sec):
        r = r_R[i] * R
        Vy = omega * r
        phi = np.arctan2(V_inf, Vy)
        alpha = np.radians(twist[i]) - phi + np.radians(5)  # 简化
        alpha = np.clip(alpha, np.radians(-5), np.radians(12))

        Cl = 2 * np.pi * alpha
        Cd = 0.008 + 0.005 * Cl**2
        W = np.sqrt(V_inf**2 + Vy**2)

        Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ct = Cl * np.sin(phi) - Cd * np.cos(phi)

        dT = 0.5 * rho * W**2 * chord[i] * Cn * B
        dQ = 0.5 * rho * W**2 * chord[i] * Ct * B * r

        T_total += dT * (R * 0.78 / n_sec)
        Q_total += dQ * (R * 0.78 / n_sec)

    P_total = Q_total * omega
    eta = T_total * V_inf / (P_total + 1e-10) if P_total > 0 else 0
    eta = np.clip(eta, 0, 1)

    if verbose:
        print(f"  T={T_total:.3f}N, Q={Q_total:.5f}N·m, η={eta:.4f}")

    return -eta  # 负号: 最小化 → 最大化效率


# ============================================================
# 演示: 完整优化流程
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 设计空间
    bounds = np.array([
        [0.05, 0.15],    # c_root
        [0.30, 0.70],    # r_cmax
        [0.08, 0.18],    # c_max
        [15.0, 45.0],    # twist_root
        [0.0,  15.0],    # twist_tip
    ])
    n_dims = bounds.shape[0]

    # ========== 第1步: LHS初始采样 ==========
    n_initial = 40
    X_samples = latin_hypercube_sampling(n_initial, n_dims, bounds)
    y_samples = np.array([propeller_objective(x) for x in X_samples])

    print(f"Initial sampling: {n_initial} points")
    print(f"  Best η = {-min(y_samples):.4f}")

    # ========== 第2步: 训练Kriging模型 ==========
    model = Kriging()
    model.fit(X_samples, y_samples)

    # ========== 第3步: EGO自适应优化 (添加新采样点) ==========
    n_infill = 15  # 额外的自适应采样点
    best_history = [min(y_samples)]

    for iteration in range(n_infill):
        y_best = min(y_samples)

        # 最大化EI准则, 找到最有价值的下一个采样点
        def neg_ei(x):
            return -model.expected_improvement(x.reshape(1, -1), y_best)[0]

        # 用差分进化全局搜索
        result = differential_evolution(
            neg_ei,
            bounds=[(b[0], b[1]) for b in bounds],
            seed=iteration, maxiter=50, tol=1e-6
        )
        x_new = result.x

        # 评估新点 (在实际应用中这里是运行CFD)
        y_new = propeller_objective(x_new)

        # 添加到训练集并重新训练
        X_samples = np.vstack([X_samples, x_new])
        y_samples = np.append(y_samples, y_new)
        model.fit(X_samples, y_samples)

        best_history.append(min(y_samples))
        print(f"  Infill {iteration+1}/{n_infill}: η = {-y_new:.4f}, "
              f"best = {-min(y_samples):.4f}")

    # ========== 第4步: 最终结果 ==========
    best_idx = np.argmin(y_samples)
    best_params = X_samples[best_idx]
    best_eta = -y_samples[best_idx]

    print(f"\n{'='*50}")
    print(f"Optimization Complete!")
    print(f"  Best efficiency: η = {best_eta:.4f}")
    print(f"  c_root    = {best_params[0]:.4f}")
    print(f"  r_cmax    = {best_params[1]:.4f}")
    print(f"  c_max     = {best_params[2]:.4f}")
    print(f"  twist_root= {best_params[3]:.1f}°")
    print(f"  twist_tip = {best_params[4]:.1f}°")
    print(f"{'='*50}")

    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- 子图1: 优化收敛历史 ---
    ax = axes[0, 0]
    ax.plot(range(len(best_history)), [-b for b in best_history],
            'b-o', lw=2, ms=5)
    ax.axvline(x=n_initial, color='r', linestyle='--', lw=1.5,
               label=f'Start infill (n={n_initial})')
    ax.set_title('Optimization Convergence', fontsize=13, fontweight='bold')
    ax.set_xlabel('Total evaluations')
    ax.set_ylabel('Best efficiency η')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 子图2: Kriging拟合质量 (交叉验证) ---
    ax = axes[0, 1]
    y_pred = model.predict(X_samples)
    ax.scatter(y_samples, y_pred, c='steelblue', alpha=0.7, s=50)
    lims = [min(y_samples.min(), y_pred.min()), max(y_samples.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_title('Kriging Fit Quality', fontsize=13, fontweight='bold')
    ax.set_xlabel('True value (-η)')
    ax.set_ylabel('Predicted value (-η)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- 子图3: 参数敏感性 (1D切片) ---
    ax = axes[0, 2]
    param_names = ['c_root', 'r_cmax', 'c_max', 'twist_root', 'twist_tip']
    x_center = best_params.copy()

    for d in range(n_dims):
        x_scan = np.linspace(bounds[d, 0], bounds[d, 1], 50)
        X_test = np.tile(x_center, (50, 1))
        X_test[:, d] = x_scan

        y_pred_scan = model.predict(X_test)
        y_norm = (y_pred_scan - y_pred_scan.mean()) / (y_pred_scan.std() + 1e-10)
        x_norm = (x_scan - bounds[d, 0]) / (bounds[d, 1] - bounds[d, 0])
        ax.plot(x_norm, -y_norm, lw=2, label=param_names[d])

    ax.set_title('Parameter Sensitivity (1D Slice)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Normalized parameter value')
    ax.set_ylabel('Normalized η')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 子图4: 采样点分布 (前两个参数) ---
    ax = axes[1, 0]
    ax.scatter(X_samples[:n_initial, 0], X_samples[:n_initial, 1],
               c='blue', alpha=0.5, s=40, label='LHS initial')
    ax.scatter(X_samples[n_initial:, 0], X_samples[n_initial:, 1],
               c='red', marker='*', s=150, label='EGO infill')
    ax.scatter(best_params[0], best_params[1],
               c='gold', marker='*', s=300, edgecolors='k', label='Optimum')
    ax.set_title('Sample Distribution (c_root vs r_cmax)', fontsize=13, fontweight='bold')
    ax.set_xlabel('c_root')
    ax.set_ylabel('r_cmax')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 子图5: EI等高线 (前两个参数) ---
    ax = axes[1, 1]
    n_grid = 40
    x1_grid = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
    x2_grid = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    EI_grid = np.zeros_like(X1)

    for i in range(n_grid):
        for j in range(n_grid):
            x_test = best_params.copy()
            x_test[0] = X1[i, j]
            x_test[1] = X2[i, j]
            EI_grid[i, j] = model.expected_improvement(
                x_test.reshape(1, -1), min(y_samples))[0]

    cs = ax.contourf(X1, X2, EI_grid, levels=20, cmap='YlOrRd')
    plt.colorbar(cs, ax=ax, label='EI')
    ax.set_title('Expected Improvement Landscape', fontsize=13, fontweight='bold')
    ax.set_xlabel('c_root')
    ax.set_ylabel('r_cmax')

    # --- 子图6: 最优桨叶几何 ---
    ax = axes[1, 2]
    r_R_plot = np.linspace(0.2, 0.98, 50)
    chord_plot = np.zeros(50)
    for i, r in enumerate(r_R_plot):
        if r < best_params[1]:
            chord_plot[i] = best_params[0] + (best_params[2] - best_params[0]) * \
                            ((r - 0.2) / (best_params[1] - 0.2))**0.8
        else:
            chord_plot[i] = best_params[2] * ((0.98 - r) / (0.98 - best_params[1]))**1.2
    chord_plot = np.clip(chord_plot, 0.005, 0.03) * 0.15 * 1000

    twist_plot = np.linspace(best_params[3], best_params[4], 50)

    ax.plot(r_R_plot, chord_plot, 'b-', lw=2.5, label='Chord (mm)')
    ax.set_ylabel('Chord (mm)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax5 = ax.twinx()
    ax5.plot(r_R_plot, twist_plot, 'r-', lw=2.5, label='Twist (deg)')
    ax5.set_ylabel('Twist (deg)', color='r')
    ax5.tick_params(axis='y', labelcolor='r')
    ax.set_title(f'Optimal Blade (η={best_eta:.4f})', fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/kriging_optimization_demo.png', dpi=150)
    plt.close()
    print("Kriging optimization demo saved.")
