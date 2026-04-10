"""
NSGA-II 多目标优化框架 — 螺旋桨效率 vs 噪声
===============================================
参考:
  - Deb K. et al., "A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II," IEEE Trans. Evolutionary Computation, 2002.
  - Brooks T.F. et al., "Airfoil Self-Noise and Prediction," NASA RP-1218, 1989.
    (BPM噪声模型)

核心思想:
  多目标优化不追求"唯一最优解", 而是找到一组Pareto前沿解,
  让设计师在效率和噪声之间权衡取舍。

设计变量 (6维):
  弦长分布3个控制点 + 扭转分布3个控制点

目标函数:
  f1 = -η (最大化效率 → 最小化-η)
  f2 = SPL  (最小化噪声)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ============================================================
# 简化的螺旋桨气动模型
# ============================================================

def simple_bem_analysis(chord_ctrl, twist_ctrl, V_inf=15, RPM=8000, R=0.15, B=3):
    """
    简化BEM分析 → 推力、功率、效率

    参数:
        chord_ctrl : (3,) 弦长控制点 c/R (在r/R=0.3, 0.6, 0.9处)
        twist_ctrl : (3,) 扭转控制点 (deg, 在r/R=0.3, 0.6, 0.9处)

    返回:
        T, P, eta, chord_dist, twist_dist, r_R
    """
    omega = RPM * 2 * np.pi / 60
    rho = 1.225
    n_sec = 30

    # 展向插值
    r_ctrl = np.array([0.15, 0.30, 0.60, 0.90, 0.98])
    c_ctrl = np.array([chord_ctrl[0]*0.6, chord_ctrl[0], chord_ctrl[1],
                        chord_ctrl[2], chord_ctrl[2]*0.3])
    tw_ctrl = np.array([twist_ctrl[0]*1.3, twist_ctrl[0], twist_ctrl[1],
                         twist_ctrl[2], twist_ctrl[2]*0.5])

    cs_c = CubicSpline(r_ctrl, c_ctrl, bc_type='natural')
    cs_tw = CubicSpline(r_ctrl, tw_ctrl, bc_type='natural')

    r_R = np.linspace(0.20, 0.97, n_sec)
    chord = np.clip(cs_c(r_R), 0.03, 0.20) * R  # 转为实际弦长
    twist = cs_tw(r_R)

    T_total = 0
    Q_total = 0

    for i in range(n_sec):
        r = r_R[i] * R
        Vy = omega * r
        W = np.sqrt(V_inf**2 + Vy**2)
        phi = np.arctan2(V_inf, Vy)

        alpha = np.radians(twist[i]) - phi
        alpha = np.clip(alpha, np.radians(-3), np.radians(10))

        # 翼型模型
        Cl = 5.5 * alpha  # 稍低于理论值, 模拟有限展弦比
        Cd = 0.010 + 0.04 * alpha**2

        Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ct = Cl * np.sin(phi) - Cd * np.cos(phi)

        # Prandtl修正
        f_tip = B * (R - r) / (2 * r * abs(np.sin(phi)) + 1e-6)
        F = (2/np.pi) * np.arccos(np.clip(np.exp(-f_tip), -1, 1))

        dT = 0.5 * rho * W**2 * chord[i] * Cn * B * F
        dQ = 0.5 * rho * W**2 * chord[i] * Ct * B * r * F

        dr = (0.97 - 0.20) * R / n_sec
        T_total += dT * dr
        Q_total += dQ * dr

    P_total = Q_total * omega
    eta = max(T_total * V_inf / (abs(P_total) + 1e-6), 0)
    eta = min(eta, 0.95)

    return T_total, P_total, eta, chord, twist, r_R


# ============================================================
# 简化的噪声模型 (BPM经验公式简化版)
# ============================================================

def simple_noise_model(chord_ctrl, twist_ctrl, V_inf=15, RPM=8000, R=0.15, B=3):
    """
    简化的BPM (Brooks-Pope-Marcolini) 噪声估算

    主要噪声源:
      1. 湍流边界层-尾缘噪声 (TBL-TE): ∝ V^5 · c · δ*
      2. 叶尖涡噪声: ∝ V_tip^6

    返回:
        SPL_total : 总声压级 (dB, 参考点距螺旋桨1m)
    """
    omega = RPM * 2 * np.pi / 60
    rho = 1.225
    c0 = 343.0  # 声速
    r_obs = 1.0  # 观测距离

    n_sec = 30
    r_ctrl = np.array([0.15, 0.30, 0.60, 0.90, 0.98])
    c_ctrl = np.array([chord_ctrl[0]*0.6, chord_ctrl[0], chord_ctrl[1],
                        chord_ctrl[2], chord_ctrl[2]*0.3])

    cs_c = CubicSpline(r_ctrl, c_ctrl, bc_type='natural')
    r_R = np.linspace(0.20, 0.97, n_sec)
    chord = np.clip(cs_c(r_R), 0.03, 0.20) * R

    # --- TBL-TE噪声 ---
    # 简化: SPL_TE ∝ 10·log10(M^5 · c · δ* · Dr / r²)
    # δ* ≈ c · Re^(-0.5)  (层流边界层位移厚度)

    SPL_sections = np.zeros(n_sec)
    for i in range(n_sec):
        r = r_R[i] * R
        V_local = np.sqrt(V_inf**2 + (omega * r)**2)
        M_local = V_local / c0
        Re_local = rho * V_local * chord[i] / 1.8e-5

        # 位移厚度 (简化)
        delta_star = chord[i] / np.sqrt(Re_local + 1e-6)

        # 方向性函数 (简化为1)
        D = 1.0

        # SPL贡献
        power_acoustic = rho * V_local**5 * chord[i] * delta_star * D / (c0**2 * r_obs**2)
        SPL_sections[i] = 10 * np.log10(max(power_acoustic, 1e-20) / 1e-12)

    # 总SPL (能量叠加)
    p_sq_total = np.sum(10**(SPL_sections / 10))

    # --- 叶尖涡噪声 ---
    V_tip = omega * R
    M_tip = V_tip / c0
    # 叶尖涡强度与叶尖载荷梯度有关
    SPL_tip = 50 * np.log10(M_tip) + 60  # 经验公式简化

    # 合并 + 多叶片修正
    SPL_tblte = 10 * np.log10(max(p_sq_total, 1e-20))
    p_sq_total_all = 10**(SPL_tblte/10) + 10**(SPL_tip/10)
    SPL_total = 10 * np.log10(max(p_sq_total_all, 1e-20)) + 10 * np.log10(B)

    return SPL_total


# ============================================================
# NSGA-II 实现
# ============================================================

def nsga2_optimize(obj_func, bounds, pop_size=80, n_gen=100,
                    crossover_rate=0.9, mutation_rate=0.1):
    """
    NSGA-II 多目标优化

    参数:
        obj_func : 目标函数, 输入(n_dims,) → 输出(n_obj,)
        bounds   : (n_dims, 2) 设计变量上下界
        pop_size : 种群大小
        n_gen    : 迭代代数

    返回:
        pareto_front : Pareto前沿解集 (目标值)
        pareto_set   : Pareto前沿解集 (设计变量)
        history      : 每代的超体积或前沿大小
    """
    n_dims = bounds.shape[0]

    # --- 初始化种群 ---
    pop = np.random.rand(pop_size, n_dims)
    for d in range(n_dims):
        pop[:, d] = bounds[d, 0] + pop[:, d] * (bounds[d, 1] - bounds[d, 0])

    # 评估初始种群
    obj = np.array([obj_func(ind) for ind in pop])
    n_obj = obj.shape[1]

    history = []

    for gen in range(n_gen):
        # --- 生成子代 ---
        offspring = np.zeros_like(pop)
        for i in range(0, pop_size, 2):
            # 锦标赛选择
            p1 = tournament_select(pop, obj)
            p2 = tournament_select(pop, obj)

            # SBX交叉
            if np.random.rand() < crossover_rate:
                c1, c2 = sbx_crossover(p1, p2, bounds)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # 多项式变异
            c1 = polynomial_mutation(c1, bounds, mutation_rate)
            c2 = polynomial_mutation(c2, bounds, mutation_rate)

            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        # 评估子代
        obj_offspring = np.array([obj_func(ind) for ind in offspring])

        # --- 合并父代和子代 ---
        combined_pop = np.vstack([pop, offspring])
        combined_obj = np.vstack([obj, obj_offspring])

        # --- 非支配排序 ---
        fronts = non_dominated_sort(combined_obj)

        # --- 拥挤距离 + 选择 ---
        new_pop = np.zeros_like(pop)
        new_obj = np.zeros_like(obj)
        count = 0

        for front in fronts:
            if count + len(front) <= pop_size:
                for idx in front:
                    new_pop[count] = combined_pop[idx]
                    new_obj[count] = combined_obj[idx]
                    count += 1
            else:
                # 需要用拥挤距离截断
                distances = crowding_distance(combined_obj[front])
                sorted_indices = np.argsort(-distances)  # 降序
                remaining = pop_size - count
                for j in range(remaining):
                    idx = front[sorted_indices[j]]
                    new_pop[count] = combined_pop[idx]
                    new_obj[count] = combined_obj[idx]
                    count += 1
                break

        pop = new_pop
        obj = new_obj

        # 记录Pareto前沿大小
        front0 = non_dominated_sort(obj)[0]
        history.append(len(front0))

        if (gen + 1) % 20 == 0:
            print(f"  Gen {gen+1}/{n_gen}: Pareto front size = {len(front0)}")

    # 提取最终Pareto前沿
    front0 = non_dominated_sort(obj)[0]
    pareto_front = obj[front0]
    pareto_set = pop[front0]

    return pareto_front, pareto_set, history


def non_dominated_sort(obj):
    """快速非支配排序"""
    n = len(obj)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(obj[i], obj[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(obj[j], obj[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def dominates(a, b):
    """a是否支配b (所有目标不差, 至少一个严格更好)"""
    return np.all(a <= b) and np.any(a < b)


def crowding_distance(obj):
    """计算拥挤距离"""
    n = len(obj)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    n_obj = obj.shape[1] if obj.ndim > 1 else 1

    for m in range(n_obj):
        sorted_idx = np.argsort(obj[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = obj[sorted_idx[-1], m] - obj[sorted_idx[0], m]
        if obj_range < 1e-10:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += \
                (obj[sorted_idx[i+1], m] - obj[sorted_idx[i-1], m]) / obj_range

    return distances


def tournament_select(pop, obj, tournament_size=3):
    """锦标赛选择"""
    indices = np.random.choice(len(pop), tournament_size, replace=False)
    # 选择非支配等级最高的; 简化: 按第一个目标排序
    best = indices[np.argmin(obj[indices, 0])]
    return pop[best].copy()


def sbx_crossover(p1, p2, bounds, eta_c=20):
    """模拟二进制交叉 (SBX)"""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if np.random.rand() < 0.5:
            if abs(p1[i] - p2[i]) > 1e-10:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                c1[i] = np.clip(c1[i], bounds[i, 0], bounds[i, 1])
                c2[i] = np.clip(c2[i], bounds[i, 0], bounds[i, 1])
    return c1, c2


def polynomial_mutation(x, bounds, pm, eta_m=20):
    """多项式变异"""
    x_new = x.copy()
    for i in range(len(x)):
        if np.random.rand() < pm:
            delta_max = bounds[i, 1] - bounds[i, 0]
            u = np.random.rand()
            if u < 0.5:
                delta = (2*u)**(1/(eta_m+1)) - 1
            else:
                delta = 1 - (2*(1-u))**(1/(eta_m+1))
            x_new[i] = x[i] + delta * delta_max
            x_new[i] = np.clip(x_new[i], bounds[i, 0], bounds[i, 1])
    return x_new


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 设计空间
    # [chord_30%, chord_60%, chord_90%, twist_30%, twist_60%, twist_90%]
    bounds = np.array([
        [0.06, 0.18],   # chord r/R=0.3 (c/R)
        [0.05, 0.16],   # chord r/R=0.6
        [0.03, 0.12],   # chord r/R=0.9
        [20.0, 45.0],   # twist r/R=0.3 (deg)
        [8.0,  25.0],   # twist r/R=0.6
        [2.0,  15.0],   # twist r/R=0.9
    ])

    def multi_objective(x):
        chord_ctrl = x[:3]
        twist_ctrl = x[3:]
        _, _, eta, _, _, _ = simple_bem_analysis(chord_ctrl, twist_ctrl)
        spl = simple_noise_model(chord_ctrl, twist_ctrl)
        return np.array([-eta, spl])  # 最小化: [-η, SPL]

    print("Running NSGA-II optimization...")
    pareto_front, pareto_set, history = nsga2_optimize(
        multi_objective, bounds,
        pop_size=60, n_gen=80,
        crossover_rate=0.9, mutation_rate=0.15
    )

    print(f"\nPareto front: {len(pareto_front)} solutions")
    print(f"η range:   [{-pareto_front[:,0].max():.4f}, {-pareto_front[:,0].min():.4f}]")
    print(f"SPL range: [{pareto_front[:,1].min():.1f}, {pareto_front[:,1].max():.1f}] dB")

    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- 子图1: Pareto前沿 ---
    ax = axes[0, 0]
    ax.scatter(-pareto_front[:, 0], pareto_front[:, 1],
               c='steelblue', s=60, alpha=0.8, edgecolors='navy', zorder=5)
    ax.set_xlabel('Efficiency η', fontsize=12)
    ax.set_ylabel('Noise SPL (dB)', fontsize=12)
    ax.set_title('Pareto Front: Efficiency vs Noise', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注极端解和折中解
    idx_best_eta = np.argmin(pareto_front[:, 0])
    idx_best_spl = np.argmin(pareto_front[:, 1])
    # 折中解: 最小归一化距离到理想点
    f_norm = pareto_front.copy()
    for j in range(2):
        fmin, fmax = f_norm[:, j].min(), f_norm[:, j].max()
        if fmax > fmin:
            f_norm[:, j] = (f_norm[:, j] - fmin) / (fmax - fmin)
    dist = np.sqrt(f_norm[:, 0]**2 + f_norm[:, 1]**2)
    idx_compromise = np.argmin(dist)

    for idx, label, color, marker in [
        (idx_best_eta, 'Best η', 'red', '*'),
        (idx_best_spl, 'Lowest SPL', 'green', 's'),
        (idx_compromise, 'Compromise', 'orange', 'D'),
    ]:
        ax.scatter(-pareto_front[idx, 0], pareto_front[idx, 1],
                   c=color, s=200, marker=marker, edgecolors='k',
                   zorder=10, label=label)
    ax.legend(fontsize=10)

    # --- 子图2: 收敛历史 ---
    ax = axes[0, 1]
    ax.plot(range(1, len(history)+1), history, 'b-', lw=2)
    ax.set_title('Pareto Front Size vs Generation', fontsize=13, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('# Pareto solutions')
    ax.grid(True, alpha=0.3)

    # --- 子图3: 设计变量平行坐标图 ---
    ax = axes[0, 2]
    param_names = ['c₃₀', 'c₆₀', 'c₉₀', 'θ₃₀', 'θ₆₀', 'θ₉₀']
    # 归一化到 [0,1]
    ps_norm = (pareto_set - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    # 用效率着色
    eta_vals = -pareto_front[:, 0]
    colors = plt.cm.coolwarm((eta_vals - eta_vals.min()) / (eta_vals.max() - eta_vals.min() + 1e-10))

    for i in range(len(pareto_set)):
        ax.plot(range(6), ps_norm[i], '-', color=colors[i], alpha=0.5, lw=1.5)

    ax.set_xticks(range(6))
    ax.set_xticklabels(param_names, fontsize=10)
    ax.set_ylabel('Normalized value')
    ax.set_title('Parallel Coordinates (color=η)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # --- 子图4-6: 三个典型解的桨叶几何对比 ---
    solutions = {
        'Best η': (pareto_set[idx_best_eta], 'red'),
        'Compromise': (pareto_set[idx_compromise], 'orange'),
        'Lowest SPL': (pareto_set[idx_best_spl], 'green'),
    }

    # 子图4: 弦长分布对比
    ax = axes[1, 0]
    for name, (params, color) in solutions.items():
        _, _, _, chord, twist, r_R = simple_bem_analysis(params[:3], params[3:])
        ax.plot(r_R, chord * 1000, '-', lw=2.5, color=color, label=name)
    ax.set_title('Chord Distribution Comparison', fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Chord (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图5: 扭转分布对比
    ax = axes[1, 1]
    for name, (params, color) in solutions.items():
        _, _, _, chord, twist, r_R = simple_bem_analysis(params[:3], params[3:])
        ax.plot(r_R, twist, '-', lw=2.5, color=color, label=name)
    ax.set_title('Twist Distribution Comparison', fontsize=13, fontweight='bold')
    ax.set_xlabel('r/R')
    ax.set_ylabel('Twist (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图6: 目标权衡雷达图
    ax = axes[1, 2]
    categories = ['η', '1/SPL', 'Chord\narea', 'Twist\nrange', 'Tip\nchord']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for name, (params, color) in solutions.items():
        _, _, eta, chord, twist, r_R = simple_bem_analysis(params[:3], params[3:])
        spl = simple_noise_model(params[:3], params[3:])

        vals = [
            eta / 0.95,                                    # 效率 (归一化)
            1 - (spl - 40) / 60,                          # 噪声倒数 (归一化)
            np.trapezoid(chord, r_R) / 0.003,               # 弦长面积
            (twist.max() - twist.min()) / 40,              # 扭转范围
            chord[-1] / chord.max(),                       # 叶尖弦长比
        ]
        vals = [max(min(v, 1), 0) for v in vals]
        vals += vals[:1]

        ax.plot(angles, vals, '-o', lw=2, color=color, label=name, ms=5)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Design Trade-off Radar', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/parametric_codes/python/nsga2_demo.png', dpi=150)
    plt.close()
    print("NSGA-II demo saved.")
