%% Hicks-Henne 凸包函数翼型参数化方法
%  参考: Hicks R.M. & Henne P.A., J. Aircraft, 1978.
%  核心: y = y_base + Σ b_i * f_i(ψ), 在基准翼型上叠加凸包
%
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 演示 ==========
figure('Position', [100 100 1200 900]);
x = linspace(0, 1, 301);

% --- 示例1: 凸包函数形状 ---
subplot(2,2,1);
positions = [0.1, 0.25, 0.5, 0.75, 0.9];
for idx = 1:length(positions)
    f = hh_bump(x, positions(idx), 3.0);
    plot(x, f, '-', 'LineWidth', 2, ...
         'DisplayName', sprintf('t = %.2f', positions(idx))); hold on;
end
grid on; legend('Location', 'best');
title('Hicks-Henne Bump Functions (e=3)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('\psi = x/c'); ylabel('f(\psi)');

% --- 示例2: 宽度指数e的影响 ---
subplot(2,2,2);
e_vals = [1, 2, 3, 5, 8];
for idx = 1:length(e_vals)
    f = hh_bump(x, 0.5, e_vals(idx));
    plot(x, f, '-', 'LineWidth', 2, ...
         'DisplayName', sprintf('e = %d', e_vals(idx))); hold on;
end
grid on; legend('Location', 'best');
title('Effect of Width Exponent e (t=0.5)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('\psi = x/c'); ylabel('f(\psi)');

% --- 示例3: 在NACA0012上叠加凸包 ---
y_base = naca0012_upper(x);

subplot(2,2,3);
plot(x, y_base, 'k--', 'LineWidth', 2, 'DisplayName', 'NACA 0012'); hold on;

% 前缘增厚
bumps_A = [0.15, 0.010; 0.25, 0.008];
y_A = hh_deform(x, y_base, bumps_A);
plot(x, y_A, 'b-', 'LineWidth', 2, 'DisplayName', '+ front bumps');

% 后缘减薄
bumps_B = [0.65, -0.005; 0.80, -0.008];
y_B = hh_deform(x, y_base, bumps_B);
plot(x, y_B, 'r-', 'LineWidth', 2, 'DisplayName', '- rear bumps');

axis equal; grid on; legend('Location', 'best');
title('Deforming NACA 0012 with Bumps', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例4: 8个凸包均匀覆盖 ---
subplot(2,2,4);
plot(x, y_base, 'k--', 'LineWidth', 2, 'DisplayName', 'Baseline'); hold on;

rng(42);
n_bumps = 8;
t_pos = linspace(0.05, 0.95, n_bumps);
amps = 0.006 * (rand(1, n_bumps) - 0.5);
bumps_rand = [t_pos', amps'];
y_rand = hh_deform(x, y_base, bumps_rand);
plot(x, y_rand, 'b-', 'LineWidth', 2, ...
     'DisplayName', sprintf('%d random bumps', n_bumps));

axis equal; grid on; legend('Location', 'best');
title(sprintf('Random Perturbation (%d Bumps)', n_bumps), ...
      'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

sgtitle('Hicks-Henne Parametric Method', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function f = hh_bump(psi, t_i, e_i)
    % 单个Hicks-Henne凸包函数
    psi_safe = max(min(psi, 1-1e-10), 1e-10);
    exponent = log(0.5) / log(t_i);
    f = sin(pi * psi_safe.^exponent).^e_i;
end

function y_new = hh_deform(x, y_base, bumps, e_i)
    % bumps: Nx2 矩阵, 每行 [t_i, b_i]
    if nargin < 4, e_i = 3.0; end
    y_new = y_base;
    for k = 1:size(bumps, 1)
        y_new = y_new + bumps(k,2) * hh_bump(x, bumps(k,1), e_i);
    end
end

function y = naca0012_upper(x)
    t = 0.12;
    y = 5*t*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x.^2 ...
             + 0.2843*x.^3 - 0.1015*x.^4);
end
