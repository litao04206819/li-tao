%% CST (Class-Shape Transformation) 翼型参数化方法
%  参考: Kulfan B.M., J. Aircraft, 2008.
%  核心: y(psi) = C(psi) * S(psi) + psi * dz_te
%  C = psi^N1 * (1-psi)^N2, S = Sum Ai * Bernstein_i
%
%  运行: 直接执行此脚本即可生成四张图

clear; clc; close all;

%% ========== 核心函数 (定义在文件末尾) ==========
% bernstein_basis, cst_airfoil, fit_cst, naca0012

%% ========== 演示1: 直接生成翼型 ==========
A_up = [0.17, 0.16, 0.14, 0.16, 0.12, 0.10];
A_lo = [-0.17, -0.08, -0.06, -0.04, -0.02, -0.005];

[x, yu, yl] = cst_airfoil(A_up, A_lo);

figure('Position', [100 100 1200 900]);
subplot(2,2,1);
fill([x, fliplr(x)], [yu, fliplr(yl)], [0.7 0.85 1], 'FaceAlpha', 0.3); hold on;
plot(x, yu, 'b-', 'LineWidth', 2);
plot(x, yl, 'r-', 'LineWidth', 2);
axis equal; grid on;
title('CST Direct Generation', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

%% ========== 演示2: 拟合 NACA 0012 ==========
subplot(2,2,2);
x_fit = linspace(0.001, 1, 200);
y_target = naca0012(x_fit);

colors = lines(4);
for idx = 1:3
    orders = [3, 5, 7];
    order = orders(idx);
    A_fitted = fit_cst(x_fit, y_target, order);
    [xg, yug, ~] = cst_airfoil(A_fitted, -A_fitted);
    plot(xg, yug, '-', 'LineWidth', 2, 'Color', colors(idx,:), ...
         'DisplayName', sprintf('order=%d', order)); hold on;
end
plot(x_fit, y_target, 'k--', 'LineWidth', 1.5, 'DisplayName', 'NACA 0012 exact');
axis equal; grid on; legend('Location', 'best');
title('CST Fitting NACA 0012', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

%% ========== 演示3: 单个系数的影响 ==========
subplot(2,2,3);
A_base = [0.15, 0.15, 0.15, 0.15, 0.15];
[xb, yub, ~] = cst_airfoil(A_base, -A_base);
plot(xb, yub, 'k--', 'LineWidth', 2, 'DisplayName', 'Baseline'); hold on;

for i = 1:5
    A_mod = A_base;
    A_mod(i) = A_mod(i) + 0.08;
    [xm, yum, ~] = cst_airfoil(A_mod, -A_base);
    plot(xm, yum, '-', 'LineWidth', 2, ...
         'DisplayName', sprintf('A[%d] += 0.08', i));
end
axis equal; grid on; legend('Location', 'best', 'FontSize', 8);
title('Effect of Each CST Coefficient', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

%% ========== 演示4: Bernstein基函数 ==========
subplot(2,2,4);
psi = linspace(0, 1, 200);
n = 5;
for k = 0:n
    bk = bernstein_basis(n, k, psi);
    plot(psi, bk, '-', 'LineWidth', 2, ...
         'DisplayName', sprintf('B_{%d,%d}', k, n)); hold on;
end
grid on; legend('Location', 'best');
title(sprintf('Bernstein Basis Functions (n=%d)', n), ...
      'FontSize', 13, 'FontWeight', 'bold');
xlabel('\psi = x/c'); ylabel('B(\psi)');

sgtitle('CST Parametric Method', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function B = bernstein_basis(n, k, psi)
    % n阶第k个Bernstein多项式
    B = nchoosek(n, k) .* psi.^k .* (1 - psi).^(n - k);
end

function [x, yu, yl] = cst_airfoil(A_upper, A_lower, varargin)
    % CST方法生成翼型
    % 可选参数: n_pts, N1, N2, dz_te
    p = inputParser;
    addOptional(p, 'n_pts', 201);
    addOptional(p, 'N1', 0.5);
    addOptional(p, 'N2', 1.0);
    addOptional(p, 'dz_te', 0.0);
    parse(p, varargin{:});

    n_pts = p.Results.n_pts;
    N1 = p.Results.N1;
    N2 = p.Results.N2;
    dz_te = p.Results.dz_te;

    x = linspace(0, 1, n_pts);

    % 类别函数
    C = x.^N1 .* (1 - x).^N2;

    % 形状函数
    n_up = length(A_upper) - 1;
    n_lo = length(A_lower) - 1;

    S_upper = zeros(size(x));
    for k = 0:n_up
        S_upper = S_upper + A_upper(k+1) * bernstein_basis(n_up, k, x);
    end

    S_lower = zeros(size(x));
    for k = 0:n_lo
        S_lower = S_lower + A_lower(k+1) * bernstein_basis(n_lo, k, x);
    end

    yu = C .* S_upper + x * dz_te;
    yl = C .* S_lower - x * dz_te;
end

function A = fit_cst(x_data, y_data, order, varargin)
    % CST最小二乘拟合
    p = inputParser;
    addOptional(p, 'N1', 0.5);
    addOptional(p, 'N2', 1.0);
    parse(p, varargin{:});

    N1 = p.Results.N1;
    N2 = p.Results.N2;
    C = x_data.^N1 .* (1 - x_data).^N2;

    B = zeros(length(x_data), order + 1);
    for k = 0:order
        B(:, k+1) = C(:) .* bernstein_basis(order, k, x_data(:));
    end

    A = (B \ y_data(:))';
end

function y = naca0012(x)
    t = 0.12;
    y = 5*t*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x.^2 ...
             + 0.2843*x.^3 - 0.1015*x.^4);
end
