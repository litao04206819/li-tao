%% B样条 / NURBS 翼型参数化方法
%  参考: Piegl L. & Tiller W., "The NURBS Book," Springer, 1997.
%  核心: C(u) = Σ P_i · N_{i,p}(u), 移动控制点改变翼型形状
%
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 演示 ==========
figure('Position', [100 100 1200 900]);

% 控制点定义
ctrl_up = [0.00 0.000;
           0.00 0.030;
           0.10 0.065;
           0.30 0.065;
           0.60 0.045;
           0.85 0.018;
           1.00 0.001];

ctrl_lo = [0.00  0.000;
           0.00 -0.025;
           0.10 -0.050;
           0.30 -0.048;
           0.60 -0.025;
           0.85 -0.010;
           1.00 -0.001];

p = 3;  % 三次B样条

% --- 示例1: 基本B样条翼型 ---
subplot(2,2,1);
[xu, yu] = bspline_curve(ctrl_up, p);
[xl, yl] = bspline_curve(ctrl_lo, p);

fill([xu; flipud(xl)], [yu; flipud(yl)], [0.7 0.85 1], 'FaceAlpha', 0.3); hold on;
plot(xu, yu, 'b-', 'LineWidth', 2);
plot(xl, yl, 'r-', 'LineWidth', 2);
plot(ctrl_up(:,1), ctrl_up(:,2), 'b^--', 'MarkerSize', 8, ...
     'MarkerFaceColor', [0.3 0.3 1], 'DisplayName', 'Ctrl (upper)');
plot(ctrl_lo(:,1), ctrl_lo(:,2), 'rv--', 'MarkerSize', 8, ...
     'MarkerFaceColor', [1 0.3 0.3], 'DisplayName', 'Ctrl (lower)');
axis equal; grid on; legend('Location', 'best');
title('B-Spline Airfoil (p=3)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例2: 移动控制点P[4]的效果 ---
subplot(2,2,2);
[xb, yb] = bspline_curve(ctrl_up, p);
plot(xb, yb, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Original'); hold on;

deltas = [0.02, 0.04, -0.02];
colors_arr = [0 0 1; 0 0.7 0; 1 0 0];
for idx = 1:3
    ctrl_mod = ctrl_up;
    ctrl_mod(4, 2) = ctrl_mod(4, 2) + deltas(idx);
    [xm, ym] = bspline_curve(ctrl_mod, p);
    plot(xm, ym, '-', 'LineWidth', 2, 'Color', colors_arr(idx,:), ...
         'DisplayName', sprintf('P[4].y += %.3f', deltas(idx)));
    plot(ctrl_mod(4,1), ctrl_mod(4,2), 'o', 'MarkerSize', 10, ...
         'Color', colors_arr(idx,:), 'MarkerFaceColor', colors_arr(idx,:));
end
axis equal; grid on; legend('Location', 'best');
title('Moving Control Point P[4]', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例3: 不同阶数p ---
subplot(2,2,3);
for deg = [2, 3, 4, 5]
    if deg < size(ctrl_up, 1)
        [xp, yp] = bspline_curve(ctrl_up, deg);
        plot(xp, yp, '-', 'LineWidth', 2, ...
             'DisplayName', sprintf('p = %d', deg)); hold on;
    end
end
plot(ctrl_up(:,1), ctrl_up(:,2), 'ks--', 'MarkerSize', 6, ...
     'DisplayName', 'Ctrl pts');
axis equal; grid on; legend('Location', 'best');
title('Effect of B-Spline Degree', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例4: B样条基函数 ---
subplot(2,2,4);
n_ctrl = 7;
deg = 3;
U = uniform_knot_vector(n_ctrl - 1, deg);
u_plot = linspace(0, 1 - 1e-10, 500);

for i = 0:(n_ctrl-1)
    N_vals = zeros(size(u_plot));
    for uidx = 1:length(u_plot)
        span = find_span(n_ctrl-1, deg, u_plot(uidx), U);
        N = basis_funs(span, u_plot(uidx), deg, U);
        for j = 0:deg
            if (span - deg + j) == i
                N_vals(uidx) = N(j+1);
            end
        end
    end
    plot(u_plot, N_vals, '-', 'LineWidth', 2, ...
         'DisplayName', sprintf('N_{%d,%d}', i, deg)); hold on;
end
grid on; legend('Location', 'best', 'FontSize', 8);
title(sprintf('B-Spline Basis (n=%d, p=%d)', n_ctrl-1, deg), ...
      'FontSize', 13, 'FontWeight', 'bold');
xlabel('u'); ylabel('N(u)');

sgtitle('B-Spline Parametric Method', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function [xc, yc] = bspline_curve(P, p, n_pts)
    if nargin < 3, n_pts = 201; end
    n = size(P, 1) - 1;
    U = uniform_knot_vector(n, p);
    u_vals = linspace(U(p+1), U(n+2) - 1e-10, n_pts);

    xc = zeros(n_pts, 1);
    yc = zeros(n_pts, 1);
    for idx = 1:n_pts
        span = find_span(n, p, u_vals(idx), U);
        N = basis_funs(span, u_vals(idx), p, U);
        for j = 0:p
            ci = span - p + j + 1;  % MATLAB 1-indexed
            xc(idx) = xc(idx) + N(j+1) * P(ci, 1);
            yc(idx) = yc(idx) + N(j+1) * P(ci, 2);
        end
    end
end

function U = uniform_knot_vector(n, p)
    m = n + p + 1;
    U = zeros(1, m + 1);
    for i = 0:m
        if i <= p
            U(i+1) = 0;
        elseif i >= m - p
            U(i+1) = 1;
        else
            U(i+1) = (i - p) / (m - 2*p);
        end
    end
end

function span = find_span(n, p, u, U)
    if u >= U(n+2), span = n; return; end
    if u <= U(p+1), span = p; return; end
    lo = p; hi = n + 1;
    mid = floor((lo + hi) / 2);
    while u < U(mid+1) || u >= U(mid+2)
        if u < U(mid+1)
            hi = mid;
        else
            lo = mid;
        end
        mid = floor((lo + hi) / 2);
    end
    span = mid;
end

function N = basis_funs(i, u, p, U)
    N = zeros(1, p+1);
    left = zeros(1, p+1);
    right = zeros(1, p+1);
    N(1) = 1;
    for j = 1:p
        left(j+1) = u - U(i+1-j+1);
        right(j+1) = U(i+j+1) - u;
        saved = 0;
        for r = 0:(j-1)
            temp = N(r+1) / (right(r+2) + left(j-r+1));
            N(r+1) = saved + right(r+2) * temp;
            saved = left(j-r+1) * temp;
        end
        N(j+1) = saved;
    end
end
