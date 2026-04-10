%% 自由形变 (FFD) 方法
%  参考: Sederberg & Parry, ACM SIGGRAPH, 1986.
%        Kenway & Martins, AIAA Journal, 2016.
%  核心: 将物体嵌入控制体, 移动控制点实现空间变形
%
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 生成NACA0012翼型 ==========
n_af = 101;
x_af = linspace(0, 1, n_af);
t = 0.12;
yt = 5*t*(0.2969*sqrt(x_af) - 0.1260*x_af - 0.3516*x_af.^2 ...
          + 0.2843*x_af.^3 - 0.1015*x_af.^4);
airfoil = [x_af', yt'; flipud([x_af', -yt'])];

%% ========== 演示 ==========
figure('Position', [100 100 1200 900]);

% --- 示例1: 增厚 ---
subplot(2,2,1);
[P, nx, ny] = init_ffd([-0.05 1.05], [-0.12 0.12], 5, 3);
% 上排外移, 下排外移
for i = 1:nx
    P(i, ny, 2) = P(i, ny, 2) + 0.03;
    P(i, 1, 2)  = P(i, 1, 2)  - 0.03;
end
deformed = ffd_deform(airfoil, P, [-0.05 1.05], [-0.12 0.12]);
plot_ffd(gca, P, airfoil, deformed);
title('FFD: Thickness Increase', 'FontSize', 13, 'FontWeight', 'bold');

% --- 示例2: 弯度 ---
subplot(2,2,2);
[P, nx, ny] = init_ffd([-0.05 1.05], [-0.12 0.12], 5, 3);
for j = 1:ny
    P(2, j, 2) = P(2, j, 2) + 0.02;
    P(3, j, 2) = P(3, j, 2) + 0.035;
    P(4, j, 2) = P(4, j, 2) + 0.02;
end
deformed = ffd_deform(airfoil, P, [-0.05 1.05], [-0.12 0.12]);
plot_ffd(gca, P, airfoil, deformed);
title('FFD: Camber Change', 'FontSize', 13, 'FontWeight', 'bold');

% --- 示例3: 局部后缘修型 ---
subplot(2,2,3);
[P, nx, ny] = init_ffd([-0.05 1.05], [-0.12 0.12], 7, 3);
P(6, ny, 2) = P(6, ny, 2) + 0.02;
P(7, ny, 2) = P(7, ny, 2) - 0.01;
P(6, 1, 2)  = P(6, 1, 2)  + 0.01;
P(7, 2, 1)  = P(7, 2, 1)  + 0.02;
deformed = ffd_deform(airfoil, P, [-0.05 1.05], [-0.12 0.12]);
plot_ffd(gca, P, airfoil, deformed);
title('FFD: Trailing-Edge Mod (7×3)', 'FontSize', 13, 'FontWeight', 'bold');

% --- 示例4: 扭转 ---
subplot(2,2,4);
[P, nx, ny] = init_ffd([-0.05 1.05], [-0.12 0.12], 5, 3);
twist = deg2rad(5);
for j = 1:ny
    dy = -0.5 * sin(twist) * 0.5;
    P(4, j, 2) = P(4, j, 2) - dy*(j-2)*0.5;
    P(5, j, 2) = P(5, j, 2) - dy*(j-2);
end
deformed = ffd_deform(airfoil, P, [-0.05 1.05], [-0.12 0.12]);
plot_ffd(gca, P, airfoil, deformed);
title('FFD: Twist Deformation', 'FontSize', 13, 'FontWeight', 'bold');

sgtitle('Free-Form Deformation (FFD)', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function [P, nx, ny] = init_ffd(x_range, y_range, nx, ny)
    % 初始化FFD控制点网格
    xs = linspace(x_range(1), x_range(2), nx);
    ys = linspace(y_range(1), y_range(2), ny);
    P = zeros(nx, ny, 2);
    for i = 1:nx
        for j = 1:ny
            P(i, j, :) = [xs(i), ys(j)];
        end
    end
end

function pts_new = ffd_deform(pts, P, x_range, y_range)
    % FFD变形
    [nx, ny, ~] = size(P);
    l = nx - 1; m = ny - 1;
    N = size(pts, 1);

    % 世界坐标 → 局部坐标
    s = (pts(:,1) - x_range(1)) / (x_range(2) - x_range(1));
    t_loc = (pts(:,2) - y_range(1)) / (y_range(2) - y_range(1));
    s = max(min(s, 1), 0);
    t_loc = max(min(t_loc, 1), 0);

    pts_new = zeros(N, 2);
    for k = 1:N
        for i = 1:nx
            Bi = nchoosek(l, i-1) * s(k)^(i-1) * (1-s(k))^(l-i+1);
            for j = 1:ny
                Bj = nchoosek(m, j-1) * t_loc(k)^(j-1) * (1-t_loc(k))^(m-j+1);
                pts_new(k, :) = pts_new(k, :) + Bi * Bj * squeeze(P(i, j, :))';
            end
        end
    end
end

function plot_ffd(ax, P, original, deformed)
    axes(ax); hold on;
    [nx, ny, ~] = size(P);
    % 控制网格
    for i = 1:nx
        plot(squeeze(P(i,:,1)), squeeze(P(i,:,2)), 'g-o', ...
             'MarkerSize', 5, 'LineWidth', 1, 'Color', [0 0.7 0]);
    end
    for j = 1:ny
        plot(squeeze(P(:,j,1)), squeeze(P(:,j,2)), 'g-o', ...
             'MarkerSize', 5, 'LineWidth', 1, 'Color', [0 0.7 0]);
    end
    plot(original(:,1), original(:,2), 'b--', 'LineWidth', 1.5);
    plot(deformed(:,1), deformed(:,2), 'r-', 'LineWidth', 2.5);
    axis equal; grid on;
    legend('', '', 'Original', 'Deformed', 'Location', 'best');
end
