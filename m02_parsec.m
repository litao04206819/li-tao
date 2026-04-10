%% PARSEC 翼型参数化方法
%  参考: Sobieczky H., Notes on Numerical Fluid Mechanics, 1999.
%  核心: 12个有明确几何含义的参数定义翼型
%        每面用6阶多项式 z = Σ a_k · ψ^(k-1/2)
%
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 演示 ==========
figure('Position', [100 100 1200 900]);

% --- 示例1: 对称翼型 ---
params_sym.r_le = 0.008;
params_sym.x_up = 0.30; params_sym.z_up = 0.06; params_sym.zxx_up = -0.40;
params_sym.x_lo = 0.30; params_sym.z_lo = -0.06; params_sym.zxx_lo = 0.40;
params_sym.z_te = 0.0; params_sym.dz_te = 0.0;
params_sym.alpha_te = 0.0; params_sym.beta_te = 0.16;

[x, yu, yl] = parsec_airfoil(params_sym);

subplot(2,2,1);
fill([x, fliplr(x)], [yu, fliplr(yl)], [0.7 0.85 1], 'FaceAlpha', 0.3); hold on;
plot(x, yu, 'b-', 'LineWidth', 2);
plot(x, yl, 'r-', 'LineWidth', 2);
axis equal; grid on;
title('Symmetric Airfoil (PARSEC)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例2: 有弯度的翼型 ---
params_camb = params_sym;
params_camb.r_le = 0.010;
params_camb.x_up = 0.35; params_camb.z_up = 0.08; params_camb.zxx_up = -0.45;
params_camb.x_lo = 0.25; params_camb.z_lo = -0.03; params_camb.zxx_lo = 0.30;
params_camb.z_te = 0.002; params_camb.dz_te = 0.001;
params_camb.alpha_te = -0.02; params_camb.beta_te = 0.12;

[x, yu, yl] = parsec_airfoil(params_camb);
subplot(2,2,2);
fill([x, fliplr(x)], [yu, fliplr(yl)], [1 0.85 0.8], 'FaceAlpha', 0.3); hold on;
plot(x, yu, 'b-', 'LineWidth', 2);
plot(x, yl, 'r-', 'LineWidth', 2);
axis equal; grid on;
title('Cambered Airfoil (PARSEC)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例3: 改变最大厚度位置 ---
subplot(2,2,3);
x_up_vals = [0.20, 0.30, 0.40, 0.50];
colors = lines(4);
for idx = 1:4
    p = params_sym;
    p.x_up = x_up_vals(idx);
    p.x_lo = x_up_vals(idx);
    [x, yu, yl] = parsec_airfoil(p);
    plot(x, yu, '-', 'LineWidth', 2, 'Color', colors(idx,:), ...
         'DisplayName', sprintf('x_{up} = %.2f', x_up_vals(idx))); hold on;
    plot(x, yl, '-', 'LineWidth', 2, 'Color', colors(idx,:), ...
         'HandleVisibility', 'off');
end
axis equal; grid on; legend('Location', 'best');
title('Effect of Max-Thickness Position', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例4: 改变前缘半径 ---
subplot(2,2,4);
r_le_vals = [0.004, 0.008, 0.015, 0.025];
for idx = 1:4
    p = params_sym;
    p.r_le = r_le_vals(idx);
    [x, yu, yl] = parsec_airfoil(p);
    plot(x, yu, '-', 'LineWidth', 2, 'Color', colors(idx,:), ...
         'DisplayName', sprintf('r_{LE} = %.3f', r_le_vals(idx))); hold on;
    plot(x, yl, '-', 'LineWidth', 2, 'Color', colors(idx,:), ...
         'HandleVisibility', 'off');
end
xlim([-0.02, 0.25]); ylim([-0.08, 0.10]);
axis equal; grid on; legend('Location', 'best');
title('Effect of Leading-Edge Radius (Zoomed)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

sgtitle('PARSEC Parametric Method', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function [x, yu, yl] = parsec_airfoil(params, n_pts)
    if nargin < 2, n_pts = 201; end

    exps = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];

    % --- 上表面 ---
    [A_up, b_up] = build_parsec_system(params, exps, true);
    a_upper = A_up \ b_up;

    % --- 下表面 ---
    [A_lo, b_lo] = build_parsec_system(params, exps, false);
    a_lower = A_lo \ b_lo;

    % --- 生成坐标 ---
    x = linspace(0, 1, n_pts);
    yu = zeros(size(x));
    yl = zeros(size(x));
    for j = 1:6
        mask = x > 0;
        yu(mask) = yu(mask) + a_upper(j) * x(mask).^exps(j);
        yl(mask) = yl(mask) + a_lower(j) * x(mask).^exps(j);
    end
end

function [A, b] = build_parsec_system(params, exps, is_upper)
    A = zeros(6, 6);
    b = zeros(6, 1);

    if is_upper
        x_ext = params.x_up; z_ext = params.z_up; zxx_ext = params.zxx_up;
    else
        x_ext = params.x_lo; z_ext = params.z_lo; zxx_ext = params.zxx_lo;
    end

    % 约束1: z(1) = z_te ± dz_te
    A(1,:) = ones(1,6);
    if is_upper
        b(1) = params.z_te + params.dz_te;
    else
        b(1) = params.z_te - params.dz_te;
    end

    % 约束2: z(x_ext) = z_ext
    for j = 1:6, A(2,j) = x_ext^exps(j); end
    b(2) = z_ext;

    % 约束3: z'(x_ext) = 0
    for j = 1:6, A(3,j) = exps(j) * x_ext^(exps(j)-1); end
    b(3) = 0;

    % 约束4: z''(x_ext) = zxx
    for j = 1:6, A(4,j) = exps(j)*(exps(j)-1) * x_ext^(exps(j)-2); end
    b(4) = zxx_ext;

    % 约束5: z'(1) = 尾缘斜率
    for j = 1:6, A(5,j) = exps(j); end
    if is_upper
        b(5) = tan(params.alpha_te - params.beta_te/2);
    else
        b(5) = tan(params.alpha_te + params.beta_te/2);
    end

    % 约束6: 前缘半径
    A(6,1) = 1;
    if is_upper
        b(6) = sqrt(2 * params.r_le);
    else
        b(6) = -sqrt(2 * params.r_le);
    end
end
