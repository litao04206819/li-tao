%% 三维螺旋桨桨叶几何生成器
%  整合 CST翼型 + 展向样条 → 三维叶片表面 + STL/CSV导出
%
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 桨叶参数定义 ==========
R = 0.15;      % 叶尖半径 m
Rhub = 0.02;   % 叶根半径 m
B = 3;         % 叶片数

% CST翼型系数 (低雷诺数翼型)
A_up = [0.18, 0.22, 0.18, 0.16, 0.12, 0.08];
A_lo = [-0.18, -0.06, -0.02, -0.01, 0.00, -0.01];

% 展向控制点
r_ctrl = [0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95];
chord_ctrl = [0.012, 0.020, 0.024, 0.022, 0.018, 0.012, 0.005];
twist_ctrl = [38, 30, 22, 16, 11, 7, 4];       % deg
sweep_ctrl = [0, 0, 0, 0.001, 0.002, 0.003, 0.004];  % m

%% ========== 展向插值 ==========
n_span = 25;
r_span = linspace(r_ctrl(1), r_ctrl(end), n_span);
chord_span = spline(r_ctrl, chord_ctrl, r_span);
twist_span = spline(r_ctrl, twist_ctrl, r_span);  % deg
sweep_span = spline(r_ctrl, sweep_ctrl, r_span);

%% ========== 生成三维表面 ==========
n_chord = 51;
n_total = 2*n_chord - 2;  % 闭合截面

X_surf = zeros(n_span, n_total);
Y_surf = zeros(n_span, n_total);
Z_surf = zeros(n_span, n_total);

psi = linspace(0, 1, n_chord);
C = psi.^0.5 .* (1-psi).^1.0;

% 计算CST翼型
n_up = length(A_up) - 1;
n_lo = length(A_lo) - 1;
S_up = zeros(size(psi)); S_lo = zeros(size(psi));
for k = 0:n_up
    S_up = S_up + A_up(k+1) * nchoosek(n_up,k) .* psi.^k .* (1-psi).^(n_up-k);
end
for k = 0:n_lo
    S_lo = S_lo + A_lo(k+1) * nchoosek(n_lo,k) .* psi.^k .* (1-psi).^(n_lo-k);
end
dz_te = 0.001;
yu = C .* S_up + psi * dz_te;
yl = C .* S_lo - psi * dz_te;

% 闭合截面坐标 (上→下)
x_sec = [psi, fliplr(psi(2:end-1))];
y_sec = [yu, fliplr(yl(2:end-1))];

for i = 1:n_span
    r = r_span(i) * R;
    c = chord_span(i);
    tw = deg2rad(twist_span(i));
    sw = sweep_span(i);
    pa = 0.25;  % 变距轴

    % 缩放
    xs = x_sec * c;
    ys = y_sec * c;

    % 平移变距轴到原点
    xs = xs - pa * c;

    % 扭转旋转
    x_rot = xs * cos(tw) - ys * sin(tw);
    y_rot = xs * sin(tw) + ys * cos(tw);

    % 三维坐标
    X_surf(i, :) = y_rot;       % 轴向
    Y_surf(i, :) = r;           % 展向(径向)
    Z_surf(i, :) = x_rot + sw;  % 弦向 + 后掠
end

%% ========== 多叶片 ==========
figure('Position', [50 50 1600 700]);

% --- 单叶片 ---
subplot(1,3,1);
surf(Z_surf*1000, Y_surf*1000, X_surf*1000, ...
     'FaceColor', 'interp', 'EdgeColor', [0.5 0.5 0.5], 'EdgeAlpha', 0.2);
colormap(cool);
xlabel('Z (mm)'); ylabel('r (mm)'); zlabel('X (mm)');
title('Single Blade', 'FontSize', 13, 'FontWeight', 'bold');
axis equal; grid on; view([-60 25]);
light('Position', [1 1 1]);

% --- 全桨 ---
subplot(1,3,2);
hold on;
cmaps = {cool, autumn, winter};
for b = 1:B
    angle = 2*pi*(b-1)/B;
    Y_rot = Y_surf*cos(angle) - Z_surf*sin(angle);
    Z_rot = Y_surf*sin(angle) + Z_surf*cos(angle);
    surf(Z_rot*1000, Y_rot*1000, X_surf*1000, ...
         'FaceAlpha', 0.8, 'EdgeColor', 'none');
end
% 桨毂
[th_h, z_h] = meshgrid(linspace(0,2*pi,30), [-2 2]);
x_h = Rhub*1000*cos(th_h);
y_h = Rhub*1000*sin(th_h);
surf(x_h, y_h, z_h, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none');
xlabel('Z (mm)'); ylabel('Y (mm)'); zlabel('X (mm)');
title(sprintf('%d-Blade Propeller', B), 'FontSize', 13, 'FontWeight', 'bold');
axis equal; grid on; view([-45 30]);
light('Position', [1 1 1]); lighting gouraud;

% --- 展向分布 ---
subplot(1,3,3);
yyaxis left;
plot(r_span, chord_span*1000, 'b-', 'LineWidth', 2.5); hold on;
plot(r_ctrl, chord_ctrl*1000, 'b^', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('Chord (mm)');
yyaxis right;
plot(r_span, twist_span, 'r-', 'LineWidth', 2.5);
plot(r_ctrl, twist_ctrl, 'rv', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('Twist (deg)');
xlabel('r/R'); grid on;
title('Spanwise Distributions', 'FontSize', 13, 'FontWeight', 'bold');
legend('Chord', 'Ctrl (chord)', 'Twist', 'Ctrl (twist)', 'Location', 'best');

sgtitle('3D Propeller Blade Generator', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 导出CSV ==========
fid = fopen('/home/claude/parametric_codes/matlab/blade_surface.csv', 'w');
fprintf(fid, 'X,Y,Z,Section\n');
for i = 1:n_span
    for j = 1:n_total
        fprintf(fid, '%.6f,%.6f,%.6f,%d\n', X_surf(i,j), Y_surf(i,j), Z_surf(i,j), i);
    end
end
fclose(fid);
fprintf('Blade surface exported to blade_surface.csv\n');

%% ========== 截面曲线叠图 ==========
figure('Position', [200 200 600 500]);
n_show = 8;
indices = round(linspace(1, n_span, n_show));
for idx = indices
    c = chord_span(idx) * 1000;  % mm
    offset = r_span(idx) * R * 1000 * 0.15;
    fill([psi*c, fliplr(psi*c)], [yu*c+offset, fliplr(yl*c+offset)], ...
         [0.7 0.85 1], 'FaceAlpha', 0.3, 'EdgeColor', 'b');
    hold on;
    text(c+0.5, offset, sprintf('r/R=%.2f\nc=%.1fmm', r_span(idx), c), ...
         'FontSize', 7);
end
axis equal; grid on;
title('Stacked Airfoil Sections', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Chord (mm)');
