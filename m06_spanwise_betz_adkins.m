%% 展向分布参数化 + Betz-Prandtl + Adkins-Liebeck
%  参考: Ganguli & Chopra, J. Aircraft, 1995 (展向样条)
%        Betz A., 1919; Prandtl (叶尖损失)
%        Adkins & Liebeck, J. Propulsion and Power, 1994
%
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 演示 ==========
figure('Position', [100 100 1200 1000]);
r_R = linspace(0.15, 0.98, 200);

% ==================== 展向分布参数化 ====================

% --- 示例1: 弦长分布 ---
subplot(3,2,1);
r_ctrl = [0.15, 0.30, 0.50, 0.70, 0.85, 0.98];

chord_data = {
    'Rect-like',  [0.08, 0.10, 0.10, 0.10, 0.09, 0.04];
    'Tapered',    [0.12, 0.11, 0.09, 0.07, 0.05, 0.02];
    'High-perf',  [0.06, 0.12, 0.11, 0.08, 0.05, 0.01];
};

for idx = 1:size(chord_data, 1)
    cs = spline(r_ctrl, chord_data{idx, 2});
    c_interp = ppval(cs, r_R);
    plot(r_R, c_interp, '-', 'LineWidth', 2.5, ...
         'DisplayName', chord_data{idx, 1}); hold on;
    plot(r_ctrl, chord_data{idx, 2}, 'o', 'MarkerSize', 7, ...
         'Color', gca().Children(1).Color, 'HandleVisibility', 'off');
end
grid on; legend('Location', 'best');
title('Chord Distribution c(r/R)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); ylabel('c/R');

% --- 示例2: 扭转角分布 ---
subplot(3,2,2);
twist_data = {
    'Linear',   [30, 24, 18, 12, 8, 5];
    'Ideal',    [35, 22, 14, 9, 6, 3];
    'Washout',  [25, 20, 15, 10, 6, 2];
};

for idx = 1:size(twist_data, 1)
    cs = spline(r_ctrl, twist_data{idx, 2});
    th_interp = ppval(cs, r_R);
    plot(r_R, th_interp, '-', 'LineWidth', 2.5, ...
         'DisplayName', twist_data{idx, 1}); hold on;
    plot(r_ctrl, twist_data{idx, 2}, 's', 'MarkerSize', 7, ...
         'Color', gca().Children(1).Color, 'HandleVisibility', 'off');
end
grid on; legend('Location', 'best');
title('Twist Distribution \theta(r/R)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); ylabel('\theta (deg)');

% ==================== Prandtl 叶尖损失 ====================

% --- 示例3: Prandtl因子 ---
subplot(3,2,3);
B_vals = [2, 3, 4, 6];
phi_tip = deg2rad(10);
for idx = 1:length(B_vals)
    F = prandtl_tip_loss(r_R, B_vals(idx), phi_tip);
    plot(r_R, F, '-', 'LineWidth', 2.5, ...
         'DisplayName', sprintf('B = %d', B_vals(idx))); hold on;
end
ylim([0 1.1]); grid on; legend('Location', 'best');
title('Prandtl Tip-Loss Factor F(r/R)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); ylabel('F');

% ==================== Betz 最优环量 ====================

% --- 示例4: 最优环量分布 ---
subplot(3,2,4);
V_inf = 15; omega = 200; B = 3; R = 1.0;
[Gamma, phi, F] = betz_optimal(r_R, V_inf, omega, B, R);
Gamma_norm = Gamma / max(Gamma);

plot(r_R, Gamma_norm, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Betz-Prandtl'); hold on;
plot(r_R, F, 'g--', 'LineWidth', 2, 'DisplayName', 'Prandtl F');
grid on; legend('Location', 'best');
title('Betz Optimal Circulation', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); ylabel('\Gamma / \Gamma_{max}');

% ==================== Adkins-Liebeck ====================

% --- 示例5 & 6: 设计结果 ---
r_des = linspace(0.20, 0.95, 100);
[chord, beta, alpha_d, phi_d] = adkins_liebeck(r_des, 20, 300, 50, 3, 0.6, 0.012, 0.15);

subplot(3,2,5);
plot(r_des, chord*1000, 'b-', 'LineWidth', 2.5);
grid on;
title('Adkins-Liebeck: Chord', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); ylabel('Chord (mm)');

subplot(3,2,6);
plot(r_des, beta, 'r-', 'LineWidth', 2.5, 'DisplayName', '\beta (pitch)'); hold on;
plot(r_des, phi_d, 'b--', 'LineWidth', 2, 'DisplayName', '\phi (inflow)');
grid on; legend('Location', 'best');
title('Adkins-Liebeck: Angles', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); ylabel('Angle (deg)');

sgtitle('Spanwise + Betz-Prandtl + Adkins-Liebeck', ...
        'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function F = prandtl_tip_loss(r_R, B, phi_tip)
    r = r_R;
    R = 1.0;
    exponent = -B * (R - r) ./ (2 * r .* sin(phi_tip) + 1e-10);
    F = (2/pi) * acos(max(min(exp(exponent), 1), -1));
end

function [Gamma, phi, F] = betz_optimal(r_R, V_inf, omega, B, R)
    r = r_R * R;
    phi = atan2(V_inf, omega * r);
    phi_tip = atan2(V_inf, omega * R);
    F = prandtl_tip_loss(r_R, B, phi_tip);
    Gamma = 2*pi * r .* V_inf .* sin(phi) .* cos(phi) .* F / B;
end

function [chord, beta, alpha_d, phi_deg] = adkins_liebeck(r_R, V_inf, omega, T_req, B, Cl_d, Cd_d, R)
    rho = 1.225;
    r = r_R * R;
    lambda_r = omega * r / V_inf;

    phi_tip = atan2(V_inf, omega * R);
    F = prandtl_tip_loss(r_R, B, phi_tip);

    phi = atan2(V_inf, omega * r);

    % 迭代
    for iter = 1:10
        sp = sin(phi); cp = cos(phi);
        a = 1 ./ (1 + 4*F.*sp.^2 ./ (B*Cl_d*cp./(2*pi*r) + 1e-10) + 1);
        a = min(max(a, 0), 0.4);
        a_p = 1 ./ (4*F.*cp ./ (B*Cl_d./(2*pi*r) + 1e-10) - 1 + 1e-10);
        a_p = min(max(a_p, 0), 0.4);
        phi = atan2(V_inf*(1+a), omega*r.*(1-a_p));
    end

    W = sqrt((V_inf*(1+a)).^2 + (omega*r.*(1-a_p)).^2);

    chord = 8*pi*r.*F.*sin(phi)/B/Cl_d .* ...
            (cos(phi) - lambda_r.*sin(phi)) ./ ...
            (sin(phi) + lambda_r.*cos(phi) + 1e-10);
    chord = abs(chord);
    chord = min(max(chord, 0.01*R), 0.5*R);

    alpha_design = rad2deg(atan2(Cl_d, 2*pi));
    beta = rad2deg(phi) + alpha_design;
    alpha_d = alpha_design * ones(size(r_R));
    phi_deg = rad2deg(phi);
end
