%% CCBlade 式 BEM 求解器 (Ning保证收敛法)
%  参考: Ning A., Wind Energy, 2014.
%  核心: 将BEM方程整理为f(φ)=0, 用Brent方法求根
%
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 定义螺旋桨 ==========
R = 0.15;       % 叶尖半径 m
Rhub = 0.02;    % 叶根半径 m
B = 3;

r_R = linspace(0.20, 0.98, 25)';
chord = R * (0.12 * sin(pi * (r_R - 0.15) / 0.85) + 0.02);
twist_deg = 30 * (0.3 ./ r_R) - 5;
n_sec = length(r_R);

%% ========== 单工况分析 ==========
V_inf = 15;   % m/s
RPM = 8000;
omega = RPM * 2 * pi / 60;
rho = 1.225;
n_rev = RPM / 60;

phi_arr = zeros(n_sec, 1);
alpha_arr = zeros(n_sec, 1);
a_arr = zeros(n_sec, 1);
ap_arr = zeros(n_sec, 1);
dT_arr = zeros(n_sec, 1);
dQ_arr = zeros(n_sec, 1);

for i = 1:n_sec
    r = r_R(i) * R;
    Vx = V_inf;
    Vy = omega * r;
    tw = deg2rad(twist_deg(i));

    [phi, alpha, Cl, Cd, a, ap, Cn, Ct, W] = ...
        solve_bem(r, R, Rhub, chord(i), tw, Vx, Vy, B);

    phi_arr(i) = phi;
    alpha_arr(i) = alpha;
    a_arr(i) = a;
    ap_arr(i) = ap;
    dT_arr(i) = 0.5 * rho * W^2 * chord(i) * Cn * B;
    dQ_arr(i) = 0.5 * rho * W^2 * chord(i) * Ct * B * r;
end

T = trapz(r_R * R, dT_arr);
Q = trapz(r_R * R, dQ_arr);
P = Q * omega;
D = 2 * R;
CT = T / (rho * n_rev^2 * D^4);
CP = P / (rho * n_rev^3 * D^5);
J = V_inf / (n_rev * D);
eta = J * CT / (CP + 1e-10);

fprintf('BEM Results: T=%.3f N, Q=%.5f N·m, η=%.4f\n', T, Q, eta);

%% ========== J 扫描 ==========
V_range = linspace(2, 35, 30);
J_scan = zeros(size(V_range));
CT_scan = zeros(size(V_range));
CP_scan = zeros(size(V_range));
eta_scan = zeros(size(V_range));

for vi = 1:length(V_range)
    V = V_range(vi);
    T_t = 0; Q_t = 0;
    for i = 1:n_sec
        r = r_R(i) * R;
        tw = deg2rad(twist_deg(i));
        [~,~,~,~,a,ap,Cn,Ct,W] = solve_bem(r,R,Rhub,chord(i),tw,V,omega*r,B);
        dT = 0.5*rho*W^2*chord(i)*Cn*B;
        dQ = 0.5*rho*W^2*chord(i)*Ct*B*r;
        if i > 1
            dr = (r_R(i) - r_R(i-1)) * R;
            T_t = T_t + 0.5*(dT + dT_arr(i)) * dr; % 简化积分
            Q_t = Q_t + 0.5*(dQ + dQ_arr(i)) * dr;
        end
    end
    P_t = Q_t * omega;
    J_scan(vi) = V / (n_rev * D);
    CT_scan(vi) = T_t / (rho * n_rev^2 * D^4);
    CP_scan(vi) = P_t / (rho * n_rev^3 * D^5);
    eta_scan(vi) = J_scan(vi) * CT_scan(vi) / (CP_scan(vi) + 1e-10);
    eta_scan(vi) = max(min(eta_scan(vi), 1), 0);
end

%% ========== 可视化 ==========
figure('Position', [100 100 1400 900]);

subplot(2,3,1);
yyaxis left; plot(r_R, chord*1000, 'b-', 'LineWidth', 2.5); ylabel('Chord (mm)');
yyaxis right; plot(r_R, twist_deg, 'r-', 'LineWidth', 2.5); ylabel('Twist (deg)');
title('Blade Geometry', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); grid on;

subplot(2,3,2);
plot(r_R, rad2deg(alpha_arr), 'b-', 'LineWidth', 2.5); hold on;
plot(r_R, rad2deg(phi_arr), 'r--', 'LineWidth', 2);
legend('\alpha (AoA)', '\phi (inflow)');
title('Section Angles', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); ylabel('Angle (deg)'); grid on;

subplot(2,3,3);
plot(r_R, a_arr, 'b-', 'LineWidth', 2.5); hold on;
plot(r_R, ap_arr, 'r-', 'LineWidth', 2.5);
legend('a (axial)', "a' (tangential)");
title('Induction Factors', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); grid on;

subplot(2,3,4);
yyaxis left; plot(r_R, dT_arr, 'b-', 'LineWidth', 2.5); ylabel('dT/dr (N/m)');
yyaxis right; plot(r_R, dQ_arr, 'r-', 'LineWidth', 2.5); ylabel('dQ/dr (N·m/m)');
title('Loading Distribution', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('r/R'); grid on;

subplot(2,3,5);
yyaxis left;
plot(J_scan, CT_scan, 'b-', 'LineWidth', 2.5); hold on;
plot(J_scan, CP_scan, 'r-', 'LineWidth', 2.5);
ylabel('C_T, C_P'); legend('C_T', 'C_P');
yyaxis right;
plot(J_scan, eta_scan, 'g--', 'LineWidth', 2.5);
ylabel('\eta');
title('Performance Curves', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('J'); grid on;

% 残差函数可视化
subplot(2,3,6);
phi_scan = linspace(0.02, pi/2-0.02, 200);
r_test = 0.5 * R;
c_test = interp1(r_R, chord, 0.5);
tw_test = deg2rad(interp1(r_R, twist_deg, 0.5));
Vy_test = omega * r_test;
resid = zeros(size(phi_scan));
for k = 1:length(phi_scan)
    resid(k) = ning_resid(phi_scan(k), r_test, R, Rhub, c_test, tw_test, V_inf, Vy_test, B);
end
plot(rad2deg(phi_scan), resid, 'b-', 'LineWidth', 2.5); hold on;
yline(0, 'k--');
title('Ning Residual f(\phi) at r/R=0.5', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('\phi (deg)'); ylabel('f(\phi)'); grid on;

sgtitle('CCBlade-style BEM Solver (Brent Method)', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function [phi,alpha,Cl,Cd,a,ap,Cn,Ct,W] = solve_bem(r,R,Rhub,c,tw,Vx,Vy,B)
    eps = 1e-6;
    phi_min = eps; phi_max = pi/2 - eps;

    fa = ning_resid(phi_min, r,R,Rhub,c,tw,Vx,Vy,B);
    fb = ning_resid(phi_max, r,R,Rhub,c,tw,Vx,Vy,B);

    if fa*fb < 0
        phi = brent_solve(@(p) ning_resid(p,r,R,Rhub,c,tw,Vx,Vy,B), phi_min, phi_max, 1e-8);
    else
        % 细分搜索
        phi = atan2(Vx, Vy);
        pts = linspace(phi_min, phi_max, 50);
        for k = 1:49
            f1 = ning_resid(pts(k),r,R,Rhub,c,tw,Vx,Vy,B);
            f2 = ning_resid(pts(k+1),r,R,Rhub,c,tw,Vx,Vy,B);
            if f1*f2 < 0
                phi = brent_solve(@(p) ning_resid(p,r,R,Rhub,c,tw,Vx,Vy,B), pts(k), pts(k+1), 1e-8);
                break;
            end
        end
    end

    alpha = phi - tw;
    [Cl, Cd] = airfoil_cl_cd(alpha);
    sp = sin(phi); cp = cos(phi);
    Cn = Cl*cp + Cd*sp;
    Ct = Cl*sp - Cd*cp;
    sigma = B*c/(2*pi*r);
    F = prandtl_F(r,R,Rhub,B,phi);

    k_val = sigma*Cn/(4*F*sp^2 + 1e-10);
    if k_val <= 2/3
        a = k_val/(1+k_val);
    else
        g2 = 2*F*k_val - F*(4/3-F);
        a = 1 - 1/(2*sqrt(max(g2,1e-10)));
    end
    a = max(min(a,0.95),-0.5);

    kp = sigma*Ct/(4*F*sp*cp + 1e-10);
    if abs(1-kp) > 1e-6
        ap = kp/(1-kp);
    else
        ap = 0;
    end
    ap = max(min(ap,5),-1);
    W = sqrt((Vx*(1-a))^2 + (Vy*(1+ap))^2);
end

function res = ning_resid(phi, r, R, Rhub, c, tw, Vx, Vy, B)
    sp = sin(phi); cp = cos(phi);
    if abs(sp)<1e-6 || abs(cp)<1e-6, res=1e6; return; end

    alpha = phi - tw;
    [Cl, Cd] = airfoil_cl_cd(alpha);
    Cn = Cl*cp + Cd*sp;
    Ct = Cl*sp - Cd*cp;
    sigma = B*c/(2*pi*r);
    F = prandtl_F(r,R,Rhub,B,phi);

    k_val = sigma*Cn/(4*F*sp^2 + 1e-10);
    if k_val <= 2/3
        a = k_val/(1+k_val);
    else
        g2 = 2*F*k_val - F*(4/3-F);
        a = 1 - 1/(2*sqrt(max(g2,1e-10)));
    end
    a = max(min(a,0.95),-0.5);

    kp = sigma*Ct/(4*F*sp*cp + 1e-10);
    if abs(1-kp) > 1e-6, ap = kp/(1-kp); else, ap = 0; end
    ap = max(min(ap,5),-1);

    phi_c = atan2(Vx*(1-a), Vy*(1+ap));
    res = phi - phi_c;
end

function F = prandtl_F(r, R, Rhub, B, phi)
    sp = abs(sin(phi)) + 1e-10;
    f_tip = B*(R-r)/(2*r*sp);
    F_tip = (2/pi)*acos(max(min(exp(-f_tip),1),-1));
    f_hub = B*(r-Rhub)/(2*r*sp);
    F_hub = (2/pi)*acos(max(min(exp(-f_hub),1),-1));
    F = max(F_tip * F_hub, 1e-4);
end

function [Cl, Cd] = airfoil_cl_cd(alpha)
    a_stall = deg2rad(12);
    Cl_slope = 2*pi;
    if abs(alpha) < a_stall
        Cl = Cl_slope * alpha;
    else
        Cl_max = Cl_slope * a_stall;
        Cl = Cl_max * sign(alpha) * (1 - 0.5*(abs(alpha)-a_stall)/a_stall);
    end
    Cd = 0.008 + 0.005*Cl^2;
end

function x = brent_solve(f, a, b, tol)
    fa = f(a); fb = f(b);
    c = a; fc = fa; d = b-a; e = d;
    for iter = 1:200
        if fb*fc > 0, c=a; fc=fa; d=b-a; e=d; end
        if abs(fc) < abs(fb)
            a=b; b=c; c=a; fa=fb; fb=fc; fc=fa;
        end
        m = 0.5*(c-b);
        if abs(m) <= tol || fb == 0, x=b; return; end
        if abs(e) >= tol && abs(fa) > abs(fb)
            s = fb/fa;
            if a == c
                p = 2*m*s; q = 1-s;
            else
                q_v = fa/fc; r_v = fb/fc;
                p = s*(2*m*q_v*(q_v-r_v)-(b-a)*(r_v-1));
                q = (q_v-1)*(r_v-1)*(s-1);
            end
            if p > 0, q = -q; else, p = -p; end
            if 2*p < 3*m*q-abs(tol*q) && 2*p < abs(e*q)
                e = d; d = p/q;
            else
                d = m; e = m;
            end
        else
            d = m; e = m;
        end
        a = b; fa = fb;
        if abs(d) > tol, b = b+d; else, b = b+sign(m)*tol; end
        fb = f(b);
    end
    x = b;
end
