%% NACA 翼型族生成器 + 坐标工具
%  参考: Abbott & von Doenhoff, "Theory of Wing Sections," Dover, 1959.
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 演示 ==========
figure('Position', [50 50 1400 900]);

% --- 示例1: NACA 4位系列 ---
subplot(2,3,1);
names4 = {'0012','2412','4412','6412'};
for k = 1:4
    [xu,yu,xl,yl] = naca4_gen(names4{k});
    plot(xu, yu, '-', 'LineWidth', 2, 'DisplayName', ['NACA ' names4{k}]); hold on;
    plot(xl, yl, '-', 'LineWidth', 2, 'Color', gca().Children(1).Color, 'HandleVisibility','off');
end
axis equal; grid on; legend; xlabel('x/c'); ylabel('y/c');
title('NACA 4-Digit Series', 'FontSize', 13, 'FontWeight', 'bold');

% --- 示例2: NACA 5位系列 ---
subplot(2,3,2);
names5 = {'23012','23015','23018','23021'};
for k = 1:4
    [xu,yu,xl,yl] = naca5_gen(names5{k});
    plot(xu, yu, '-', 'LineWidth', 2, 'DisplayName', ['NACA ' names5{k}]); hold on;
    plot(xl, yl, '-', 'LineWidth', 2, 'Color', gca().Children(1).Color, 'HandleVisibility','off');
end
axis equal; grid on; legend; xlabel('x/c'); ylabel('y/c');
title('NACA 5-Digit Series', 'FontSize', 13, 'FontWeight', 'bold');

% --- 示例3: 弯度线对比 ---
subplot(2,3,3);
for k = 1:4
    [~,~,~,~,xc,yc] = naca4_gen(names4{k});
    plot(xc, yc, '-', 'LineWidth', 2.5, 'DisplayName', ['NACA ' names4{k}]); hold on;
end
grid on; legend; xlabel('x/c'); ylabel('y_c/c');
title('Camber Lines', 'FontSize', 13, 'FontWeight', 'bold');

% --- 示例4: 几何分析 ---
subplot(2,3,4);
an = {'0012','2412','4415','6418'};
tc = zeros(4,1); cam = zeros(4,1); rle = zeros(4,1);
for k = 1:4
    [xu,yu,xl,yl] = naca4_gen(an{k});
    info = analyze_af(xu, yu, xl, yl);
    tc(k) = info.max_t*100; cam(k) = info.max_c*100; rle(k) = info.r_le*100;
end
bar([tc, cam, rle]);
set(gca, 'XTickLabel', cellfun(@(s) ['NACA ' s], an, 'UniformOutput', false));
legend('t/c %', 'camber %', 'r_{LE}×100');
title('Geometric Analysis', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('% chord'); grid on;

% --- 示例5: CST逆向拟合 ---
subplot(2,3,5);
[xu0,yu0,xl0,yl0] = naca4_gen('2412', 201);
plot(xu0, yu0, 'k-', 'LineWidth', 3, 'Color', [0 0 0 0.3], 'DisplayName', 'Original'); hold on;
plot(xl0, yl0, 'k-', 'LineWidth', 3, 'Color', [0 0 0 0.3], 'HandleVisibility', 'off');

for order = [4, 6, 8]
    A_up = fit_cst_af(xu0, yu0, order);
    A_lo = fit_cst_af(xl0, yl0, order);
    psi = linspace(0, 1, 201);
    C = psi.^0.5 .* (1-psi).^1.0;
    S_up = zeros(size(psi)); S_lo = zeros(size(psi));
    for k = 0:order
        bk = nchoosek(order,k) * psi.^k .* (1-psi).^(order-k);
        S_up = S_up + A_up(k+1)*bk;
        S_lo = S_lo + A_lo(k+1)*bk;
    end
    plot(psi, C.*S_up, '--', 'LineWidth', 2, 'DisplayName', sprintf('CST n=%d', order));
    plot(psi, C.*S_lo, '--', 'LineWidth', 2, 'Color', gca().Children(1).Color, 'HandleVisibility','off');
end
axis equal; grid on; legend('FontSize', 8);
title('CST Inverse Fitting NACA 2412', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

% --- 示例6: 余弦 vs 均匀采样 ---
subplot(2,3,6);
[xu_c,yu_c] = naca4_gen('4412', 31, true);
[xu_u,yu_u] = naca4_gen('4412', 31, false);
[xu_r,yu_r] = naca4_gen('4412', 201);
plot(xu_r, yu_r, 'k-', 'LineWidth', 1, 'Color', [0 0 0 0.3]); hold on;
plot(xu_c, yu_c, 'bo-', 'MarkerSize', 5, 'LineWidth', 1.5, 'DisplayName', 'Cosine (31)');
plot(xu_u, yu_u, 'rs-', 'MarkerSize', 5, 'LineWidth', 1.5, 'DisplayName', 'Uniform (31)');
axis equal; grid on; legend;
title('Cosine vs Uniform Spacing', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x/c'); ylabel('y/c');

sgtitle('NACA Airfoil Tools', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数 ==========

function [xu,yu,xl,yl,xc_out,yc_out] = naca4_gen(code, n, cosine)
    if nargin<2, n=101; end
    if nargin<3, cosine=true; end
    m_v = str2double(code(1))/100;
    p_v = str2double(code(2))/10;
    t_v = str2double(code(3:4))/100;
    if cosine
        beta = linspace(0, pi, n);
        x = 0.5*(1-cos(beta));
    else
        x = linspace(0, 1, n);
    end
    yt = 5*t_v*(0.2969*sqrt(x)-0.1260*x-0.3516*x.^2+0.2843*x.^3-0.1015*x.^4);
    yc = zeros(size(x)); dyc = zeros(size(x));
    if m_v > 0 && p_v > 0
        fr = x<=p_v; rr = x>p_v;
        yc(fr) = m_v/p_v^2*(2*p_v*x(fr)-x(fr).^2);
        yc(rr) = m_v/(1-p_v)^2*((1-2*p_v)+2*p_v*x(rr)-x(rr).^2);
        dyc(fr) = 2*m_v/p_v^2*(p_v-x(fr));
        dyc(rr) = 2*m_v/(1-p_v)^2*(p_v-x(rr));
    end
    th = atan(dyc);
    xu = x - yt.*sin(th); yu = yc + yt.*cos(th);
    xl = x + yt.*sin(th); yl = yc - yt.*cos(th);
    xc_out = x; yc_out = yc;
end

function [xu,yu,xl,yl,xc_out,yc_out] = naca5_gen(code, n)
    if nargin<2, n=101; end
    p_v = str2double(code(2:3))/200;
    t_v = str2double(code(4:5))/100;
    beta = linspace(0,pi,n); x = 0.5*(1-cos(beta));
    yt = 5*t_v*(0.2969*sqrt(x)-0.1260*x-0.3516*x.^2+0.2843*x.^3-0.1015*x.^4);
    k1_map = containers.Map({210,220,230,240,250},...
        {[0.058,361.4],[0.126,51.64],[0.2025,15.957],[0.29,6.643],[0.391,3.23]});
    key = str2double(code(1:3));
    if isKey(k1_map, key), v=k1_map(key); r_v=v(1); k1=v(2);
    else, r_v=0.2025; k1=15.957; end
    yc=zeros(size(x)); dyc=zeros(size(x));
    fr=x<=r_v; rr=x>r_v;
    yc(fr)=k1/6*(x(fr).^3-3*r_v*x(fr).^2+r_v^2*(3-r_v)*x(fr));
    yc(rr)=k1*r_v^3/6*(1-x(rr));
    dyc(fr)=k1/6*(3*x(fr).^2-6*r_v*x(fr)+r_v^2*(3-r_v));
    dyc(rr)=-k1*r_v^3/6;
    th=atan(dyc);
    xu=x-yt.*sin(th); yu=yc+yt.*cos(th);
    xl=x+yt.*sin(th); yl=yc-yt.*cos(th);
    xc_out=x; yc_out=yc;
end

function info = analyze_af(xu, yu, xl, yl)
    xc = linspace(0.001, 1, 500);
    yui = interp1(xu, yu, xc); yli = interp1(xl, yl, xc);
    thick = yui - yli;
    [info.max_t, idx_t] = max(thick); info.max_t_pos = xc(idx_t);
    cam = 0.5*(yui+yli);
    [info.max_c, idx_c] = max(abs(cam)); info.max_c_pos = xc(idx_c);
    n_le = min(20, floor(length(xu)/4));
    if n_le > 3
        cf = polyfit(xu(2:n_le), yu(2:n_le).^2, 1);
        info.r_le = abs(cf(1)/2);
    else
        info.r_le = 0;
    end
end

function A = fit_cst_af(x_data, y_data, order)
    mask = x_data > 1e-4 & x_data < 0.999;
    x = x_data(mask); y = y_data(mask);
    C = x(:).^0.5 .* (1-x(:)).^1.0;
    B = zeros(length(x), order+1);
    for k = 0:order
        B(:,k+1) = C .* nchoosek(order,k) .* x(:).^k .* (1-x(:)).^(order-k);
    end
    A = B \ y(:);
end
