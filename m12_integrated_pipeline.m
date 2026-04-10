%% 完整集成流水线 — NACA翼型 → CST → 展向分布 → BEM → 三维桨叶
%  运行: 直接执行此脚本

clear; clc; close all;

%% ========== 设计参数 ==========
R = 0.15; Rhub = 0.02; B = 3;
V_inf = 15; RPM = 8000;
omega = RPM*2*pi/60; rho = 1.225;

fprintf('========== PROPELLER DESIGN PIPELINE ==========\n');

%% Step 1: NACA翼型 → CST拟合
fprintf('[01] Step 1: NACA 4412 → CST fit\n');
x_af = linspace(0.001, 0.999, 200);
m=0.04; p=0.4; t=0.12;
yt = 5*t*(0.2969*sqrt(x_af)-0.1260*x_af-0.3516*x_af.^2+0.2843*x_af.^3-0.1015*x_af.^4);
yc = zeros(size(x_af)); fr=x_af<=p; rr=x_af>p;
yc(fr)=m/p^2*(2*p*x_af(fr)-x_af(fr).^2);
yc(rr)=m/(1-p)^2*((1-2*p)+2*p*x_af(rr)-x_af(rr).^2);
y_up = yc + yt; y_lo = yc - yt;

order = 6;
A_up = cst_fit(x_af, y_up, order);
A_lo = cst_fit(x_af, y_lo, order);
fprintf('  CST coefficients fitted (order=%d)\n', order);

%% Step 2: 展向分布
fprintf('[02] Step 2: Spanwise distributions\n');
n_sec = 30;
r_R = linspace(0.20, 0.97, n_sec);
r = r_R * R;
phi_ideal = atan2(V_inf, omega*r);
twist = rad2deg(phi_ideal + deg2rad(5));
phi_tip = atan2(V_inf, omega*R);
F_pr = (2/pi)*acos(max(min(exp(-B*(1-r_R)./(2*r_R*abs(sin(phi_tip))+1e-6)),1),-1));
sigma_opt = 8*pi*r_R.*F_pr.*sin(phi_ideal)/(B*0.6) .* ...
    abs(cos(phi_ideal)-r_R*omega*R/V_inf.*sin(phi_ideal)) ./ ...
    (sin(phi_ideal)+r_R*omega*R/V_inf.*cos(phi_ideal)+1e-6);
chord = max(min(abs(sigma_opt).*2*pi.*r/B, 0.035), 0.005);
fprintf('  Chord: [%.1f, %.1f] mm, Twist: [%.1f, %.1f] deg\n', ...
    min(chord)*1e3, max(chord)*1e3, min(twist), max(twist));

%% Step 3: BEM分析
fprintf('[03] Step 3: BEM analysis\n');
dT = zeros(n_sec,1); dQ = zeros(n_sec,1);
for i = 1:n_sec
    Vy=omega*r(i); W=sqrt(V_inf^2+Vy^2); phi=atan2(V_inf,Vy);
    al=deg2rad(twist(i))-phi; al=max(min(al,deg2rad(10)),deg2rad(-3));
    Cl=2*pi*al*0.9; Cd=0.008+0.004*Cl^2;
    Cn=Cl*cos(phi)+Cd*sin(phi); Ct=Cl*sin(phi)-Cd*cos(phi);
    f_t=B*(1-r_R(i))/(2*r_R(i)*abs(sin(phi))+1e-6);
    Fv=(2/pi)*acos(max(min(exp(-f_t),1),-1));
    dT(i)=0.5*rho*W^2*chord(i)*Cn*B*Fv;
    dQ(i)=0.5*rho*W^2*chord(i)*Ct*B*r(i)*Fv;
end
T=trapz(r, dT); Q=trapz(r, dQ); P=Q*omega;
n_rev=RPM/60; D=2*R;
CT=T/(rho*n_rev^2*D^4); CP=P/(rho*n_rev^3*D^5);
J=V_inf/(n_rev*D); eta=max(J*CT/(CP+1e-10),0);
fprintf('  T=%.3fN, Q=%.5fN·m, η=%.4f\n', T, Q, eta);

%% Step 4: Hicks-Henne微调
fprintf('[04] Step 4: Hicks-Henne refinement\n');
psi = linspace(0,1,101);
C_fn = psi.^0.5.*(1-psi).^1.0;
S_up = zeros(size(psi)); S_lo = zeros(size(psi));
for k = 0:order
    bk = nchoosek(order,k)*psi.^k.*(1-psi).^(order-k);
    S_up = S_up + A_up(k+1)*bk;
    S_lo = S_lo + A_lo(k+1)*bk;
end
yu_cst = C_fn.*S_up + psi*0.001;
yl_cst = C_fn.*S_lo - psi*0.001;

% 凸包微调
bumps = [0.12, 0.003; 0.35, 0.002; 0.75, -0.002];
yu_ref = yu_cst;
for k = 1:size(bumps,1)
    ps = max(min(psi, 1-1e-10), 1e-10);
    hh = sin(pi*ps.^(log(0.5)/log(bumps(k,1)))).^3;
    yu_ref = yu_ref + bumps(k,2)*hh;
end
fprintf('  Applied %d bumps\n', size(bumps,1));

%% Step 5: 三维桨叶
fprintf('[05] Step 5: 3D blade geometry\n');
n_span = 25; n_ch = 51;
r_sp = linspace(r_R(1), r_R(end), n_span);
chord_sp = interp1(r_R, chord, r_sp, 'spline');
twist_sp = interp1(r_R, twist, r_sp, 'spline');

psi2 = linspace(0,1,n_ch);
C2 = psi2.^0.5.*(1-psi2).^1.0;
Su=zeros(size(psi2)); Sl=zeros(size(psi2));
for k=0:order
    bk=nchoosek(order,k)*psi2.^k.*(1-psi2).^(order-k);
    Su=Su+A_up(k+1)*bk; Sl=Sl+A_lo(k+1)*bk;
end
yu2=C2.*Su+psi2*0.001; yl2=C2.*Sl-psi2*0.001;
x_sec=[psi2, fliplr(psi2(2:end-1))];
y_sec=[yu2, fliplr(yl2(2:end-1))];
nt=length(x_sec);

Xs=zeros(n_span,nt); Ys=zeros(n_span,nt); Zs=zeros(n_span,nt);
for i=1:n_span
    rv=r_sp(i)*R; c=chord_sp(i); tw=deg2rad(twist_sp(i));
    xs=x_sec*c-0.25*c; ys=y_sec*c;
    xr=xs*cos(tw)-ys*sin(tw); yr=xs*sin(tw)+ys*cos(tw);
    Xs(i,:)=yr; Ys(i,:)=rv; Zs(i,:)=xr;
end
fprintf('  Surface: %d×%d = %d pts\n', n_span, nt, n_span*nt);

%% ========== 可视化 ==========
figure('Position', [50 50 1600 1000]);

subplot(2,3,1);
plot(x_af, y_up, 'k-', 'LineWidth', 1.5, 'Color', [0 0 0 0.3]); hold on;
plot(x_af, y_lo, 'k-', 'LineWidth', 1.5, 'Color', [0 0 0 0.3]);
plot(psi, yu_cst, 'b--', 'LineWidth', 2);
plot(psi, yl_cst, 'b--', 'LineWidth', 2);
plot(psi, yu_ref, 'r-', 'LineWidth', 2);
legend('NACA','','CST','','HH refined'); axis equal; grid on;
title('Step 1+4: Airfoil', 'FontSize', 12, 'FontWeight', 'bold');

subplot(2,3,2);
yyaxis left; plot(r_R, chord*1e3, 'b-', 'LineWidth', 2.5); ylabel('Chord (mm)');
yyaxis right; plot(r_R, twist, 'r-', 'LineWidth', 2.5); ylabel('Twist (deg)');
title('Step 2: Distributions', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('r/R'); grid on;

subplot(2,3,3);
yyaxis left; plot(r_R, dT, 'b-', 'LineWidth', 2.5); ylabel('dT/dr (N/m)');
yyaxis right; plot(r_R, dQ, 'r-', 'LineWidth', 2.5); ylabel('dQ/dr');
title(sprintf('Step 3: BEM (η=%.4f)', eta), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('r/R'); grid on;

subplot(2,3,4);
surf(Zs*1e3, Ys*1e3, Xs*1e3, 'FaceColor','interp','EdgeColor',[0.5 0.5 0.5],'EdgeAlpha',0.2);
colormap(cool); xlabel('Z'); ylabel('r'); zlabel('X');
title('Step 5: Single Blade', 'FontSize', 12, 'FontWeight', 'bold');
axis equal; grid on; view(-60,25); light;

subplot(2,3,5);
hold on;
for b=1:B
    ang=2*pi*(b-1)/B;
    Yr=Ys*cos(ang)-Zs*sin(ang); Zr=Ys*sin(ang)+Zs*cos(ang);
    surf(Zr*1e3, Yr*1e3, Xs*1e3, 'FaceAlpha',0.7,'EdgeColor','none');
end
title(sprintf('Step 5: %d-Blade', B), 'FontSize', 12, 'FontWeight', 'bold');
axis equal; grid on; view(-45,30); light; lighting gouraud;

subplot(2,3,6); axis off;
txt = sprintf(['DESIGN SUMMARY\n' ...
    '───────────────────────\n' ...
    'R = %.0f mm, Hub = %.0f mm, B = %d\n' ...
    'V = %.0f m/s, RPM = %d\n' ...
    'Airfoil: NACA 4412\n' ...
    '───────────────────────\n' ...
    'T = %.3f N\n' ...
    'Q = %.5f N·m\n' ...
    'P = %.2f W\n' ...
    'CT = %.4f, CP = %.4f\n' ...
    'J = %.3f, η = %.4f\n' ...
    '───────────────────────\n' ...
    'Sections: %d\n' ...
    'Surface points: %d'], ...
    R*1e3, Rhub*1e3, B, V_inf, RPM, T, Q, P, CT, CP, J, eta, n_span, n_span*nt);
text(0.1, 0.9, txt, 'FontSize', 10, 'FontName', 'FixedWidth', ...
     'VerticalAlignment', 'top', 'BackgroundColor', [1 1 0.9]);

sgtitle('Integrated Propeller Design Pipeline', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数 ==========
function A = cst_fit(x_d, y_d, n_ord)
    mask = x_d>1e-4 & x_d<0.999;
    x=x_d(mask); y=y_d(mask);
    C=x(:).^0.5.*(1-x(:)).^1.0;
    B_m=zeros(length(x),n_ord+1);
    for k=0:n_ord
        B_m(:,k+1)=C.*nchoosek(n_ord,k).*x(:).^k.*(1-x(:)).^(n_ord-k);
    end
    A=B_m\y(:);
end
