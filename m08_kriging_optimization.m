%% Kriging代理模型 + 螺旋桨优化框架
%  参考: Jones D.R. et al., J. Global Optimization, 1998 (EGO)
%        Sacks J. et al., Statistical Science, 1989 (Kriging)
%
%  流程: LHS采样 → 评估 → Kriging拟合 → EI自适应采样 → 优化
%  运行: 直接执行此脚本

clear; clc; close all;
rng(42);

%% ========== 设计空间 ==========
% 5个设计变量: c_root, r_cmax, c_max, twist_root, twist_tip
lb = [0.05, 0.30, 0.08, 15, 0];
ub = [0.15, 0.70, 0.18, 45, 15];
n_dims = 5;

%% ========== 第1步: LHS初始采样 ==========
n_init = 40;
X_samples = lhs_sample(n_init, lb, ub);
y_samples = zeros(n_init, 1);
for i = 1:n_init
    y_samples(i) = prop_objective(X_samples(i,:));
end
fprintf('Initial LHS: %d points, best η = %.4f\n', n_init, -min(y_samples));

%% ========== 第2步: Kriging训练 + EGO优化 ==========
n_infill = 15;
best_hist = zeros(n_init + n_infill, 1);
best_hist(1:n_init) = cummin(y_samples);

for it = 1:n_infill
    % 训练Kriging
    [theta, mu_k, sigma2_k, R_mat, R_inv, X_norm, y_norm, X_mean, X_std, y_mean, y_std] = ...
        kriging_train(X_samples, y_samples);

    % 最大化EI找下一个采样点
    y_best = min(y_samples);
    best_ei = -inf; x_new = X_samples(1,:);

    % 简单多起点搜索
    for trial = 1:200
        x_try = lb + (ub - lb) .* rand(1, n_dims);
        [~, ~, ei] = kriging_predict(x_try, X_samples, y_samples, ...
            theta, mu_k, sigma2_k, R_mat, R_inv, X_norm, y_norm, X_mean, X_std, y_mean, y_std, y_best);
        if ei > best_ei
            best_ei = ei;
            x_new = x_try;
        end
    end

    % 评估新点
    y_new = prop_objective(x_new);
    X_samples = [X_samples; x_new];
    y_samples = [y_samples; y_new];
    best_hist(n_init + it) = min(y_samples);

    fprintf('  Infill %2d/%d: η = %.4f, best = %.4f\n', ...
            it, n_infill, -y_new, -min(y_samples));
end

%% ========== 结果 ==========
[~, best_idx] = min(y_samples);
best_x = X_samples(best_idx, :);
fprintf('\n=== Optimization Complete ===\n');
fprintf('  Best η = %.4f\n', -y_samples(best_idx));
fprintf('  c_root=%.4f, r_cmax=%.4f, c_max=%.4f, tw_root=%.1f°, tw_tip=%.1f°\n', ...
        best_x(1), best_x(2), best_x(3), best_x(4), best_x(5));

%% ========== 可视化 ==========
figure('Position', [100 100 1400 500]);

subplot(1,3,1);
plot(1:length(best_hist), -best_hist, 'b-o', 'LineWidth', 2, 'MarkerSize', 3);
hold on; xline(n_init, 'r--', 'LineWidth', 1.5);
title('Optimization Convergence', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Evaluations'); ylabel('Best \eta'); grid on;

subplot(1,3,2);
% Kriging预测 vs 实际
[theta_f, mu_f, s2_f, R_f, Ri_f, Xn_f, yn_f, Xm_f, Xs_f, ym_f, ys_f] = ...
    kriging_train(X_samples, y_samples);
y_pred = zeros(size(y_samples));
for i = 1:length(y_samples)
    y_pred(i) = kriging_predict(X_samples(i,:), X_samples, y_samples, ...
        theta_f, mu_f, s2_f, R_f, Ri_f, Xn_f, yn_f, Xm_f, Xs_f, ym_f, ys_f, 0);
end
scatter(-y_samples, -y_pred, 40, 'filled', 'MarkerFaceAlpha', 0.6);
hold on; plot(xlim, xlim, 'r--', 'LineWidth', 2);
title('Kriging Fit Quality', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('True \eta'); ylabel('Predicted \eta'); grid on; axis equal;

subplot(1,3,3);
scatter(X_samples(1:n_init,1), X_samples(1:n_init,2), 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(X_samples(n_init+1:end,1), X_samples(n_init+1:end,2), 100, 'r', 'p', 'filled');
scatter(best_x(1), best_x(2), 200, 'gold', 'h', 'filled', 'MarkerEdgeColor', 'k');
legend('LHS', 'Infill', 'Optimum');
title('Sample Distribution', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('c_{root}'); ylabel('r_{cmax}'); grid on;

sgtitle('Kriging Surrogate Model Optimization', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数定义 ==========

function X = lhs_sample(n, lb, ub)
    d = length(lb);
    X = zeros(n, d);
    for j = 1:d
        perm = randperm(n);
        for i = 1:n
            lo = lb(j) + (ub(j)-lb(j))*(perm(i)-1)/n;
            hi = lb(j) + (ub(j)-lb(j))*perm(i)/n;
            X(i,j) = lo + (hi-lo)*rand;
        end
    end
end

function [theta,mu,s2,R_mat,R_inv,X_n,y_n,X_m,X_s,y_m,y_s] = kriging_train(X, y)
    [n, d] = size(X);
    X_m = mean(X); X_s = std(X) + 1e-10;
    y_m = mean(y); y_s = std(y) + 1e-10;
    X_n = (X - X_m) ./ X_s;
    y_n = (y - y_m) / y_s;

    % 优化theta
    best_nll = inf; theta = ones(1,d);
    for trial = 1:10
        lt0 = -2 + 5*rand(1,d);
        try
            opts = optimoptions('fminunc','Display','off','MaxIterations',100);
            [lt_opt, nll] = fminunc(@(lt) neg_loglik(lt, X_n, y_n), lt0, opts);
            if nll < best_nll
                best_nll = nll; theta = exp(lt_opt);
            end
        catch
        end
    end

    R_mat = corr_matrix(X_n, X_n, theta) + 1e-8*eye(n);
    R_inv = R_mat \ eye(n);
    ones_n = ones(n,1);
    mu = (ones_n'*R_inv*y_n) / (ones_n'*R_inv*ones_n);
    res = y_n - mu;
    s2 = (res'*R_inv*res) / n;
end

function [y_p, y_std, ei] = kriging_predict(x_new, X, y, theta, mu, s2, R_mat, R_inv, X_n, y_n, X_m, X_s, y_m, y_s, y_best)
    x_n = (x_new - X_m) ./ X_s;
    r = corr_matrix(x_n, X_n, theta);
    y_p_n = mu + r * R_inv * (y_n - mu);
    y_p = y_p_n * y_s + y_m;

    n = size(X,1);
    ones_n = ones(n,1);
    mse = s2 * (1 - r*R_inv*r' + (1-r*R_inv*ones_n)^2/(ones_n'*R_inv*ones_n));
    y_std = sqrt(max(mse,0)) * y_s;

    if y_std > 1e-8
        z = (y_best - y_p) / y_std;
        ei = (y_best - y_p) * normcdf(z) + y_std * normpdf(z);
    else
        ei = 0;
    end
end

function R = corr_matrix(X1, X2, theta)
    n1 = size(X1,1); n2 = size(X2,1);
    d = size(X1,2);
    R = zeros(n1, n2);
    for k = 1:d
        diff = X1(:,k) - X2(:,k)';
        R = R + theta(k) * diff.^2;
    end
    R = exp(-R);
end

function nll = neg_loglik(log_theta, X, y)
    theta = exp(log_theta);
    n = size(X,1);
    R = corr_matrix(X, X, theta) + 1e-8*eye(n);
    [~, flag] = chol(R);
    if flag > 0, nll = 1e10; return; end
    R_inv = R \ eye(n);
    ones_n = ones(n,1);
    mu = (ones_n'*R_inv*y) / (ones_n'*R_inv*ones_n);
    res = y - mu;
    s2 = (res'*R_inv*res) / n;
    nll = 0.5*n*log(max(s2,1e-10)) + 0.5*log(det(R));
    if ~isfinite(nll), nll = 1e10; end
end

function val = prop_objective(params)
    c_root = params(1); r_cmax = params(2); c_max = params(3);
    twist_root = params(4); twist_tip = params(5);
    R = 0.15; n_sec = 20; B = 3; V_inf = 15; omega = 8000*2*pi/60; rho = 1.225;
    r_R = linspace(0.2, 0.98, n_sec);
    chord = zeros(1, n_sec);
    for i = 1:n_sec
        r = r_R(i);
        if r < r_cmax
            chord(i) = c_root + (c_max-c_root)*((r-0.2)/(r_cmax-0.2+1e-10))^0.8;
        else
            chord(i) = c_max*((0.98-r)/(0.98-r_cmax+1e-10))^1.2;
        end
    end
    chord = max(min(chord*R, 0.03), 0.005);
    twist = linspace(twist_root, twist_tip, n_sec);

    T_t = 0; Q_t = 0;
    for i = 1:n_sec
        r = r_R(i)*R; Vy = omega*r;
        phi = atan2(V_inf, Vy);
        alpha = deg2rad(twist(i)) - phi + deg2rad(5);
        alpha = max(min(alpha, deg2rad(12)), deg2rad(-5));
        Cl = 2*pi*alpha; Cd = 0.008+0.005*Cl^2;
        W = sqrt(V_inf^2 + Vy^2);
        Cn = Cl*cos(phi)+Cd*sin(phi); Ct = Cl*sin(phi)-Cd*cos(phi);
        dT = 0.5*rho*W^2*chord(i)*Cn*B;
        dQ = 0.5*rho*W^2*chord(i)*Ct*B*r;
        T_t = T_t + dT*(R*0.78/n_sec);
        Q_t = Q_t + dQ*(R*0.78/n_sec);
    end
    P_t = Q_t*omega;
    eta = T_t*V_inf/(P_t+1e-10);
    eta = max(min(eta,1),0);
    val = -eta;
end
