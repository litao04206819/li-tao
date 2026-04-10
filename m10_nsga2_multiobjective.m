%% NSGA-II 多目标优化 — 效率 vs 噪声
%  参考: Deb K. et al., IEEE Trans. Evol. Computation, 2002.
%  运行: 直接执行此脚本

clear; clc; close all;
rng(42);

%% ========== 设计空间 ==========
lb = [0.06, 0.05, 0.03, 20, 8, 2];
ub = [0.18, 0.16, 0.12, 45, 25, 15];
n_dims = 6;

%% ========== NSGA-II ==========
pop_size = 60; n_gen = 80;
pop = repmat(lb, pop_size, 1) + rand(pop_size, n_dims) .* repmat(ub-lb, pop_size, 1);
obj = zeros(pop_size, 2);
for i = 1:pop_size, obj(i,:) = multi_obj(pop(i,:)); end

fprintf('Running NSGA-II (%d gen, pop=%d)...\n', n_gen, pop_size);
history = zeros(n_gen, 1);

for gen = 1:n_gen
    % 生成子代
    offspring = zeros(pop_size, n_dims);
    for i = 1:2:pop_size
        p1 = tournament(pop, obj); p2 = tournament(pop, obj);
        [c1, c2] = sbx_cross(p1, p2, lb, ub);
        offspring(i,:) = poly_mut(c1, lb, ub, 0.15);
        if i+1 <= pop_size, offspring(i+1,:) = poly_mut(c2, lb, ub, 0.15); end
    end
    obj_off = zeros(pop_size, 2);
    for i = 1:pop_size, obj_off(i,:) = multi_obj(offspring(i,:)); end

    % 合并 + 非支配排序 + 截断
    comb_pop = [pop; offspring]; comb_obj = [obj; obj_off];
    [pop, obj] = nsga2_select(comb_pop, comb_obj, pop_size);

    front0 = find_front0(obj);
    history(gen) = length(front0);
    if mod(gen,20)==0, fprintf('  Gen %d: Pareto size = %d\n', gen, length(front0)); end
end

%% ========== 结果 ==========
front0 = find_front0(obj);
pf = obj(front0,:); ps = pop(front0,:);
fprintf('\nPareto front: %d solutions\n', length(front0));
fprintf('η: [%.4f, %.4f], SPL: [%.1f, %.1f] dB\n', ...
        -max(pf(:,1)), -min(pf(:,1)), min(pf(:,2)), max(pf(:,2)));

%% ========== 可视化 ==========
figure('Position', [50 50 1400 500]);

subplot(1,3,1);
scatter(-pf(:,1), pf(:,2), 50, 'filled', 'MarkerFaceAlpha', 0.7); hold on;
[~, ib] = min(pf(:,1)); scatter(-pf(ib,1), pf(ib,2), 200, 'r', 'p', 'filled');
[~, in] = min(pf(:,2)); scatter(-pf(in,1), pf(in,2), 200, 'g', 's', 'filled');
xlabel('Efficiency \eta'); ylabel('Noise SPL (dB)');
title('Pareto Front', 'FontSize', 13, 'FontWeight', 'bold');
legend('Pareto', 'Best \eta', 'Lowest SPL'); grid on;

subplot(1,3,2);
plot(1:n_gen, history, 'b-', 'LineWidth', 2);
title('Pareto Front Size', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Generation'); ylabel('# Solutions'); grid on;

subplot(1,3,3);
% 平行坐标
pn = (ps - lb) ./ (ub - lb);
eta_v = -pf(:,1);
eta_norm = (eta_v - min(eta_v)) / (max(eta_v)-min(eta_v)+1e-10);
colors_cm = cool(256);
for i = 1:size(pn,1)
    ci = max(1, min(256, round(eta_norm(i)*255)+1));
    plot(1:6, pn(i,:), '-', 'Color', [colors_cm(ci,:) 0.5], 'LineWidth', 1.5); hold on;
end
set(gca, 'XTick', 1:6, 'XTickLabel', {'c30','c60','c90','tw30','tw60','tw90'});
ylabel('Normalized'); grid on;
title('Parallel Coordinates', 'FontSize', 13, 'FontWeight', 'bold');

sgtitle('NSGA-II Multi-Objective Optimization', 'FontSize', 15, 'FontWeight', 'bold');

%% ========== 函数 ==========
function f = multi_obj(x)
    [~,~,eta] = simple_bem(x(1:3), x(4:6));
    spl = simple_noise(x(1:3), x(4:6));
    f = [-eta, spl];
end

function [T,P,eta] = simple_bem(cc, tc)
    R=0.15; B=3; V=15; omega=8000*2*pi/60; rho=1.225;
    r_c = [0.15,0.3,0.6,0.9,0.98];
    c_c = [cc(1)*0.6, cc(1), cc(2), cc(3), cc(3)*0.3];
    t_c = [tc(1)*1.3, tc(1), tc(2), tc(3), tc(3)*0.5];
    rR = linspace(0.2,0.97,30);
    chord = max(min(spline(r_c,c_c,rR),0.20),0.03)*R;
    twist = spline(r_c,t_c,rR);
    T=0; Q=0;
    for i=1:30
        r=rR(i)*R; Vy=omega*r; W=sqrt(V^2+Vy^2); phi=atan2(V,Vy);
        al = deg2rad(twist(i))-phi; al=max(min(al,deg2rad(10)),deg2rad(-3));
        Cl=5.5*al; Cd=0.01+0.04*al^2;
        Cn=Cl*cos(phi)+Cd*sin(phi); Ct=Cl*sin(phi)-Cd*cos(phi);
        f_t=B*(1-rR(i))/(2*rR(i)*abs(sin(phi))+1e-6);
        F=(2/pi)*acos(max(min(exp(-f_t),1),-1));
        dT=0.5*rho*W^2*chord(i)*Cn*B*F; dQ=0.5*rho*W^2*chord(i)*Ct*B*r*F;
        dr=0.77*R/30; T=T+dT*dr; Q=Q+dQ*dr;
    end
    P=Q*omega; eta=max(min(T*V/(abs(P)+1e-6),0.95),0);
end

function spl = simple_noise(cc, tc)
    R=0.15; B=3; omega=8000*2*pi/60; V=15; rho=1.225; c0=343;
    r_c=[0.15,0.3,0.6,0.9,0.98]; c_c=[cc(1)*0.6,cc(1),cc(2),cc(3),cc(3)*0.3];
    rR=linspace(0.2,0.97,30); chord=max(min(spline(r_c,c_c,rR),0.20),0.03)*R;
    p2=0;
    for i=1:30
        r=rR(i)*R; Vl=sqrt(V^2+(omega*r)^2); Re=rho*Vl*chord(i)/1.8e-5;
        ds=chord(i)/sqrt(Re+1e-6); pw=rho*Vl^5*chord(i)*ds/c0^2;
        p2=p2+pw;
    end
    Mt=omega*R/c0; spl_te=10*log10(max(p2,1e-20)/1e-12);
    spl_tip=50*log10(Mt)+60;
    spl=10*log10(10^(spl_te/10)+10^(spl_tip/10))+10*log10(B);
end

function p = tournament(pop, obj)
    idx = randperm(size(pop,1), 3);
    [~, best] = min(obj(idx,1));
    p = pop(idx(best),:);
end

function [c1,c2] = sbx_cross(p1,p2,lb,ub)
    c1=p1; c2=p2; eta_c=20;
    for i=1:length(p1)
        if rand<0.5 && abs(p1(i)-p2(i))>1e-10
            u=rand;
            if u<=0.5, beta=(2*u)^(1/(eta_c+1)); else, beta=(1/(2*(1-u)))^(1/(eta_c+1)); end
            c1(i)=0.5*((1+beta)*p1(i)+(1-beta)*p2(i));
            c2(i)=0.5*((1-beta)*p1(i)+(1+beta)*p2(i));
            c1(i)=max(min(c1(i),ub(i)),lb(i));
            c2(i)=max(min(c2(i),ub(i)),lb(i));
        end
    end
end

function x = poly_mut(x, lb, ub, pm)
    eta_m=20;
    for i=1:length(x)
        if rand<pm
            u=rand;
            if u<0.5, d=(2*u)^(1/(eta_m+1))-1; else, d=1-(2*(1-u))^(1/(eta_m+1)); end
            x(i)=x(i)+d*(ub(i)-lb(i));
            x(i)=max(min(x(i),ub(i)),lb(i));
        end
    end
end

function [new_pop, new_obj] = nsga2_select(pop, obj, n)
    fronts = find_all_fronts(obj);
    new_pop = zeros(n, size(pop,2)); new_obj = zeros(n, size(obj,2));
    cnt = 0;
    for f = 1:length(fronts)
        fr = fronts{f};
        if cnt + length(fr) <= n
            for k = 1:length(fr)
                cnt=cnt+1; new_pop(cnt,:)=pop(fr(k),:); new_obj(cnt,:)=obj(fr(k),:);
            end
        else
            rem = n - cnt;
            cd = crowd_dist(obj(fr,:));
            [~, si] = sort(cd, 'descend');
            for k = 1:rem
                cnt=cnt+1; new_pop(cnt,:)=pop(fr(si(k)),:); new_obj(cnt,:)=obj(fr(si(k)),:);
            end
            break;
        end
    end
end

function fronts = find_all_fronts(obj)
    n=size(obj,1); dc=zeros(n,1); ds=cell(n,1);
    for i=1:n, ds{i}=[]; end
    front1=[];
    for i=1:n
        for j=i+1:n
            if all(obj(i,:)<=obj(j,:)) && any(obj(i,:)<obj(j,:))
                ds{i}=[ds{i},j]; dc(j)=dc(j)+1;
            elseif all(obj(j,:)<=obj(i,:)) && any(obj(j,:)<obj(i,:))
                ds{j}=[ds{j},i]; dc(i)=dc(i)+1;
            end
        end
        if dc(i)==0, front1=[front1,i]; end
    end
    fronts={front1}; k=1;
    while ~isempty(fronts{k})
        nf=[];
        for i=fronts{k}
            for j=ds{i}
                dc(j)=dc(j)-1;
                if dc(j)==0, nf=[nf,j]; end
            end
        end
        k=k+1; fronts{k}=nf;
    end
    fronts=fronts(~cellfun('isempty',fronts));
end

function idx = find_front0(obj)
    n=size(obj,1); dominated=false(n,1);
    for i=1:n
        for j=1:n
            if i~=j && all(obj(j,:)<=obj(i,:)) && any(obj(j,:)<obj(i,:))
                dominated(i)=true; break;
            end
        end
    end
    idx=find(~dominated);
end

function d = crowd_dist(obj)
    [n,m]=size(obj); d=zeros(n,1);
    if n<=2, d=inf(n,1); return; end
    for k=1:m
        [~,si]=sort(obj(:,k));
        d(si(1))=inf; d(si(end))=inf;
        rng_k=obj(si(end),k)-obj(si(1),k);
        if rng_k<1e-10, continue; end
        for i=2:n-1
            d(si(i))=d(si(i))+(obj(si(i+1),k)-obj(si(i-1),k))/rng_k;
        end
    end
end
