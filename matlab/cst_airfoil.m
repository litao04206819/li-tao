%% CST Airfoil Parameterization
% CST (Class function/Shape function Transformation) method for airfoils.
%
% Based on Kulfan's CST method:
%   Kulfan, B.M., "Universal Parametric Geometry Representation Method",
%   Journal of Aircraft, Vol. 45, No. 1, 2008, pp. 142-158.
%
% The airfoil surface is represented as:
%   y(x) = C(x) * S(x) + x * dz_te
% where:
%   C(x) = x^N1 * (1-x)^N2          (class function)
%   S(x) = sum( A_i * K_i * x^i * (1-x)^(n-i) )  (shape function)

function cst_airfoil()
    %% Main script
    clc; close all;

    % CST coefficients (example: NACA 0012-like)
    A_upper = [0.17, 0.15, 0.16, 0.14, 0.12, 0.10, 0.08];
    A_lower = [-0.17, -0.15, -0.16, -0.14, -0.12, -0.10, -0.08];
    dz_te = 0.001;  % half trailing edge gap

    % Cosine-spaced x coordinates for better LE resolution
    n_pts = 201;
    theta = linspace(0, pi, n_pts);
    x = 0.5 * (1 - cos(theta));

    % Generate airfoil
    [y_upper, y_lower] = compute_cst_airfoil(x, A_upper, A_lower, dz_te);

    % Display info
    fprintf('=== CST Airfoil Parameterization ===\n');
    fprintf('Upper surface coefficients: [');
    fprintf('%.2f ', A_upper);
    fprintf(']\n');
    fprintf('Lower surface coefficients: [');
    fprintf('%.2f ', A_lower);
    fprintf(']\n');
    fprintf('Trailing edge gap: %.4f\n', 2 * dz_te);
    fprintf('Number of points: %d\n', n_pts);
    fprintf('Max thickness: %.6f\n', max(y_upper - y_lower));

    % Plot airfoil
    plot_airfoil(x, y_upper, y_lower, 'CST Parameterized Airfoil');

    % Demonstrate fitting
    order = 6;
    [A_upper_fit, A_lower_fit] = fit_cst(x, y_upper, y_lower, order, dz_te);
    fprintf('\nFitted upper coefficients: [');
    fprintf('%.6f ', A_upper_fit);
    fprintf(']\n');
    fprintf('Fitted lower coefficients: [');
    fprintf('%.6f ', A_lower_fit);
    fprintf(']\n');

    [y_upper_fit, y_lower_fit] = compute_cst_airfoil(x, A_upper_fit, A_lower_fit, dz_te);
    err = max(abs(y_upper - y_upper_fit)) + max(abs(y_lower - y_lower_fit));
    fprintf('Max fitting error: %.2e\n', err);
end

%% Class function
function C = class_func(x, N1, N2)
    if nargin < 2, N1 = 0.5; end
    if nargin < 3, N2 = 1.0; end
    C = x.^N1 .* (1 - x).^N2;
end

%% Shape function using Bernstein polynomials
function S = shape_func(x, A)
    n = length(A) - 1;
    S = zeros(size(x));
    for i = 0:n
        K = nchoosek(n, i);
        B = K .* x.^i .* (1 - x).^(n - i);
        S = S + A(i + 1) * B;
    end
end

%% Compute CST airfoil coordinates
function [y_upper, y_lower] = compute_cst_airfoil(x, A_upper, A_lower, dz_te, N1, N2)
    if nargin < 4, dz_te = 0.0; end
    if nargin < 5, N1 = 0.5; end
    if nargin < 6, N2 = 1.0; end

    C = class_func(x, N1, N2);
    S_upper = shape_func(x, A_upper);
    S_lower = shape_func(x, A_lower);

    y_upper = C .* S_upper + x * dz_te;
    y_lower = C .* S_lower - x * dz_te;
end

%% Fit CST coefficients to airfoil data
function [A_upper, A_lower] = fit_cst(x, y_upper, y_lower, order, dz_te, N1, N2)
    if nargin < 5, dz_te = 0.0; end
    if nargin < 6, N1 = 0.5; end
    if nargin < 7, N2 = 1.0; end

    n = order;
    C = class_func(x, N1, N2);

    % Build Bernstein basis matrix
    B = zeros(length(x), n + 1);
    for i = 0:n
        K = nchoosek(n, i);
        B(:, i + 1) = K .* x(:).^i .* (1 - x(:)).^(n - i);
    end

    % Weighted matrix
    W = diag(C(:)) * B;

    % Upper surface
    rhs_upper = (y_upper(:) - x(:) * dz_te);
    A_upper = (W \ rhs_upper)';

    % Lower surface
    rhs_lower = (y_lower(:) + x(:) * dz_te);
    A_lower = (W \ rhs_lower)';
end

%% Plot airfoil
function plot_airfoil(x, y_upper, y_lower, fig_title)
    figure('Position', [100 100 800 400]);
    hold on;
    plot(x, y_upper, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Upper surface');
    plot(x, y_lower, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Lower surface');
    xlabel('x/c');
    ylabel('y/c');
    title(fig_title);
    axis equal;
    legend('Location', 'best');
    grid on;
    hold off;
end
