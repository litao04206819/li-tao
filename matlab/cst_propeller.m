%% CST Propeller Blade Parameterization
% Extends the 2D CST airfoil method to 3D propeller blade geometry.
% CST profiles are applied at multiple radial stations with chord,
% twist, and CST coefficient distributions along the span.
%
% References:
%   Kulfan, B.M., "Universal Parametric Geometry Representation Method",
%   Journal of Aircraft, Vol. 45, No. 1, 2008.

function cst_propeller()
    %% Main script
    clc; close all;

    % Blade parameters
    R_hub = 0.1;
    R_tip = 1.0;
    c_root = 0.15;
    c_tip = 0.05;
    twist_root = 40.0;  % degrees
    twist_tip = 15.0;   % degrees
    dz_te = 0.001;

    % CST coefficients at root and tip
    A_upper_root = [0.20, 0.18, 0.19, 0.17, 0.15, 0.12, 0.09];
    A_lower_root = [-0.20, -0.18, -0.19, -0.17, -0.15, -0.12, -0.09];
    A_upper_tip  = [0.12, 0.10, 0.11, 0.10, 0.08, 0.06, 0.04];
    A_lower_tip  = [-0.12, -0.10, -0.11, -0.10, -0.08, -0.06, -0.04];

    % Grid parameters
    n_sections = 15;
    n_points = 101;

    % Generate blade
    [X, Y, Z] = generate_blade(n_sections, n_points, ...
        R_hub, R_tip, c_root, c_tip, twist_root, twist_tip, ...
        A_upper_root, A_lower_root, A_upper_tip, A_lower_tip, dz_te);

    fprintf('=== CST Propeller Blade Parameterization ===\n');
    fprintf('Blade surface grid: [%d x %d]\n', size(X, 1), size(X, 2));
    fprintf('Radial range: %.2f to %.2f\n', Z(1, 1), Z(1, end));
    fprintf('Number of sections: %d\n', n_sections);
    fprintf('Points per section: %d\n', size(X, 1));

    % Plot 3D blade
    plot_blade_3d(X, Y, Z, 'CST Propeller Blade');

    % Plot sections
    plot_sections(X, Y, Z);
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

%% Compute 2D CST section
function [y_upper, y_lower] = cst_2d_section(x, A_upper, A_lower, dz_te)
    if nargin < 4, dz_te = 0.0; end
    C = class_func(x, 0.5, 1.0);
    y_upper = C .* shape_func(x, A_upper) + x * dz_te;
    y_lower = C .* shape_func(x, A_lower) - x * dz_te;
end

%% Chord distribution
function chord = chord_dist(r, R_hub, R_tip, c_root, c_tip)
    eta = (r - R_hub) / (R_tip - R_hub);
    chord = c_root + (c_tip - c_root) * eta;
end

%% Twist distribution (linear, in degrees)
function twist = twist_dist(r, R_hub, R_tip, twist_root, twist_tip)
    eta = (r - R_hub) / (R_tip - R_hub);
    twist = twist_root + (twist_tip - twist_root) * eta;
end

%% Generate 3D blade geometry
function [X, Y, Z] = generate_blade(n_sections, n_points, ...
    R_hub, R_tip, c_root, c_tip, twist_root, twist_tip, ...
    A_upper_root, A_lower_root, A_upper_tip, A_lower_tip, dz_te)

    % Radial stations
    r = linspace(R_hub, R_tip, n_sections);
    eta = (r - R_hub) / (R_tip - R_hub);

    % Chordwise coordinates (cosine spacing)
    th = linspace(0, pi, n_points);
    x_norm = 0.5 * (1 - cos(th));

    % Distributions
    chords = chord_dist(r, R_hub, R_tip, c_root, c_tip);
    twists = twist_dist(r, R_hub, R_tip, twist_root, twist_tip);

    % Surface arrays
    n_surf = 2 * n_points - 1;  % upper + lower, shared TE
    X = zeros(n_surf, n_sections);
    Y = zeros(n_surf, n_sections);
    Z = zeros(n_surf, n_sections);

    for j = 1:n_sections
        % Interpolate CST coefficients
        A_upper = A_upper_root * (1 - eta(j)) + A_upper_tip * eta(j);
        A_lower = A_lower_root * (1 - eta(j)) + A_lower_tip * eta(j);

        % Generate 2D section
        [y_upper, y_lower] = cst_2d_section(x_norm, A_upper, A_lower, dz_te);

        % Scale by chord
        c = chords(j);
        x_sec = [fliplr(x_norm), x_norm(2:end)] * c;
        y_sec = [fliplr(y_upper), y_lower(2:end)] * c;

        % Apply twist (rotate about quarter-chord)
        twist_rad = deg2rad(twists(j));
        x_qc = 0.25 * c;
        x_rot = (x_sec - x_qc) * cos(twist_rad) - y_sec * sin(twist_rad) + x_qc;
        y_rot = (x_sec - x_qc) * sin(twist_rad) + y_sec * cos(twist_rad);

        X(:, j) = x_rot';
        Y(:, j) = y_rot';
        Z(:, j) = r(j);
    end
end

%% Plot 3D blade
function plot_blade_3d(X, Y, Z, fig_title)
    figure('Position', [100 100 900 700]);
    surf(X, Z, Y, 'FaceAlpha', 0.7, 'EdgeColor', 'k', 'LineWidth', 0.2);
    colormap('cool');
    xlabel('Chordwise (x)');
    ylabel('Spanwise (r)');
    zlabel('Thickness (y)');
    title(fig_title);
    axis equal;
    view([-30, 25]);
    grid on;
    light('Position', [1 1 1]);
    lighting gouraud;
end

%% Plot blade sections
function plot_sections(X, Y, Z)
    figure('Position', [100 100 800 500]);
    hold on;
    n_sec = size(X, 2);
    % Plot every few sections
    idx = round(linspace(1, n_sec, min(5, n_sec)));
    for k = 1:length(idx)
        j = idx(k);
        r_val = Z(1, j);
        plot(X(:, j), Y(:, j), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('r = %.2f', r_val));
    end
    xlabel('x');
    ylabel('y');
    title('Blade Sections at Different Radial Stations');
    axis equal;
    legend('Location', 'best');
    grid on;
    hold off;
end
