"""
CST (Class function/Shape function Transformation) parameterization for propeller blades.

Extends the 2D CST airfoil method to 3D propeller blade geometry by applying
CST profiles at multiple radial stations along the blade span, combined with
chord distribution, twist distribution, and sweep/lean.

References:
  Kulfan, B.M., "Universal Parametric Geometry Representation Method",
  Journal of Aircraft, Vol. 45, No. 1, 2008.
  Masters et al., "Geometric Comparison of Aerofoil Shape Parameterization Methods",
  AIAA Journal, Vol. 55, No. 5, 2017.
"""

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def bernstein_poly(n, i, x):
    """Compute the i-th Bernstein basis polynomial of degree n at x."""
    K = factorial(n) / (factorial(i) * factorial(n - i))
    return K * x**i * (1 - x)**(n - i)


def class_function(x, N1=0.5, N2=1.0):
    """CST class function."""
    return x**N1 * (1 - x)**N2


def shape_function(x, A):
    """CST shape function using Bernstein polynomials."""
    n = len(A) - 1
    S = np.zeros_like(x)
    for i, a in enumerate(A):
        S += a * bernstein_poly(n, i, x)
    return S


def cst_2d_section(x, A_upper, A_lower, dz_te=0.0, N1=0.5, N2=1.0):
    """Generate a single 2D airfoil section using CST."""
    C = class_function(x, N1, N2)
    y_upper = C * shape_function(x, A_upper) + x * dz_te
    y_lower = C * shape_function(x, A_lower) - x * dz_te
    return y_upper, y_lower


def chord_distribution(r, R_hub, R_tip, c_root, c_tip, c_max=None, r_max=None):
    """
    Compute chord length distribution along the blade span.

    Parameters
    ----------
    r : ndarray
        Radial stations (dimensional).
    R_hub : float
        Hub radius.
    R_tip : float
        Tip radius.
    c_root, c_tip : float
        Chord at root and tip.
    c_max : float, optional
        Maximum chord length (if None, linear distribution).
    r_max : float, optional
        Radial location of maximum chord.

    Returns
    -------
    chord : ndarray
        Chord at each radial station.
    """
    eta = (r - R_hub) / (R_tip - R_hub)  # normalized span 0..1

    if c_max is not None and r_max is not None:
        eta_max = (r_max - R_hub) / (R_tip - R_hub)
        chord = np.where(
            eta <= eta_max,
            c_root + (c_max - c_root) * (eta / eta_max),
            c_max + (c_tip - c_max) * ((eta - eta_max) / (1 - eta_max))
        )
    else:
        chord = c_root + (c_tip - c_root) * eta

    return chord


def twist_distribution(r, R_hub, R_tip, twist_root, twist_tip):
    """
    Linear twist distribution (in degrees).

    Parameters
    ----------
    r : ndarray
        Radial stations.
    R_hub, R_tip : float
        Hub and tip radii.
    twist_root, twist_tip : float
        Twist angles at root and tip (degrees).

    Returns
    -------
    twist : ndarray
        Twist angle at each radial station (degrees).
    """
    eta = (r - R_hub) / (R_tip - R_hub)
    return twist_root + (twist_tip - twist_root) * eta


def cst_propeller_blade(n_sections=10, n_points=101,
                        R_hub=0.1, R_tip=1.0,
                        c_root=0.15, c_tip=0.05,
                        twist_root=40.0, twist_tip=15.0,
                        A_upper_root=None, A_lower_root=None,
                        A_upper_tip=None, A_lower_tip=None,
                        dz_te=0.001):
    """
    Generate 3D propeller blade geometry using CST parameterization.

    The blade is built by stacking CST airfoil sections at multiple radial
    stations, with chord, twist, and CST coefficients varying along the span.

    Parameters
    ----------
    n_sections : int
        Number of radial sections.
    n_points : int
        Number of chordwise points per section.
    R_hub, R_tip : float
        Hub and tip radii.
    c_root, c_tip : float
        Root and tip chord lengths.
    twist_root, twist_tip : float
        Root and tip twist angles (degrees).
    A_upper_root, A_lower_root : array-like
        CST coefficients at root section.
    A_upper_tip, A_lower_tip : array-like
        CST coefficients at tip section.
    dz_te : float
        Trailing edge half-thickness.

    Returns
    -------
    X, Y, Z : ndarray
        3D coordinates of the blade surface (shape: 2*n_points x n_sections).
    """
    # Default CST coefficients (thicker root, thinner tip)
    if A_upper_root is None:
        A_upper_root = np.array([0.20, 0.18, 0.19, 0.17, 0.15, 0.12, 0.09])
    if A_lower_root is None:
        A_lower_root = np.array([-0.20, -0.18, -0.19, -0.17, -0.15, -0.12, -0.09])
    if A_upper_tip is None:
        A_upper_tip = np.array([0.12, 0.10, 0.11, 0.10, 0.08, 0.06, 0.04])
    if A_lower_tip is None:
        A_lower_tip = np.array([-0.12, -0.10, -0.11, -0.10, -0.08, -0.06, -0.04])

    A_upper_root = np.asarray(A_upper_root, dtype=float)
    A_lower_root = np.asarray(A_lower_root, dtype=float)
    A_upper_tip = np.asarray(A_upper_tip, dtype=float)
    A_lower_tip = np.asarray(A_lower_tip, dtype=float)

    # Radial stations
    r = np.linspace(R_hub, R_tip, n_sections)
    eta = (r - R_hub) / (R_tip - R_hub)

    # Chordwise coordinates (cosine spacing)
    theta = np.linspace(0, np.pi, n_points)
    x_norm = 0.5 * (1 - np.cos(theta))

    # Distributions
    chords = chord_distribution(r, R_hub, R_tip, c_root, c_tip)
    twists = twist_distribution(r, R_hub, R_tip, twist_root, twist_tip)

    # Build 3D surface
    n_surf = 2 * n_points - 1  # upper + lower (shared TE point)
    X = np.zeros((n_surf, n_sections))
    Y = np.zeros((n_surf, n_sections))
    Z = np.zeros((n_surf, n_sections))

    for j in range(n_sections):
        # Interpolate CST coefficients along span
        A_upper = A_upper_root * (1 - eta[j]) + A_upper_tip * eta[j]
        A_lower = A_lower_root * (1 - eta[j]) + A_lower_tip * eta[j]

        # Generate 2D section
        y_upper, y_lower = cst_2d_section(x_norm, A_upper, A_lower, dz_te)

        # Scale by chord
        c = chords[j]
        x_sec = np.concatenate([x_norm[::-1], x_norm[1:]]) * c
        y_sec = np.concatenate([y_upper[::-1], y_lower[1:]]) * c

        # Apply twist (rotation about quarter-chord)
        twist_rad = np.radians(twists[j])
        x_qc = 0.25 * c  # quarter chord
        x_rot = (x_sec - x_qc) * np.cos(twist_rad) - y_sec * np.sin(twist_rad) + x_qc
        y_rot = (x_sec - x_qc) * np.sin(twist_rad) + y_sec * np.cos(twist_rad)

        X[:, j] = x_rot
        Y[:, j] = y_rot
        Z[:, j] = r[j]

    return X, Y, Z


def plot_propeller_blade(X, Y, Z, title="CST Propeller Blade"):
    """Plot the 3D propeller blade surface."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Z, Y, alpha=0.7, cmap='coolwarm', edgecolor='k', linewidth=0.2)
    ax.set_xlabel('Chordwise (x)')
    ax.set_ylabel('Spanwise (r)')
    ax.set_zlabel('Thickness (y)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_blade_sections(n_sections=5, **kwargs):
    """Plot individual blade sections at different radial stations."""
    X, Y, Z = cst_propeller_blade(n_sections=n_sections, **kwargs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for j in range(n_sections):
        r_val = Z[0, j]
        ax.plot(X[:, j], Y[:, j], label=f'r = {r_val:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Blade Sections at Different Radial Stations')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---- Example usage ----
if __name__ == "__main__":
    print("=== CST Propeller Blade Parameterization ===\n")

    # Generate blade
    X, Y, Z = cst_propeller_blade(
        n_sections=15,
        n_points=101,
        R_hub=0.1,
        R_tip=1.0,
        c_root=0.15,
        c_tip=0.05,
        twist_root=40.0,
        twist_tip=15.0,
        dz_te=0.001
    )

    print(f"Blade surface grid: {X.shape}")
    print(f"Radial range: {Z[0, 0]:.2f} to {Z[0, -1]:.2f}")
    print(f"Number of sections: {X.shape[1]}")
    print(f"Points per section: {X.shape[0]}")

    # Plot 3D blade
    plot_propeller_blade(X, Y, Z)

    # Plot individual sections
    plot_blade_sections(n_sections=5)
