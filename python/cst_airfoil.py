"""
CST (Class function/Shape function Transformation) parameterization for airfoils.

Based on Kulfan's CST method:
  Kulfan, B.M., "Universal Parametric Geometry Representation Method",
  Journal of Aircraft, Vol. 45, No. 1, 2008, pp. 142-158.

The airfoil surface is represented as:
  y(x) = C(x) * S(x) + x * dz_te
where:
  C(x) = x^N1 * (1-x)^N2          (class function)
  S(x) = sum( A_i * K_i * x^i * (1-x)^(n-i) )  (shape function, Bernstein polynomials)
  dz_te = trailing edge thickness
  N1, N2 = 0.5, 1.0 for round-nose airfoils
"""

import numpy as np
from math import factorial
import matplotlib.pyplot as plt


def bernstein_poly(n, i, x):
    """Compute the i-th Bernstein basis polynomial of degree n at x."""
    K = factorial(n) / (factorial(i) * factorial(n - i))
    return K * x**i * (1 - x)**(n - i)


def class_function(x, N1=0.5, N2=1.0):
    """
    CST class function for airfoils.
    N1=0.5, N2=1.0 gives a round leading edge and sharp trailing edge.
    """
    return x**N1 * (1 - x)**N2


def shape_function(x, A):
    """
    CST shape function using Bernstein polynomials.
    A: array of CST coefficients (Bernstein weights).
    """
    n = len(A) - 1
    S = np.zeros_like(x)
    for i, a in enumerate(A):
        S += a * bernstein_poly(n, i, x)
    return S


def cst_airfoil(x, A_upper, A_lower, dz_te=0.0, N1=0.5, N2=1.0):
    """
    Generate airfoil coordinates using CST parameterization.

    Parameters
    ----------
    x : ndarray
        Normalized chordwise coordinates, 0 <= x <= 1.
    A_upper : array-like
        CST coefficients for the upper surface.
    A_lower : array-like
        CST coefficients for the lower surface.
    dz_te : float
        Half trailing edge thickness (applied symmetrically).
    N1, N2 : float
        Class function exponents (default 0.5, 1.0 for airfoils).

    Returns
    -------
    y_upper : ndarray
        Upper surface y-coordinates.
    y_lower : ndarray
        Lower surface y-coordinates.
    """
    A_upper = np.asarray(A_upper, dtype=float)
    A_lower = np.asarray(A_lower, dtype=float)

    C = class_function(x, N1, N2)
    S_upper = shape_function(x, A_upper)
    S_lower = shape_function(x, A_lower)

    y_upper = C * S_upper + x * dz_te
    y_lower = C * S_lower - x * dz_te

    return y_upper, y_lower


def fit_cst_airfoil(x_data, y_upper_data, y_lower_data, order=6,
                    dz_te=0.0, N1=0.5, N2=1.0):
    """
    Fit CST coefficients to existing airfoil coordinate data using least squares.

    Parameters
    ----------
    x_data : ndarray
        Chordwise coordinates (0 to 1).
    y_upper_data : ndarray
        Upper surface y-coordinates.
    y_lower_data : ndarray
        Lower surface y-coordinates.
    order : int
        Order of the Bernstein polynomial (number of coefficients = order + 1).
    dz_te : float
        Half trailing edge thickness.
    N1, N2 : float
        Class function exponents.

    Returns
    -------
    A_upper : ndarray
        Fitted upper surface CST coefficients.
    A_lower : ndarray
        Fitted lower surface CST coefficients.
    """
    C = class_function(x_data, N1, N2)
    n = order

    # Build Bernstein basis matrix
    B = np.zeros((len(x_data), n + 1))
    for i in range(n + 1):
        B[:, i] = bernstein_poly(n, i, x_data)

    # Upper surface: y_upper - x*dz_te = C * S_upper = C * B @ A_upper
    rhs_upper = y_upper_data - x_data * dz_te
    W_upper = np.diag(C) @ B
    A_upper, _, _, _ = np.linalg.lstsq(W_upper, rhs_upper, rcond=None)

    # Lower surface: y_lower + x*dz_te = C * S_lower = C * B @ A_lower
    rhs_lower = y_lower_data + x_data * dz_te
    W_lower = np.diag(C) @ B
    A_lower, _, _, _ = np.linalg.lstsq(W_lower, rhs_lower, rcond=None)

    return A_upper, A_lower


def plot_airfoil(x, y_upper, y_lower, title="CST Airfoil"):
    """Plot the airfoil shape."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x, y_upper, 'b-', linewidth=1.5, label='Upper surface')
    ax.plot(x, y_lower, 'r-', linewidth=1.5, label='Lower surface')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---- Example usage ----
if __name__ == "__main__":
    # Define CST coefficients (example: NACA 0012-like)
    A_upper = [0.17, 0.15, 0.16, 0.14, 0.12, 0.10, 0.08]
    A_lower = [-0.17, -0.15, -0.16, -0.14, -0.12, -0.10, -0.08]
    dz_te = 0.001  # small trailing edge gap

    # Generate airfoil coordinates
    x = np.linspace(0, 1, 201)
    # Use cosine spacing for better LE resolution
    theta = np.linspace(0, np.pi, 201)
    x_cos = 0.5 * (1 - np.cos(theta))

    y_upper, y_lower = cst_airfoil(x_cos, A_upper, A_lower, dz_te)

    print("=== CST Airfoil Parameterization ===")
    print(f"Upper surface coefficients: {A_upper}")
    print(f"Lower surface coefficients: {A_lower}")
    print(f"Trailing edge gap: {2 * dz_te}")
    print(f"Number of points: {len(x_cos)}")
    print(f"Max thickness: {np.max(y_upper - y_lower):.6f}")

    # Plot the airfoil
    plot_airfoil(x_cos, y_upper, y_lower, title="CST Parameterized Airfoil")

    # Demonstrate fitting: fit CST to the generated airfoil
    A_upper_fit, A_lower_fit = fit_cst_airfoil(x_cos, y_upper, y_lower, order=6, dz_te=dz_te)
    print(f"\nFitted upper coefficients: {np.round(A_upper_fit, 6)}")
    print(f"Fitted lower coefficients: {np.round(A_lower_fit, 6)}")

    y_upper_fit, y_lower_fit = cst_airfoil(x_cos, A_upper_fit, A_lower_fit, dz_te)
    error = np.max(np.abs(y_upper - y_upper_fit)) + np.max(np.abs(y_lower - y_lower_fit))
    print(f"Max fitting error: {error:.2e}")
