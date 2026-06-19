"""Chebyshev pseudospectral differentiation utilities.

For smooth wormhole metric functions (shape, redshift, lapse) spectral
collocation gives exponential convergence.  These helpers build the standard
Chebyshev-Gauss-Lobatto nodes and the associated differentiation matrix
(Trefethen, *Spectral Methods in MATLAB*, 2000), useful for solving the
boundary-value problems that arise in static-wormhole field equations.
"""

from __future__ import annotations

import numpy as np


def chebyshev_nodes(n: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes x_j = cos(pi j / n) on [-1, 1]."""
    return np.cos(np.pi * np.arange(n + 1) / n)


def chebyshev_diff_matrix(n: int):
    """Return (D, x): the (n+1)x(n+1) Chebyshev differentiation matrix and nodes.

    Spectral derivative of a function sampled at the returned nodes x is D @ f.
    Map to a physical interval [a, b] by scaling: D_phys = (2/(b-a)) * D.
    """
    if n == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = chebyshev_nodes(n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def solve_bvp_dirichlet(rhs, bc_left, bc_right, n=64, interval=(-1.0, 1.0)):
    """Solve u'' = rhs(x) with Dirichlet BCs on a Chebyshev grid.

    Demonstrator for spectral solution of static field equations on a finite
    radial domain.  Returns (x_physical, u).
    """
    a, b = interval
    D, x = chebyshev_diff_matrix(n)
    scale = 2.0 / (b - a)
    D = scale * D
    D2 = D @ D
    xp = a + (x + 1.0) * (b - a) / 2.0

    A = D2.copy()
    f = rhs(xp).astype(float)
    # Apply Dirichlet conditions at the endpoints (x[0]=+1 -> b, x[-1]=-1 -> a)
    A[0, :] = 0.0
    A[0, 0] = 1.0
    f[0] = bc_right
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    f[-1] = bc_left
    u = np.linalg.solve(A, f)
    return xp, u
