"""Geodesic integration for arbitrary metrics.

Solves   d^2 x^mu/dl^2 + Gamma^mu_{ab} dx^a/dl dx^b/dl = 0

as a first-order system in state = [x^mu, u^mu] using an adaptive
Dormand-Prince (RK45) integrator from SciPy, with a hand-written RK4 fallback
that matches the pseudocode in the documentation.

A conserved diagnostic, the norm  eps = g_{mu nu} u^mu u^nu, is tracked; it
should remain ~0 for null rays and ~ -1 for unit-normalized timelike particles.
Drift in eps is the primary integration-quality metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:
    from scipy.integrate import solve_ivp

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from core.metrics import Metric


def null_initial_velocity(metric: Metric, x: np.ndarray, spatial_dir: np.ndarray) -> np.ndarray:
    """Build a future-pointing null 4-velocity at x with given spatial direction.

    Given a 3-vector ``spatial_dir`` (components in r, theta, phi), solve the
    quadratic g_{mu nu} u^mu u^nu = 0 for the time component u^t > 0.
    """
    g = metric.components(x)
    u = np.zeros(4)
    u[1:] = spatial_dir
    # g_tt (u^t)^2 + 2 g_ti u^t u^i + g_ij u^i u^j = 0
    A = g[0, 0]
    B = 2.0 * (g[0, 1:] @ u[1:])
    C = u[1:] @ g[1:, 1:] @ u[1:]
    disc = B * B - 4 * A * C
    if disc < 0:
        raise ValueError("no real null solution for this spatial direction")
    ut = (-B + np.sqrt(disc)) / (2 * A)
    ut2 = (-B - np.sqrt(disc)) / (2 * A)
    u[0] = max(ut, ut2)  # future-pointing
    return u


@dataclass
class GeodesicResult:
    affine: np.ndarray          # affine parameter samples, shape (N,)
    coords: np.ndarray          # x^mu(lambda), shape (N, 4)
    velocity: np.ndarray        # u^mu(lambda), shape (N, 4)
    norm_drift: np.ndarray      # g_{mu nu} u^mu u^nu along the path, shape (N,)
    success: bool = True


@dataclass
class GeodesicSolver:
    """Integrate geodesics for a given metric.

    Parameters
    ----------
    metric : a core.metrics.Metric instance.
    rtol, atol : tolerances passed to the adaptive integrator.
    method : 'RK45' (SciPy Dormand-Prince) or 'rk4' (fixed-step fallback).
    """

    metric: Metric
    rtol: float = 1e-9
    atol: float = 1e-9
    method: str = "RK45"
    _events: list = field(default_factory=list, repr=False)

    def _rhs(self, _l, state):
        return self.metric.geodesic_rhs(np.asarray(state))

    def integrate(self, x0, u0, affine_span=(0.0, 50.0), n_eval=400) -> GeodesicResult:
        x0 = np.asarray(x0, dtype=float)
        u0 = np.asarray(u0, dtype=float)
        state0 = np.concatenate([x0, u0])
        l_eval = np.linspace(affine_span[0], affine_span[1], n_eval)

        if _HAVE_SCIPY and self.method.upper() == "RK45":
            sol = solve_ivp(
                self._rhs,
                affine_span,
                state0,
                t_eval=l_eval,
                rtol=self.rtol,
                atol=self.atol,
                method="RK45",
                dense_output=False,
            )
            coords = sol.y[:4].T
            velocity = sol.y[4:].T
            affine = sol.t
            success = sol.success
        else:
            affine, states = self._rk4(state0, affine_span, n_eval)
            coords = states[:, :4]
            velocity = states[:, 4:]
            success = True

        norm = np.array(
            [self.metric.line_element_norm(coords[i], velocity[i]) for i in range(len(affine))]
        )
        return GeodesicResult(affine, coords, velocity, norm, success)

    def _rk4(self, state0, span, n):
        """Fixed-step classical RK4 (mirrors the documentation pseudocode)."""
        l0, l1 = span
        h = (l1 - l0) / (n - 1)
        states = np.zeros((n, len(state0)))
        states[0] = state0
        s = state0.copy()
        for i in range(1, n):
            k1 = self._rhs(0, s)
            k2 = self._rhs(0, s + 0.5 * h * k1)
            k3 = self._rhs(0, s + 0.5 * h * k2)
            k4 = self._rhs(0, s + h * k3)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            states[i] = s
        affine = np.linspace(l0, l1, n)
        return affine, states


def integrate_geodesic(metric, x0, u0, step=0.05, n_steps=1000):
    """Functional RK4 geodesic integrator matching the docs pseudocode.

    Returns an (n_steps+1, 8) array of [x^mu, u^mu] states.
    """
    solver = GeodesicSolver(metric, method="rk4")
    span = (0.0, step * n_steps)
    res = solver.integrate(x0, u0, affine_span=span, n_eval=n_steps + 1)
    return np.hstack([res.coords, res.velocity])
