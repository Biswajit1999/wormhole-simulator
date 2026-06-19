"""Curvature, Einstein tensor and energy-condition diagnostics.

All tensors are computed numerically from a ``Metric`` instance so the module
works for any subclass.  For the canonical static Morris-Thorne profile we also
provide closed-form stress-energy components, which are used by the unit tests
as a cross-check against the numerical pipeline.

Einstein equations in geometrized units (G = c = 1):  G_{mu nu} = 8 pi T_{mu nu}.
"""

from __future__ import annotations

import numpy as np

from .metrics import Metric

_FD = 1e-5


def riemann(metric: Metric, x: np.ndarray) -> np.ndarray:
    """Riemann tensor R^rho_{sigma mu nu} via finite differences of Christoffels.

    R^r_{s m n} = d_m Gamma^r_{n s} - d_n Gamma^r_{m s}
                  + Gamma^r_{m l} Gamma^l_{n s} - Gamma^r_{n l} Gamma^l_{m s}.
    """
    x = np.asarray(x, dtype=float)
    n = metric.dim

    # Christoffel derivatives
    dGamma = np.zeros((n, n, n, n))  # [a, r, m, nu] = d_a Gamma^r_{m nu}
    for a in range(n):
        xp, xm = x.copy(), x.copy()
        xp[a] += _FD
        xm[a] -= _FD
        dGamma[a] = (metric.christoffel(xp) - metric.christoffel(xm)) / (2.0 * _FD)

    G = metric.christoffel(x)
    R = np.zeros((n, n, n, n))
    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(n):
                    term = dGamma[mu, rho, nu, sig] - dGamma[nu, rho, mu, sig]
                    for lam in range(n):
                        term += (
                            G[rho, mu, lam] * G[lam, nu, sig]
                            - G[rho, nu, lam] * G[lam, mu, sig]
                        )
                    R[rho, sig, mu, nu] = term
    return R


def ricci(metric: Metric, x: np.ndarray) -> np.ndarray:
    """Ricci tensor R_{sigma nu} = R^mu_{sigma mu nu}."""
    R = riemann(metric, x)
    return np.einsum("msmn->sn", R)


def ricci_scalar(metric: Metric, x: np.ndarray) -> float:
    """Ricci scalar R = g^{sigma nu} R_{sigma nu}."""
    Ric = ricci(metric, x)
    ginv = metric.inverse(x)
    return float(np.einsum("sn,sn->", ginv, Ric))


def einstein_tensor(metric: Metric, x: np.ndarray) -> np.ndarray:
    """Einstein tensor G_{mu nu} = R_{mu nu} - 1/2 g_{mu nu} R."""
    Ric = ricci(metric, x)
    Rs = ricci_scalar(metric, x)
    g = metric.components(x)
    return Ric - 0.5 * g * Rs


def stress_energy(metric: Metric, x: np.ndarray) -> np.ndarray:
    """Stress-energy tensor T_{mu nu} = G_{mu nu} / (8 pi)."""
    return einstein_tensor(metric, x) / (8.0 * np.pi)


def kretschmann(metric: Metric, x: np.ndarray) -> float:
    """Kretschmann scalar K = R_{abcd} R^{abcd} (curvature magnitude)."""
    R = riemann(metric, x)  # R^a_{bcd}
    g = metric.components(x)
    ginv = metric.inverse(x)
    # Lower the first index: R_{abcd} = g_{ae} R^e_{bcd}
    R_low = np.einsum("ae,ebcd->abcd", g, R)
    # Raise all indices for the full contraction
    R_up = np.einsum("ap,bq,cr,ds,pqrs->abcd", ginv, ginv, ginv, ginv, R_low)
    return float(np.einsum("abcd,abcd->", R_low, R_up))


# --------------------------------------------------------------------------------------
# Energy conditions
# --------------------------------------------------------------------------------------
def _orthonormal_frame(metric: Metric, x: np.ndarray):
    """Return a diagonal orthonormal tetrad assuming a diagonal metric.

    Works for the static spherical metrics in this package; rotating metrics
    require a more careful tetrad and are handled by null-vector sampling.
    """
    g = metric.components(x)
    e_t = np.array([1.0 / np.sqrt(-g[0, 0]), 0, 0, 0])
    e_r = np.array([0, 1.0 / np.sqrt(g[1, 1]), 0, 0])
    e_th = np.array([0, 0, 1.0 / np.sqrt(g[2, 2]), 0])
    e_ph = np.array([0, 0, 0, 1.0 / np.sqrt(g[3, 3])])
    return e_t, e_r, e_th, e_ph


def null_energy_condition(metric: Metric, x: np.ndarray, n_dirs: int = 64) -> float:
    """Minimum of T_{mu nu} k^mu k^nu over sampled null directions k.

    A negative return value signals NEC violation at ``x`` (required at a
    traversable wormhole throat).
    """
    T = stress_energy(metric, x)
    g = metric.components(x)
    e_t, e_r, e_th, e_ph = _orthonormal_frame(metric, x)
    spatial = [e_r, e_th, e_ph]

    worst = np.inf
    for phi in np.linspace(0, 2 * np.pi, n_dirs, endpoint=False):
        for ct in np.linspace(-1, 1, max(2, n_dirs // 8)):
            st = np.sqrt(max(0.0, 1 - ct * ct))
            n_hat = ct * spatial[0] + st * np.cos(phi) * spatial[1] + st * np.sin(phi) * spatial[2]
            k = e_t + n_hat  # null vector: timelike + unit spacelike
            # Normalize away tiny numerical non-nullity
            val = float(k @ T @ k)
            worst = min(worst, val)
    return worst


def weak_energy_density(metric: Metric, x: np.ndarray) -> float:
    """Energy density rho = T_{mu nu} u^mu u^nu measured by a static observer."""
    T = stress_energy(metric, x)
    e_t, *_ = _orthonormal_frame(metric, x)
    return float(e_t @ T @ e_t)


# --------------------------------------------------------------------------------------
# Closed-form Morris-Thorne stress-energy (cross-check)
# --------------------------------------------------------------------------------------
def morris_thorne_components(b, bp, Phi, Phip, r):
    """Analytic orthonormal-frame stress-energy of a Morris-Thorne wormhole.

    Parameters are the shape function b(r), its derivative b'(r), the redshift
    Phi(r) and its derivative Phi'(r) evaluated at radius r.  Returns
    (rho, p_r, p_t) in geometrized units (Morris & Thorne 1988, Eqs. 11-13).
    """
    rho = bp / (8.0 * np.pi * r ** 2)
    p_r = (1.0 / (8.0 * np.pi)) * (-b / r ** 3 + 2.0 * (1.0 - b / r) * Phip / r)
    p_t = (1.0 / (8.0 * np.pi)) * (
        (1.0 - b / r)
        * (Phip ** 2 + (bp * r - b) / (2.0 * r * (r - b)) * (-Phip) + Phip / r)
    )
    return rho, p_r, p_t


# --------------------------------------------------------------------------------------
# full energy-condition suite (SEC, DEC), ANEC, exotic-matter quantifier
# --------------------------------------------------------------------------------------
def strong_energy_condition(metric: Metric, x: np.ndarray) -> float:
    """(T_{mu nu} - 1/2 T g_{mu nu}) u^mu u^nu for a static observer.

    Negative => SEC violation (generic for wormholes; the SEC governs whether
    gravity is attractive). Returns the static-observer value.
    """
    T = stress_energy(metric, x)
    g = metric.components(x)
    ginv = metric.inverse(x)
    trace = float(np.einsum("mn,mn->", ginv, T))
    e_t, *_ = _orthonormal_frame(metric, x)
    Tbar = T - 0.5 * trace * g
    return float(e_t @ Tbar @ e_t)


def principal_pressures(metric: Metric, x: np.ndarray):
    """Return (rho, p_r, p_theta, p_phi) in the static orthonormal frame."""
    T = stress_energy(metric, x)
    e_t, e_r, e_th, e_ph = _orthonormal_frame(metric, x)
    rho = float(e_t @ T @ e_t)
    p_r = float(e_r @ T @ e_r)
    p_th = float(e_th @ T @ e_th)
    p_ph = float(e_ph @ T @ e_ph)
    return rho, p_r, p_th, p_ph


def dominant_energy_condition(metric: Metric, x: np.ndarray) -> bool:
    """DEC: rho >= 0 and rho >= |p_i| for each principal pressure."""
    rho, p_r, p_th, p_ph = principal_pressures(metric, x)
    return bool(rho >= 0 and rho >= abs(p_r) and rho >= abs(p_th) and rho >= abs(p_ph))


def anec_integral(metric: Metric, result) -> float:
    """Averaged null energy along a geodesic: integral T_{mu nu} k^mu k^nu d lambda.

    ``result`` is a numerics.geodesic.GeodesicResult for a null ray.  A negative
    value indicates ANEC violation along that ray (a sharper obstruction than the
    pointwise NEC).
    """
    vals = []
    for i in range(len(result.affine)):
        T = stress_energy(metric, result.coords[i])
        k = result.velocity[i]
        vals.append(float(k @ T @ k))
    return float(np.trapz(np.asarray(vals), result.affine))


def exotic_matter_integral(shape_fn, dshape_fn, r0, r_max=50.0, n=4000) -> float:
    """Volume-integral quantifier of exotic matter for a Phi=0 wormhole.

    I = integral (rho + p_r) dV  with proper radial measure, using the analytic
    Morris-Thorne components.  For Phi = 0,

        rho + p_r = (1/8pi)(b' r - b)/r^3 ,   dV = 4 pi r^2 dr / sqrt(1 - b/r).

    A negative I (in units where it is finite) measures the total amount of
    energy-condition-violating matter threading the throat (Visser-Kar-Dadhich).
    """
    rs = np.linspace(r0 * (1 + 1e-6), r_max, n)
    b = shape_fn(rs)
    bp = dshape_fn(rs)
    rho_pr = (bp * rs - b) / (8.0 * np.pi * rs ** 3)
    dV = 4.0 * np.pi * rs ** 2 / np.sqrt(np.clip(1.0 - b / rs, 1e-12, None))
    return float(np.trapz(rho_pr * dV, rs))
