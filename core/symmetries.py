"""Killing symmetries and conserved quantities.

Static spherical and stationary axisymmetric wormholes possess the Killing
vectors xi_(t) = d/dt and xi_(phi) = d/dphi.  Each gives a constant of geodesic
motion:

    E = -xi_(t) . u = -g_{t mu} u^mu        (conserved energy)
    L =  xi_(phi) . u =  g_{phi mu} u^mu     (conserved axial angular momentum)

These reduce equatorial geodesics to a 1-D radial problem (numerics.orbits) and
give an independent integration-quality check.
"""

from __future__ import annotations

import numpy as np

from .metrics import Metric


def energy(metric: Metric, x: np.ndarray, u: np.ndarray) -> float:
    """Conserved energy E = -g_{t mu} u^mu."""
    g = metric.components(np.asarray(x, float))
    return float(-(g[0] @ np.asarray(u, float)))


def angular_momentum(metric: Metric, x: np.ndarray, u: np.ndarray) -> float:
    """Conserved axial angular momentum L = g_{phi mu} u^mu."""
    g = metric.components(np.asarray(x, float))
    return float(g[3] @ np.asarray(u, float))


def conserved_quantities(metric: Metric, x: np.ndarray, u: np.ndarray) -> dict:
    """Return {'E', 'L', 'norm'} for a single state."""
    return {
        "E": energy(metric, x, u),
        "L": angular_momentum(metric, x, u),
        "norm": metric.line_element_norm(np.asarray(x, float), np.asarray(u, float)),
    }


def monitor_conservation(metric: Metric, result) -> dict:
    """Max absolute drift of E, L and the norm along a GeodesicResult.

    Returns {'E_drift', 'L_drift', 'norm_drift'} — small values confirm the
    integration respected the spacetime symmetries.
    """
    E = np.array([energy(metric, result.coords[i], result.velocity[i])
                  for i in range(len(result.affine))])
    L = np.array([angular_momentum(metric, result.coords[i], result.velocity[i])
                  for i in range(len(result.affine))])
    return {
        "E_drift": float(np.max(np.abs(E - E[0]))),
        "L_drift": float(np.max(np.abs(L - L[0]))),
        "norm_drift": float(np.max(np.abs(result.norm_drift))),
    }
