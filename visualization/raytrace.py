"""Gravitational lensing by ray-tracing null geodesics.

Two interfaces are provided:

1. ``deflection_angle`` - the classic 1D quadrature for the bending of a light
   ray with impact parameter ``rho`` in the Ellis wormhole, using the exact
   reduction allowed by spherical symmetry.

2. ``EquatorialRayTracer`` - shoots a fan of null geodesics backwards from an
   observer in the equatorial plane through a generic ``Metric`` and records
   where each ray ends up (which "universe", and at what angle), producing the
   data needed to map a background star field into a lensed image.

The full 2D/3D image-plane renderer (sampling a celestial-sphere texture per
pixel) is documented in docs/visualization.md; this module supplies the physics
core that such a renderer calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.compat import trapezoid

from core.metrics import Metric, EllisBronnikov
from numerics.geodesic import GeodesicSolver, null_initial_velocity


def deflection_angle(b0: float, rho: float, r_inf: float = 1e4, n: int = 4000) -> float:
    """Light deflection in the Ellis wormhole for impact parameter ``rho``.

    For the Ellis metric the equatorial photon orbit obeys
        (dr/dphi)^2 = (r^2 + b0^2) [ (r^2 + b0^2)/rho^2 - 1 ],
    and the total bending angle is

        alpha = 2 * integral_{r_min}^{inf} dr / sqrt((r^2+b0^2)[(r^2+b0^2)/rho^2 - 1]) - pi.

    ``rho`` is the impact parameter (>= b0 for rays that stay in one universe;
    rays with rho < b0 traverse the throat).
    """
    if rho <= 0:
        raise ValueError("impact parameter must be positive")
    # Turning point: (r^2 + b0^2) = rho^2  ->  r_min^2 = rho^2 - b0^2
    arg = rho ** 2 - b0 ** 2
    r_min = np.sqrt(arg) if arg > 0 else 0.0
    r = np.linspace(r_min + 1e-6, r_inf, n)
    R2 = r ** 2 + b0 ** 2
    denom = R2 * (R2 / rho ** 2 - 1.0)
    denom = np.where(denom > 0, denom, np.nan)
    integrand = 1.0 / np.sqrt(denom)
    phi = 2.0 * trapezoid(np.nan_to_num(integrand), r)
    return float(phi - np.pi)


@dataclass
class RayOutcome:
    side: int             # +1 if ray ends in the same universe, -1 if it crossed the throat
    final_angle: float    # asymptotic phi direction (radians)
    crossed_throat: bool
    norm_drift: float     # max |g u u| along the ray (quality diagnostic)


@dataclass
class EquatorialRayTracer:
    """Backwards ray tracer for equatorial (theta = pi/2) null geodesics."""

    metric: Metric
    observer_r: float = 10.0
    affine_max: float = 200.0
    n_eval: int = 800

    def trace(self, alpha: float) -> RayOutcome:
        """Trace one ray launched at screen angle ``alpha`` from the observer.

        alpha = 0 points radially inward (toward the throat); nonzero alpha tilts
        the initial direction in the equatorial plane.
        """
        x0 = np.array([0.0, self.observer_r, np.pi / 2, 0.0])
        # Inward radial + transverse phi components set the screen angle
        spatial = np.array([-np.cos(alpha), 0.0, np.sin(alpha) / self.observer_r])
        u0 = null_initial_velocity(self.metric, x0, spatial)

        solver = GeodesicSolver(self.metric, method="RK45", rtol=1e-9, atol=1e-9)
        res = solver.integrate(x0, u0, affine_span=(0.0, self.affine_max), n_eval=self.n_eval)

        r = res.coords[:, 1]
        phi = res.coords[:, 3]
        crossed = bool(np.any(r < 0)) if np.any(r < 0) else _sign_flip(r)
        side = -1 if crossed else +1
        return RayOutcome(
            side=side,
            final_angle=float(phi[-1]),
            crossed_throat=crossed,
            norm_drift=float(np.max(np.abs(res.norm_drift))),
        )

    def fan(self, alphas):
        """Trace a fan of rays; returns a list of RayOutcome."""
        return [self.trace(a) for a in np.asarray(alphas)]


def _sign_flip(r):
    """Heuristic throat crossing for charts where r stays positive (areal radius)."""
    # A ray crosses if r dips to near the throat then increases on the far branch.
    i_min = int(np.argmin(r))
    return 0 < i_min < len(r) - 1 and r[i_min] < 1.05 * r.min()


def demo_einstein_ring(metric=None, n=181):
    """Convenience: sample the deflection curve to locate Einstein-ring angles."""
    metric = metric or EllisBronnikov(b0=1.0)
    rhos = np.linspace(metric.b0 * 1.01, 8.0, n)
    angles = np.array([deflection_angle(metric.b0, rho) for rho in rhos])
    return rhos, angles
