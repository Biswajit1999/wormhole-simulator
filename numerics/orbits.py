"""Equatorial geodesics via conserved quantities (reduced 1-D problem).

For a static spherical metric diag(g_tt, g_rr, r^2, r^2 sin^2 theta) the energy
E and angular momentum L reduce the equatorial (theta = pi/2) geodesic to

    (u^r)^2 = (E^2 / (-g_tt) - L^2 / g_phiphi + eps) / g_rr ,

with eps = 0 (null) or -1 (timelike).  Integrating dphi/dr = u^phi / u^r traces
the orbit in the plane, far cheaper and more accurate than the full 4-D system
because E and L are conserved *exactly* by construction.  Turning points are the
roots of (u^r)^2 = 0; the radial effective potential follows directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.metrics import Metric


def radial_velocity_sq(metric: Metric, r: float, E: float, L: float, eps: float = 0.0) -> float:
    """(u^r)^2 at equatorial radius r for given E, L and norm eps."""
    x = np.array([0.0, r, np.pi / 2, 0.0])
    g = metric.components(x)
    g_tt, g_rr, g_pp = g[0, 0], g[1, 1], g[3, 3]
    return (E ** 2 / (-g_tt) - L ** 2 / g_pp + eps) / g_rr


def effective_potential(metric: Metric, r: float, L: float, eps: float = 0.0) -> float:
    """Radial effective potential V(r) with (u^r)^2 = E^2/(-g_tt g_rr) - V.

    Defined so that turning points satisfy E^2 = V_eff-like balance; returned
    here as the L- and eps-dependent part for plotting photon/particle barriers.
    """
    x = np.array([0.0, r, np.pi / 2, 0.0])
    g = metric.components(x)
    g_rr, g_pp = g[1, 1], g[3, 3]
    return (L ** 2 / g_pp - eps) / g_rr


@dataclass
class EquatorialOrbit:
    metric: Metric
    eps: float = 0.0  # 0 null, -1 timelike

    def from_impact_parameter(self, r_obs: float, impact: float):
        """Set (E, L) for a ray from r_obs with given impact parameter b = L/E.

        Uses E = 1 (affine rescaling freedom for null rays) so L = impact.
        """
        E = 1.0
        L = impact
        return E, L

    def trace(self, r_start, r_stop, E, L, n=600, sign=-1):
        """Integrate phi(r) from r_start to r_stop. ``sign`` is the initial sign of u^r.

        Returns (r, phi) arrays. Stops/reflects at a turning point if reached.
        """
        rs = np.linspace(r_start, r_stop, n)
        phi = np.zeros(n)
        s = sign
        for i in range(1, n):
            rm = 0.5 * (rs[i] + rs[i - 1])
            vr2 = radial_velocity_sq(self.metric, rm, E, L, self.eps)
            if vr2 <= 0:  # turning point: reflect
                s = -s
                phi[i] = phi[i - 1]
                continue
            up = L / (rm ** 2)              # u^phi = L / g_phiphi (equatorial)
            ur = s * np.sqrt(vr2)
            dphi_dr = up / ur
            phi[i] = phi[i - 1] + dphi_dr * (rs[i] - rs[i - 1])
        return rs, phi

    def turning_points(self, r_grid, E, L):
        """Radii where (u^r)^2 changes sign (orbit turning points)."""
        vals = np.array([radial_velocity_sq(self.metric, r, E, L, self.eps) for r in r_grid])
        idx = np.where(np.diff(np.sign(vals)) != 0)[0]
        return r_grid[idx]
