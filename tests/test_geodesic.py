"""Unit tests for the geodesic integrator.

The key physical invariant is the conservation of the 4-velocity norm
g_{mu nu} u^mu u^nu along an affinely parameterized geodesic: 0 for null rays,
-1 for unit-normalized timelike particles.
"""

import numpy as np
import pytest

from core.metrics import EllisBronnikov, MorrisThorne
from numerics.geodesic import GeodesicSolver, null_initial_velocity


def test_null_norm_conserved_ellis():
    wh = EllisBronnikov(b0=1.0)
    x0 = np.array([0.0, 6.0, np.pi / 2, 0.0])
    u0 = null_initial_velocity(wh, x0, np.array([-1.0, 0.0, 0.05]))
    # initial vector is null
    assert wh.line_element_norm(x0, u0) == pytest.approx(0.0, abs=1e-9)

    solver = GeodesicSolver(wh, rtol=1e-10, atol=1e-10)
    res = solver.integrate(x0, u0, affine_span=(0, 20), n_eval=200)
    assert np.max(np.abs(res.norm_drift)) < 1e-5


def test_radial_null_crosses_throat_ellis():
    """A purely radial null ray aimed inward should pass r=0 (throat crossing)."""
    wh = EllisBronnikov(b0=1.0)
    x0 = np.array([0.0, 5.0, np.pi / 2, 0.0])
    u0 = null_initial_velocity(wh, x0, np.array([-1.0, 0.0, 0.0]))
    solver = GeodesicSolver(wh, rtol=1e-10, atol=1e-10)
    res = solver.integrate(x0, u0, affine_span=(0, 12), n_eval=300)
    assert np.min(res.coords[:, 1]) < 0.0  # crossed to the other universe


def test_timelike_norm_conserved_morris_thorne():
    wh = MorrisThorne(b0=1.0)
    x0 = np.array([0.0, 8.0, np.pi / 2, 0.0])
    # Build a unit timelike velocity: mostly time, small radial
    g = wh.components(x0)
    ur = 0.1
    # solve g_tt ut^2 + g_rr ur^2 = -1
    ut = np.sqrt((1 + g[1, 1] * ur ** 2) / (-g[0, 0]))
    u0 = np.array([ut, ur, 0.0, 0.0])
    assert wh.line_element_norm(x0, u0) == pytest.approx(-1.0, abs=1e-9)

    solver = GeodesicSolver(wh, rtol=1e-10, atol=1e-10)
    res = solver.integrate(x0, u0, affine_span=(0, 10), n_eval=200)
    assert np.max(np.abs(res.norm_drift + 1.0)) < 1e-4
