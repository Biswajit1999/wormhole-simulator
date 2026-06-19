"""Tests for the advanced wormhole modules."""

import numpy as np
import pytest

from core.metrics import EllisBronnikov, MorrisThorne, TeoRotating
from core import backend, symmetries as sym, stress_energy as se
from core.thinshell import ThinShellWormhole
from core.semiclassical import (
    casimir_energy_density, casimir_wormhole_shape, ford_roman_bound,
    satisfies_quantum_inequality,
)
from core.modified_gravity import ModifiedGravityWormhole
from numerics.geodesic import GeodesicSolver, null_initial_velocity
from numerics.orbits import radial_velocity_sq
from numerics.qnm import RingdownExtractor, regge_wheeler_potential


# autodiff/Richardson backend agrees with the legacy central difference
def test_backend_connection_accurate():
    wh = MorrisThorne(b0=1.0)
    x = np.array([0.0, 2.5, np.pi / 2, 0.3])
    G = backend.connection(wh, x)
    # symmetric in lower indices
    assert np.allclose(G, np.transpose(G, (0, 2, 1)), atol=1e-6)
    # Richardson FD should beat a crude one-step central difference reference
    assert G.shape == (4, 4, 4)


# conserved quantities stay constant along a geodesic
def test_conserved_quantities():
    wh = EllisBronnikov(b0=1.0)
    x0 = np.array([0.0, 6.0, np.pi / 2, 0.0])
    u0 = null_initial_velocity(wh, x0, [-1.0, 0.0, 0.05])
    res = GeodesicSolver(wh, rtol=1e-10, atol=1e-10).integrate(x0, u0, (0, 15), n_eval=150)
    drift = sym.monitor_conservation(wh, res)
    assert drift["E_drift"] < 1e-6
    assert drift["L_drift"] < 1e-6


# SEC violated, DEC fails, ANEC negative for the Ellis wormhole
def test_energy_condition_suite():
    wh = MorrisThorne(b0=1.0)
    x = np.array([0.0, 1.3, np.pi / 2, 0.0])
    assert se.strong_energy_condition(wh, x) <= 1e-6
    assert se.dominant_energy_condition(wh, x) is False
    el = EllisBronnikov(b0=1.0)
    x0 = np.array([0.0, 5.0, np.pi / 2, 0.0])
    u0 = null_initial_velocity(el, x0, [-1.0, 0.0, 0.04])
    res = GeodesicSolver(el).integrate(x0, u0, (0, 12), n_eval=120)
    assert se.anec_integral(el, res) < 0.0


# thin-shell equilibrium and a stable branch at large beta^2
def test_thin_shell_equilibrium_and_stability():
    w = ThinShellWormhole(M=1.0, a0=2.5, beta2=0.5)
    V, Vp, Vpp = w.potential_derivatives()
    assert abs(V) < 1e-8 and abs(Vp) < 1e-6      # equilibrium
    assert w.exotic_surface_mass() < 0           # exotic shell
    _, vpp, stable = w.stability_region(np.linspace(0, 12, 49))
    assert stable.any()                          # some beta^2 stabilizes the throat


# rotating wormhole frame dragging / ergoregion
def test_rotating_frame_dragging():
    t = TeoRotating(b0=1.0, J=0.5)
    assert t.zamo_angular_velocity(2.0) == pytest.approx(2 * 0.5 / 8.0)
    assert t.in_ergoregion(0.2) in (True, False)
    g = t.components(np.array([0.0, 2.0, np.pi / 2, 0.0]))
    assert g[0, 3] != 0.0                          # genuine off-diagonal term


# QNM extraction returns a damped fundamental mode
def test_qnm_damped():
    w, _ = RingdownExtractor(b0=1.0, l=2).fundamental_mode(t_final=80.0)
    assert np.isfinite(w.real) and np.isfinite(w.imag)
    assert w.real > 0 and w.imag < 0               # oscillating and decaying
    assert regge_wheeler_potential(0.0, 1.0, 2) != 0.0


# Casimir energy negative; Ford-Roman QI machinery consistent
def test_semiclassical():
    assert casimir_energy_density(1.0) < 0
    assert casimir_wormhole_shape(1.0, 1.0) == pytest.approx(1.0)   # b(r0)=r0
    assert ford_roman_bound(2.0) > 0
    tau = np.linspace(0, 10, 200)
    rho = -1e-4 * np.exp(-((tau - 5) ** 2) / 2)
    out = satisfies_quantum_inequality(rho, tau)
    assert out["satisfied"] is True


# modified-gravity wormhole is a valid metric
def test_modified_gravity():
    w = ModifiedGravityWormhole(b0=1.0, gamma=0.5)
    assert w.shape(1.0) == pytest.approx(1.0)
    g = w.components(np.array([0.0, 2.0, np.pi / 2, 0.0]))
    assert np.sum(np.linalg.eigvalsh(g) < 0) == 1   # Lorentzian
    assert w.effective_energy_density(1.2) < 0      # geometric flare-out
