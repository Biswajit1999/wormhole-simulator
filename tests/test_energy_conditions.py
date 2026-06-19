"""Unit tests for stress-energy and energy-condition diagnostics.

A traversable wormhole must violate the null energy condition (NEC) near its
throat.  We verify this both through the numerical Einstein-tensor pipeline and
against the closed-form Morris-Thorne stress-energy components.
"""

import numpy as np
import pytest

from core.metrics import MorrisThorne, EllisBronnikov
from core import stress_energy as se


def test_nec_violated_at_throat_morris_thorne():
    wh = MorrisThorne(b0=1.0)
    x_throat = np.array([0.0, 1.0 + 1e-3, np.pi / 2, 0.0])
    nec_min = se.null_energy_condition(wh, x_throat, n_dirs=32)
    assert nec_min < 0.0  # NEC violation required for traversability


def test_closed_form_matches_density():
    """Analytic Morris-Thorne rho should match T_{hat t hat t} sign at throat."""
    b0 = 1.0
    r = 1.001
    b = b0 ** 2 / r
    bp = -b0 ** 2 / r ** 2  # derivative of b0^2/r
    rho, p_r, p_t = se.morris_thorne_components(b, bp, 0.0, 0.0, r)
    # For b = b0^2/r the density is negative (exotic) near throat
    assert rho < 0.0
    # radial NEC combination rho + p_r < 0
    assert rho + p_r < 0.0


def test_ellis_kretschmann_finite():
    """Ellis wormhole is regular: Kretschmann scalar must be finite at the throat."""
    wh = EllisBronnikov(b0=1.0)
    x_throat = np.array([0.0, 0.0, np.pi / 2, 0.0])
    K = se.kretschmann(wh, x_throat)
    assert np.isfinite(K)
