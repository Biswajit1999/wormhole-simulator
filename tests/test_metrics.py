"""Unit tests for metric definitions and basic geometry."""

import numpy as np
import pytest

from core.metrics import MorrisThorne, EllisBronnikov, ChargedWormhole, TeoRotating


@pytest.fixture(params=[
    MorrisThorne(b0=1.0),
    EllisBronnikov(b0=1.0),
    ChargedWormhole(b0=1.0, Q=0.3),
    TeoRotating(b0=1.0, J=0.2),
])
def metric(request):
    return request.param


def test_metric_signature(metric):
    """Metric should have Lorentzian signature (-,+,+,+): one negative eigenvalue."""
    x = np.array([0.0, 3.0, np.pi / 2, 0.0])
    g = metric.components(x)
    eig = np.linalg.eigvalsh(g)
    assert np.sum(eig < 0) == 1
    assert np.sum(eig > 0) == 3


def test_metric_symmetric(metric):
    x = np.array([0.0, 3.0, np.pi / 2, 0.3])
    g = metric.components(x)
    assert np.allclose(g, g.T)


def test_inverse_consistency(metric):
    x = np.array([0.0, 4.0, np.pi / 2, 0.0])
    g = metric.components(x)
    ginv = metric.inverse(x)
    assert np.allclose(g @ ginv, np.eye(4), atol=1e-8)


def test_morris_thorne_throat():
    """Shape function satisfies b(b0) = b0 for the canonical profile."""
    wh = MorrisThorne(b0=1.5)
    assert wh.shape(wh.b0) == pytest.approx(wh.b0)


def test_ellis_areal_radius_minimum():
    """Ellis areal radius is minimized at the throat r = 0, equal to b0."""
    wh = EllisBronnikov(b0=2.0)
    assert wh.embedding_radius(0.0) == pytest.approx(2.0)
    assert wh.embedding_radius(3.0) > wh.embedding_radius(0.0)


def test_christoffel_shape(metric):
    x = np.array([0.0, 3.0, np.pi / 2, 0.0])
    G = metric.christoffel(x)
    assert G.shape == (4, 4, 4)
    # Symmetric in the lower two indices
    assert np.allclose(G, np.transpose(G, (0, 2, 1)), atol=1e-6)
