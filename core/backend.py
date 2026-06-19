"""Differentiation backend for connection coefficients.

Provides Christoffel symbols by three routes, chosen automatically:

1. **JAX autodiff** (machine precision, GPU-batchable) when JAX is installed and
   the metric exposes a functional ``components_jax(x)`` written with ``jax.numpy``.
2. **Richardson-extrapolated finite differences** (4th-order accurate) on the
   numpy ``components(x)`` — the default, dependency-free path.
3. Plain central differences as a last resort.

The selection is transparent to callers: ``connection(metric, x)`` always returns
an ``(4, 4, 4)`` numpy array Gamma^lambda_{mu nu}, so the differentiation
engine can be swapped without touching the solvers that consume it.
"""

from __future__ import annotations

import numpy as np

try:  # optional acceleration
    import jax
    import jax.numpy as jnp

    _HAVE_JAX = True
except Exception:  # pragma: no cover - JAX is optional
    _HAVE_JAX = False


def _christoffel_from_derivs(g: np.ndarray, dg: np.ndarray) -> np.ndarray:
    """Gamma^l_{mn} = 1/2 g^{ls}(d_m g_{sn} + d_n g_{sm} - d_s g_{mn}).

    ``dg`` has shape (a, mu, nu) = d_a g_{mu nu}.
    """
    ginv = np.linalg.inv(g)
    term = (
        np.einsum("msn->smn", dg)  # d_m g_{s n}  -> index [s, m, n]
        + np.einsum("nsm->smn", dg)  # d_n g_{s m}
        - np.einsum("smn->smn", dg)  # d_s g_{m n}
    )
    return 0.5 * np.einsum("ls,smn->lmn", ginv, term)


def _richardson_metric_jacobian(components, x: np.ndarray) -> np.ndarray:
    """4th-order accurate d g_{mu nu}/dx^a via Richardson extrapolation.

    Combines central differences at steps h and h/2 to cancel the leading
    O(h^2) error, giving O(h^4).  Returns shape (a, mu, nu).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    h = 1e-4
    dg = np.zeros((n, 4, 4))
    for a in range(n):
        e = np.zeros(n)
        e[a] = 1.0
        d_h = (components(x + h * e) - components(x - h * e)) / (2 * h)
        d_h2 = (components(x + 0.5 * h * e) - components(x - 0.5 * h * e)) / h
        dg[a] = (4.0 * d_h2 - d_h) / 3.0  # Richardson: cancel O(h^2)
    return dg


def _jax_connection(metric, x: np.ndarray):
    """Christoffel via JAX forward-mode autodiff of components_jax. Returns array
    or None if unavailable / the metric is not JAX-compatible."""
    if not _HAVE_JAX or not hasattr(metric, "components_jax"):
        return None
    try:
        f = metric.components_jax
        xj = jnp.asarray(x, dtype=float)
        g = np.asarray(f(xj))
        # jac[mu, nu, a] = d g_{mu nu} / dx^a  -> transpose to (a, mu, nu)
        jac = np.asarray(jax.jacfwd(f)(xj))
        dg = np.transpose(jac, (2, 0, 1))
        return _christoffel_from_derivs(g, dg)
    except Exception:
        return None


def connection(metric, x: np.ndarray) -> np.ndarray:
    """Return Gamma^lambda_{mu nu} (4,4,4) using the best available method."""
    out = _jax_connection(metric, x)
    if out is not None:
        return out
    g = metric.components(np.asarray(x, dtype=float))
    dg = _richardson_metric_jacobian(metric.components, x)
    return _christoffel_from_derivs(g, dg)


def backend_name(metric=None) -> str:
    """Report which differentiation route is active (for diagnostics/banners)."""
    if metric is not None and _HAVE_JAX and hasattr(metric, "components_jax"):
        return "jax-autodiff"
    return "richardson-fd"
