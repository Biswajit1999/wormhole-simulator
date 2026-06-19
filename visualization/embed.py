"""Embedding diagrams for static wormholes.

A constant-t, equatorial (theta = pi/2) slice of a Morris-Thorne wormhole has
the 2-metric  dl^2 = dr^2/(1 - b/r) + r^2 dphi^2,  which can be embedded in
flat 3-space cylindrical coordinates (r, phi, z) with

    dz/dr = +/- [ r/b(r) - 1 ]^{-1/2}.

Integrating gives the classic "tunnel" surface z(r) flaring out from the throat.
This module produces the z(r) profile and a revolved 3D surface; both branches
(z>0 and z<0) are returned so the full two-sheeted wormhole is drawn.
"""

from __future__ import annotations

import numpy as np


def embedding_profile(shape_fn, r0, r_max=6.0, n=400):
    """Compute z(r) for the embedding surface of a shape function b(r).

    Parameters
    ----------
    shape_fn : callable b(r).
    r0 : throat radius (lower integration limit; b(r0) = r0).
    r_max : outer radius.
    n : number of radial samples.

    Returns
    -------
    r, z : arrays.  z is the upper sheet; the lower sheet is -z.
    """
    r = np.linspace(r0 * (1 + 1e-6), r_max, n)
    integrand = np.empty_like(r)
    for i, ri in enumerate(r):
        val = ri / shape_fn(ri) - 1.0
        integrand[i] = 1.0 / np.sqrt(val) if val > 1e-12 else 0.0
    z = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))])
    return r, z


def plot_embedding_surface(metric, r0, r_max=6.0, n_r=120, n_phi=120, ax=None, cmap="viridis"):
    """Render the revolved embedding surface with Matplotlib (returns the Axes3D).

    The metric must expose a ``shape`` attribute/callable or ``embedding_radius``.
    Works for MorrisThorne, ChargedWormhole and EllisBronnikov.
    """
    import matplotlib.pyplot as plt  # local import keeps core deps light

    # Resolve the shape function for the supported metric families
    if hasattr(metric, "shape") and callable(getattr(metric, "shape")):
        shape_fn = metric.shape
        r, z = embedding_profile(shape_fn, r0, r_max, n_r)
    elif hasattr(metric, "embedding_radius"):
        # Ellis: areal radius rho(r)=sqrt(r^2+b0^2); embed using parametric r-coordinate
        rr = np.linspace(-r_max, r_max, n_r)
        rho = np.array([metric.embedding_radius(x) for x in rr])
        # z is just the proper coordinate; use rr as height for a catenoid-like sheet
        r, z = rho, rr
    else:
        raise TypeError("metric has no shape() or embedding_radius() method")

    phi = np.linspace(0, 2 * np.pi, n_phi)
    R, P = np.meshgrid(r, phi)
    Z, _ = np.meshgrid(z, phi)
    X = R * np.cos(P)
    Y = R * np.sin(P)

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax.plot_surface(X, Y, -Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Embedding diagram: {type(metric).__name__}")
    return ax
