"""Wormholes in modified gravity (representative families).

In f(R), Gauss-Bonnet and other higher-curvature theories, the *geometric* part
of the field equations contributes effective stress-energy, so a wormhole throat
can be threaded by matter that satisfies the standard energy conditions while the
**effective** (geometry + matter) source violates them.  This module provides a
representative one-parameter family used in such studies and an effective-source
diagnostic.

The geometry is Morris-Thorne in form with a power-law shape function

    b(r) = r0 * (r0 / r)^gamma ,   0 < gamma < 1 ,

so b(r0) = r0 and the flare-out b'(r0) = -gamma < 1 holds.  In GR this requires
exotic matter; in modified gravity the higher-curvature terms can supply the
flare-out, which is the physical interest of the family (Lobo & Oliveira 2009;
Mehdizadeh et al. 2015 for Gauss-Bonnet).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .metrics import Metric


@dataclass
class ModifiedGravityWormhole(Metric):
    r"""Power-law-shape wormhole, b(r) = r0 (r0/r)^gamma, Phi = 0."""

    b0: float = 1.0
    gamma: float = 0.5

    def __post_init__(self):
        if self.b0 <= 0:
            raise ValueError("throat radius b0 must be positive")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must lie in (0, 1) for a valid flare-out")

    def shape(self, r):
        return self.b0 * (self.b0 / r) ** self.gamma

    def shape_prime(self, r):
        return -self.gamma * self.b0 ** (1 + self.gamma) * r ** (-self.gamma - 1)

    def components(self, x: np.ndarray) -> np.ndarray:
        _, r, theta, _ = x
        r = max(abs(r), 1e-9)
        b = self.shape(r)
        d = 1.0 - b / r
        d = d if abs(d) > 1e-12 else 1e-12
        g = np.zeros((4, 4))
        g[0, 0] = -1.0
        g[1, 1] = 1.0 / d
        g[2, 2] = r * r
        g[3, 3] = r * r * np.sin(theta) ** 2
        return g

    def components_jax(self, x):
        import jax.numpy as jnp
        r, theta = x[1], x[2]
        b = self.b0 * (self.b0 / r) ** self.gamma
        grr = 1.0 / (1.0 - b / r)
        return jnp.diag(jnp.array([-1.0, grr, r * r, r * r * jnp.sin(theta) ** 2]))

    def effective_energy_density(self, r):
        """Effective (geometric) energy density rho_eff = b'(r)/(8 pi r^2).

        Negative near the throat (the flare-out), supplied by curvature terms in
        modified gravity rather than by exotic matter.
        """
        return self.shape_prime(r) / (8.0 * np.pi * r ** 2)

    def _params_repr(self) -> str:
        return f"b0={self.b0}, gamma={self.gamma}"
