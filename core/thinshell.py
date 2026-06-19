"""Thin-shell wormholes and linearized stability (Poisson & Visser 1995).

Cut-and-paste construction: glue two copies of a Schwarzschild exterior
f(a) = 1 - 2M/a at a timelike shell r = a(tau).  The Israel/Lanczos junction
conditions give the static surface energy density and pressure

    sigma0 = -sqrt(f0) / (2 pi a0),
    p0     =  (1 - M/a0) / (4 pi a0 sqrt(f0)),      f0 = 1 - 2M/a0,

with sigma0 < 0 (the shell is exotic).  Shell dynamics follow an energy
equation  a_dot^2 + V(a) = 0  with  V(a) = f(a) - (2 pi a sigma(a))^2, where
sigma(a) evolves by conservation  sigma' = -(2/a)(sigma + p)  closed by a linear
equation of state  p = p0 + beta2 (sigma - sigma0),  beta2 = (dp/dsigma)|_a0
(a "speed of sound squared" parameter).

At the equilibrium radius the static junction conditions enforce V(a0) = 0 and
V'(a0) = 0 automatically; the throat is linearly **stable** iff V''(a0) > 0.
This module evaluates V, V', V'' analytically and classifies stability — the
canonical Poisson-Visser diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ThinShellWormhole:
    M: float = 1.0
    a0: float = 4.0
    beta2: float = 0.5  # (speed of sound)^2 = dp/dsigma at equilibrium

    def __post_init__(self):
        if self.a0 <= 2 * self.M:
            raise ValueError("a0 must exceed the horizon radius 2M")
        self.f0 = 1.0 - 2.0 * self.M / self.a0
        self.sf0 = np.sqrt(self.f0)
        self.sigma0 = -self.sf0 / (2.0 * np.pi * self.a0)
        self.p0 = (1.0 - self.M / self.a0) / (4.0 * np.pi * self.a0 * self.sf0)

    # static-frame derivatives of sigma from conservation + linear EOS
    def _sigma_prime0(self):
        return -(2.0 / self.a0) * (self.sigma0 + self.p0)

    def _sigma_double0(self):
        sp = self._sigma_prime0()
        pp = self.beta2 * sp  # p' = beta2 sigma'
        return -(2.0 / self.a0) * (sp + pp) + (2.0 / self.a0 ** 2) * (self.sigma0 + self.p0)

    def potential_derivatives(self):
        """Return (V, V', V'') at the equilibrium radius a0 (analytic)."""
        a0 = self.a0
        f0p = 2.0 * self.M / a0 ** 2          # f'(a0)
        f0pp = -4.0 * self.M / a0 ** 3         # f''(a0)
        G0 = 2.0 * np.pi * a0 * self.sigma0    # = -sqrt(f0)
        sp = self._sigma_prime0()
        spp = self._sigma_double0()
        Gp = 2.0 * np.pi * (self.sigma0 + a0 * sp)
        Gpp = 2.0 * np.pi * (2.0 * sp + a0 * spp)
        V = self.f0 - G0 ** 2
        Vp = f0p - 2.0 * G0 * Gp
        Vpp = f0pp - 2.0 * (Gp ** 2 + G0 * Gpp)
        return V, Vp, Vpp

    def is_stable(self):
        """Stable iff V''(a0) > 0 (V and V' vanish at equilibrium by construction)."""
        _, _, Vpp = self.potential_derivatives()
        return bool(Vpp > 0)

    def exotic_surface_mass(self):
        """Total shell surface energy 4 pi a0^2 sigma0 (< 0 => exotic)."""
        return 4.0 * np.pi * self.a0 ** 2 * self.sigma0

    def stability_region(self, beta2_grid):
        """V''(a0) and stability flag across a grid of beta2 values."""
        Vpp, stable = [], []
        for b in beta2_grid:
            w = ThinShellWormhole(M=self.M, a0=self.a0, beta2=float(b))
            _, _, vpp = w.potential_derivatives()
            Vpp.append(vpp)
            stable.append(vpp > 0)
        return np.asarray(beta2_grid), np.asarray(Vpp), np.asarray(stable)
