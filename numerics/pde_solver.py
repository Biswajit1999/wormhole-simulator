"""Finite-difference evolution of a test scalar field on a wormhole background.

We evolve a massless Klein-Gordon field phi on the static Ellis/Morris-Thorne
geometry using the proper radial coordinate l in (-L, L), where the wormhole
maps naturally onto the whole real line.  In that chart the wave operator
reduces to a 1+1D equation with an areal-radius potential:

    phi_tt = phi_ll + (2 r'(l)/r(l)) phi_l - (V_eff) phi

For the Ellis metric r(l) = sqrt(l^2 + b0^2), so r'(l) = l / r and the
"reflectionless" Poschl-Teller-like barrier appears naturally.  We use 4th-order
central differences in space and an explicit RK4 (method of lines) in time, with
outgoing (Sommerfeld) boundary conditions.

This is a pedagogical demonstrator of wave scattering / quasi-normal ringing on
a wormhole background, not a full Einstein-equation evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _d2_4th(u, dx):
    """4th-order central second derivative with one-sided edges."""
    d2 = np.empty_like(u)
    d2[2:-2] = (
        -u[:-4] + 16 * u[1:-3] - 30 * u[2:-2] + 16 * u[3:-1] - u[4:]
    ) / (12 * dx * dx)
    # 2nd-order at the two boundary-adjacent points
    d2[1] = (u[0] - 2 * u[1] + u[2]) / dx ** 2
    d2[-2] = (u[-3] - 2 * u[-2] + u[-1]) / dx ** 2
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2


def _d1_4th(u, dx):
    d1 = np.empty_like(u)
    d1[2:-2] = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dx)
    d1[1] = (u[2] - u[0]) / (2 * dx)
    d1[-2] = (u[-1] - u[-3]) / (2 * dx)
    d1[0] = (u[1] - u[0]) / dx
    d1[-1] = (u[-1] - u[-2]) / dx
    return d1


@dataclass
class FiniteDifferenceWave:
    """Scalar wave on an Ellis wormhole proper-radial chart l in [-L, L]."""

    b0: float = 1.0
    L: float = 40.0
    nx: int = 1601
    cfl: float = 0.4

    def __post_init__(self):
        self.l = np.linspace(-self.L, self.L, self.nx)
        self.dx = self.l[1] - self.l[0]
        self.dt = self.cfl * self.dx
        self.r = np.sqrt(self.l ** 2 + self.b0 ** 2)
        self.rprime = self.l / self.r            # dr/dl
        self.curv = 2.0 * self.rprime / self.r   # first-derivative coupling

    def gaussian_pulse(self, center=-10.0, width=2.0, k=2.0):
        """Initial data: an in-going Gaussian wave packet."""
        env = np.exp(-((self.l - center) ** 2) / (2 * width ** 2))
        phi = env * np.cos(k * self.l)
        # time derivative for a right-moving packet: phi_t = -phi_l
        phidot = -_d1_4th(phi, self.dx)
        return phi, phidot

    def _rhs(self, phi, phidot):
        lap = _d2_4th(phi, self.dx) + self.curv * _d1_4th(phi, self.dx)
        # Sommerfeld outgoing edges
        lap[0] = lap[1]
        lap[-1] = lap[-2]
        return phidot, lap

    def evolve(self, phi, phidot, t_final=40.0, record_every=20):
        """RK4 method-of-lines time integration. Returns (times, snapshots)."""
        nsteps = int(t_final / self.dt)
        snaps, times = [], []
        for n in range(nsteps):
            k1 = self._rhs(phi, phidot)
            k2 = self._rhs(phi + 0.5 * self.dt * k1[0], phidot + 0.5 * self.dt * k1[1])
            k3 = self._rhs(phi + 0.5 * self.dt * k2[0], phidot + 0.5 * self.dt * k2[1])
            k4 = self._rhs(phi + self.dt * k3[0], phidot + self.dt * k3[1])
            phi = phi + (self.dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            phidot = phidot + (self.dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if n % record_every == 0:
                snaps.append(phi.copy())
                times.append(n * self.dt)
        return np.array(times), np.array(snaps)

    def energy(self, phi, phidot):
        """Discrete field energy integral( phi_t^2 + phi_l^2 ) dl (conserved diag.)."""
        phil = _d1_4th(phi, self.dx)
        return float(np.trapz(phidot ** 2 + phil ** 2, self.l))
