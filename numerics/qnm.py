"""Quasinormal modes of the Ellis wormhole (scalar perturbations).

A massless scalar field on the Ellis background separates, in the proper-radial
(tortoise) coordinate x = l, into a Schrodinger-like wave equation

    psi_tt - psi_xx + V_l(x) psi = 0 ,

with the symmetric Poschl-Teller-type barrier (multipole l)

    V_l(x) = l(l+1) / (x^2 + b0^2)  -  b0^2 / (x^2 + b0^2)^2 .

After an initial perturbation the field rings down as a superposition of damped
modes  psi ~ sum A_n exp(-i omega_n t),  Im(omega_n) < 0.  We extract the
fundamental complex frequency from the time-domain signal at a fixed observer:
the real part from the late-time oscillation period, the imaginary part from the
decay of the successive peak amplitudes.

This is a time-domain estimate (good to a few percent for the fundamental mode),
not a high-precision continued-fraction/Leaver solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from numerics.pde_solver import FiniteDifferenceWave, _d2_4th, _d1_4th


def regge_wheeler_potential(x, b0=1.0, l=2):
    """Effective scalar perturbation potential V_l(x) on the Ellis wormhole."""
    denom = x ** 2 + b0 ** 2
    return l * (l + 1) / denom - b0 ** 2 / denom ** 2


@dataclass
class RingdownExtractor:
    b0: float = 1.0
    l: int = 2
    L: float = 60.0
    nx: int = 2401
    cfl: float = 0.4

    def _evolve(self, t_final=120.0, obs_x=15.0):
        """Evolve a Gaussian perturbation; return (t, signal) at observer obs_x."""
        sim = FiniteDifferenceWave(b0=self.b0, L=self.L, nx=self.nx, cfl=self.cfl)
        V = regge_wheeler_potential(sim.l, self.b0, self.l)

        # initial data: a static Gaussian bump away from the throat
        phi = np.exp(-((sim.l - 8.0) ** 2) / (2 * 1.5 ** 2))
        phidot = np.zeros_like(phi)

        def rhs(p, pd):
            lap = _d2_4th(p, sim.dx) + sim.curv * _d1_4th(p, sim.dx) - V * p
            lap[0] = lap[1]; lap[-1] = lap[-2]
            return pd, lap

        obs_i = int(np.argmin(np.abs(sim.l - obs_x)))
        nsteps = int(t_final / sim.dt)
        sig, ts = [], []
        for n in range(nsteps):
            k1 = rhs(phi, phidot)
            k2 = rhs(phi + 0.5 * sim.dt * k1[0], phidot + 0.5 * sim.dt * k1[1])
            k3 = rhs(phi + 0.5 * sim.dt * k2[0], phidot + 0.5 * sim.dt * k2[1])
            k4 = rhs(phi + sim.dt * k3[0], phidot + sim.dt * k3[1])
            phi = phi + (sim.dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            phidot = phidot + (sim.dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            sig.append(phi[obs_i]); ts.append(n * sim.dt)
        return np.array(ts), np.array(sig)

    def fundamental_mode(self, t_final=120.0):
        """Return complex omega = omega_r - i*omega_i (the fundamental QNM estimate)."""
        t, s = self._evolve(t_final=t_final)
        # restrict to the ringdown window (after the initial burst, before tail)
        i0 = int(0.35 * len(t))
        i1 = int(0.85 * len(t))
        tw, sw = t[i0:i1], s[i0:i1]

        # peaks of |signal| -> period (omega_r) and decay (omega_i)
        absid = np.abs(sw)
        peaks = (absid[1:-1] > absid[:-2]) & (absid[1:-1] > absid[2:])
        pk = np.where(peaks)[0] + 1
        if len(pk) < 3:
            return complex(float("nan"), float("nan")), (t, s)
        tp = tw[pk]
        amp = absid[pk]
        omega_r = np.pi / np.mean(np.diff(tp))          # half-period between |peaks|
        # decay rate from log-amplitude slope
        good = amp > 0
        slope = np.polyfit(tp[good], np.log(amp[good]), 1)[0]
        omega_i = -slope
        return complex(omega_r, -abs(omega_i)), (t, s)
