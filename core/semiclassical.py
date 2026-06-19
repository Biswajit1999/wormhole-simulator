"""Semiclassical ingredients: Casimir energy, Casimir wormholes, quantum bounds.

Traversable wormholes need energy-condition-violating matter.  Quantum field
theory supplies it (the Casimir effect produces a genuinely negative energy
density), but the **quantum inequalities** of Ford & Roman bound how much
negative energy an inertial observer can sample and for how long.  This module
collects the relevant closed forms.

References: Casimir (1948); Ford & Roman, Phys. Rev. D 51, 4277 (1995);
Garattini, Eur. Phys. J. C 79, 951 (2019), "Casimir Wormholes".
"""

from __future__ import annotations

import numpy as np

from .compat import trapezoid


def casimir_energy_density(plate_separation: float) -> float:
    """Negative vacuum energy density between ideal parallel plates (per unit vol).

        rho = - pi^2 / (720 d^4)     (geometrized/natural units, hbar = c = 1).

    Negative for all d > 0 — the prototypical source of exotic matter.
    """
    d = plate_separation
    if d <= 0:
        raise ValueError("plate separation must be positive")
    return -np.pi ** 2 / (720.0 * d ** 4)


def casimir_wormhole_shape(r, r0: float = 1.0) -> float:
    """Shape function of the Garattini (2019) Casimir wormhole.

        b(r) = (r0 / 3) * (1 + 2 r0^2 / r^2),

    which satisfies b(r0) = r0 and the flare-out condition, sourced by the
    Casimir energy density threading the throat.  Representative form.
    """
    return (r0 / 3.0) * (1.0 + 2.0 * r0 ** 2 / r ** 2)


def casimir_wormhole_redshift_prime(r, r0: float = 1.0) -> float:
    """Phi'(r) for the Garattini Casimir wormhole (used in tidal-force checks).

        Phi'(r) = r0^2 / [ r ( r^2 + ... ) ]  -> here the compact representative
        Phi'(r) = r0^2 / ( r (3 r^2 + r0^2) ).
    """
    return r0 ** 2 / (r * (3.0 * r ** 2 + r0 ** 2))


def ford_roman_bound(tau0: float, coefficient: float = 3.0 / (32.0 * np.pi ** 2)) -> float:
    """Magnitude of the Ford-Roman quantum-inequality bound for a massless field.

        integral <T_uu> f(tau) dtau  >=  - coefficient / tau0^4 ,

    with a Lorentzian sampling function of width tau0.  Returns the (positive)
    bound magnitude C / tau0^4; the sampled negative energy may not exceed it.
    """
    if tau0 <= 0:
        raise ValueError("sampling time tau0 must be positive")
    return coefficient / tau0 ** 4


def satisfies_quantum_inequality(energy_density, proper_time, tau0=None) -> dict:
    """Check the Ford-Roman QI for a sampled energy density along a worldline.

    Parameters
    ----------
    energy_density : array of <T_uu> samples along the observer's proper time.
    proper_time : array of proper-time samples (same length).
    tau0 : Lorentzian sampling width; defaults to the signal's timespan / 6.

    Returns a dict with the sampled (weighted) integral, the bound, and a pass flag.
    """
    energy_density = np.asarray(energy_density, float)
    tau = np.asarray(proper_time, float)
    if tau0 is None:
        tau0 = (tau[-1] - tau[0]) / 6.0
    tau_c = 0.5 * (tau[0] + tau[-1])
    f = (tau0 / np.pi) / ((tau - tau_c) ** 2 + tau0 ** 2)  # Lorentzian, unit area
    sampled = float(trapezoid(energy_density * f, tau))
    bound = -ford_roman_bound(tau0)
    return {
        "sampled_integral": sampled,
        "bound": bound,
        "tau0": tau0,
        "satisfied": bool(sampled >= bound),
    }
