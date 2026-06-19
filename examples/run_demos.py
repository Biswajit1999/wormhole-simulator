"""End-to-end demonstration script.

Runs the milestone demos and saves figures to ../figures/.  Execute from the
repository root:

    python examples/run_demos.py

Demos
-----
1. Embedding diagram of a Morris-Thorne wormhole.
2. NEC-violation profile rho + p_r across the throat.
3. Ellis-wormhole light deflection / Einstein-ring curve.
4. Scalar wave packet scattering off the wormhole potential barrier.
"""

import os
import sys

# Allow running directly (python examples/run_demos.py) by adding the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from core.metrics import MorrisThorne
from core import stress_energy as se
from visualization.embed import plot_embedding_surface
from visualization.raytrace import demo_einstein_ring
from numerics.pde_solver import FiniteDifferenceWave

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def demo_embedding():
    wh = MorrisThorne(b0=1.0)
    plot_embedding_surface(wh, r0=wh.b0, r_max=5.0)
    plt.savefig(os.path.join(FIGDIR, "embedding_morris_thorne.png"), dpi=130)
    plt.close("all")
    print("[1] embedding diagram saved")


def demo_nec():
    wh = MorrisThorne(b0=1.0)
    rs = np.linspace(1.001, 4.0, 60)
    rho_pr = []
    for r in rs:
        b = wh.shape(r)
        bp = -wh.b0 ** 2 / r ** 2
        rho, p_r, _ = se.morris_thorne_components(b, bp, 0.0, 0.0, r)
        rho_pr.append(rho + p_r)
    plt.figure(figsize=(6, 4))
    plt.axhline(0, color="k", lw=0.7)
    plt.plot(rs, rho_pr, "b-")
    plt.fill_between(rs, rho_pr, 0, where=np.array(rho_pr) < 0, color="red", alpha=0.2,
                     label="NEC violated")
    plt.xlabel("r")
    plt.ylabel(r"$\rho + p_r$")
    plt.legend()
    plt.title("Null energy condition across the throat")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "nec_profile.png"), dpi=130)
    plt.close("all")
    print("[2] NEC profile saved")


def demo_lensing():
    rhos, angles = demo_einstein_ring()
    plt.figure(figsize=(6, 4))
    plt.plot(rhos, np.degrees(angles), "g-")
    plt.xlabel("impact parameter $\\rho$")
    plt.ylabel("deflection angle (deg)")
    plt.title("Ellis wormhole light deflection")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "ellis_deflection.png"), dpi=130)
    plt.close("all")
    print("[3] deflection curve saved")


def demo_wave():
    sim = FiniteDifferenceWave(b0=1.0, L=40.0, nx=1201)
    phi, phidot = sim.gaussian_pulse(center=-12.0, width=2.0, k=2.0)
    e0 = sim.energy(phi, phidot)
    times, snaps = sim.evolve(phi, phidot, t_final=30.0, record_every=40)
    plt.figure(figsize=(7, 4))
    for t, s in zip(times[::3], snaps[::3]):
        plt.plot(sim.l, s + 0.0 * t, lw=0.8)
    plt.xlabel("proper radius $l$")
    plt.ylabel(r"$\phi$")
    plt.title("Scalar wave scattering on the wormhole barrier")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "wave_scattering.png"), dpi=130)
    plt.close("all")
    drift = abs(sim.energy(snaps[-1], phidot) - e0) / e0
    print("[4] wave demo saved (relative energy drift: %.2e)" % drift)


if __name__ == "__main__":
    demo_embedding()
    demo_nec()
    demo_lensing()
    demo_wave()
    print("All demos complete -> figures/")
