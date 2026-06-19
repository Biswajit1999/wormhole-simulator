"""Generate figures for the advanced modules.

    python examples/run_advanced.py   ->   figures/
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.thinshell import ThinShellWormhole
from numerics.qnm import RingdownExtractor
from visualization.lensing_image import save_lensed_image

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def demo_lensing():
    save_lensed_image(path=os.path.join(FIG, "lensing_ellis.png"), res=480)
    print("[A] lensed image saved")


def demo_thinshell():
    w = ThinShellWormhole(M=1.0, a0=2.5)
    beta2, vpp, stable = w.stability_region(np.linspace(0, 12, 200))
    plt.figure(figsize=(6, 4))
    plt.axhline(0, color="k", lw=0.7)
    plt.plot(beta2, vpp, "b")
    plt.fill_between(beta2, vpp, 0, where=vpp > 0, color="green", alpha=0.2, label="stable")
    plt.fill_between(beta2, vpp, 0, where=vpp < 0, color="red", alpha=0.2, label="unstable")
    plt.xlabel(r"$\beta^2 = dp/d\sigma$"); plt.ylabel(r"$V''(a_0)$")
    plt.title("Thin-shell wormhole stability (a0=2.5 M)"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "thinshell_stability.png"), dpi=130)
    plt.close("all")
    print("[B] thin-shell stability saved")


def demo_qnm():
    w, (t, s) = RingdownExtractor(b0=1.0, l=2).fundamental_mode(t_final=100.0)
    plt.figure(figsize=(7, 4))
    plt.plot(t, s, "purple", lw=0.8)
    plt.xlabel("t"); plt.ylabel(r"$\phi$ at observer")
    plt.title(f"Ellis ringdown — fundamental omega ~ {w.real:.2f} {w.imag:+.2f}i")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "qnm_ringdown.png"), dpi=130)
    plt.close("all")
    print("[C] QNM ringdown saved")


if __name__ == "__main__":
    demo_lensing()
    demo_thinshell()
    demo_qnm()
    print("upgrade figures complete -> figures/")
