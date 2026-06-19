"""Schematic Penrose (conformal) diagram for a traversable wormhole.

A traversable wormhole connects two asymptotically flat regions through a
horizonless throat, so its conformal diagram is two Minkowski diamonds joined
along the timelike throat line (no horizons, unlike the Schwarzschild bridge).
This module draws that schematic with Matplotlib for documentation figures.
"""

from __future__ import annotations

import numpy as np


def penrose_wormhole(ax=None):
    """Draw two Minkowski diamonds joined at a central timelike throat.

    Returns the Matplotlib Axes.  Labels mark null infinity (scri +/-), spatial
    infinity i0 and timelike infinity i+/-, for each asymptotic universe.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    def diamond(cx):
        # Unit conformal diamond centred at x = cx
        pts = np.array([[cx, -1], [cx + 1, 0], [cx, 1], [cx - 1, 0], [cx, -1]])
        ax.plot(pts[:, 0], pts[:, 1], "k-", lw=1.5)
        ax.text(cx + 1.05, 0, r"$i^0$", va="center")
        ax.text(cx, 1.08, r"$i^+$", ha="center")
        ax.text(cx, -1.12, r"$i^-$", ha="center")
        ax.text(cx + 0.55, 0.6, r"$\mathcal{I}^+$", fontsize=9)
        ax.text(cx + 0.55, -0.65, r"$\mathcal{I}^-$", fontsize=9)

    diamond(-1.0)
    diamond(1.0)

    # Throat: the shared timelike line where the two universes meet
    ax.plot([0, 0], [-1, 1], "r--", lw=2.0, label="wormhole throat")
    ax.text(0, 1.18, "throat (horizonless)", ha="center", color="r")

    ax.text(-1.0, -1.45, "Universe A", ha="center")
    ax.text(1.0, -1.45, "Universe B", ha="center")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.7, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Conformal (Penrose) diagram of a traversable wormhole")
    return ax
