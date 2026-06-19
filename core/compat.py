"""Compatibility helpers for supported dependency ranges."""

from __future__ import annotations

import numpy as np


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Use NumPy's trapezoidal integration API across NumPy 1.x and 2.x."""
    integrate = getattr(np, "trapezoid", np.trapz)
    return integrate(y, x=x, dx=dx, axis=axis)
