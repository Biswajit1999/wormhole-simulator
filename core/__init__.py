"""core: wormhole metrics and general-relativity utilities.

Metrics
-------
MorrisThorne, EllisBronnikov, ChargedWormhole, TeoRotating, ModifiedGravityWormhole

Submodules
----------
backend           differentiation engine (JAX autodiff / Richardson FD)
symmetries        Killing vectors, conserved E, L
stress_energy     curvature, Einstein tensor, NEC/WEC/SEC/DEC, ANEC
thinshell         thin-shell wormholes + stability
semiclassical     Casimir energy, Casimir wormhole, Ford-Roman bound
modified_gravity  power-law / modified-gravity wormholes
"""

from .metrics import (
    Metric,
    MorrisThorne,
    EllisBronnikov,
    ChargedWormhole,
    TeoRotating,
)
from .modified_gravity import ModifiedGravityWormhole

__all__ = [
    "Metric",
    "MorrisThorne",
    "EllisBronnikov",
    "ChargedWormhole",
    "TeoRotating",
    "ModifiedGravityWormhole",
]

__version__ = "0.2.0"
