"""visualization: embedding, Penrose, lensing (curve + full image)."""

from .embed import embedding_profile, plot_embedding_surface
from .penrose import penrose_wormhole
from .raytrace import EquatorialRayTracer, deflection_angle
from .lensing_image import render_lensed_image, save_lensed_image

__all__ = [
    "embedding_profile",
    "plot_embedding_surface",
    "penrose_wormhole",
    "EquatorialRayTracer",
    "deflection_angle",
    "render_lensed_image",
    "save_lensed_image",
]
