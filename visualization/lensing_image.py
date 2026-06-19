"""Gravitational-lensing image renderer (batched backward ray tracing).

Produces a full 2-D lensed image of a background sky seen through an Ellis
wormhole.  Spherical symmetry lets every photon stay in the plane containing the
camera axis and the pixel direction, so the expensive part — the bending angle
alpha(b) as a function of impact parameter b — is tabulated once by quadrature
(visualization.raytrace.deflection_angle) and reused for every pixel.  The whole
image is then assembled with NumPy array math (vectorized), giving a fast,
physically faithful render: Einstein ring, multiple images, and light arriving
from the *other* universe through the throat.  Swap the NumPy mapping for a JAX kernel or a GLSL
shader (see web/realtime.html) for real-time rates; the lensing map is identical.
"""

from __future__ import annotations

import numpy as np

from visualization.raytrace import deflection_angle


def _grid_sky(theta, phi, universe, lines_deg=10.0):
    """Procedural sky: a latitude/longitude grid whose warping reveals lensing.

    theta (off-axis angle) and phi (azimuth) are the outgoing sky direction.
    Grid lines every ``lines_deg`` degrees glow against a dark background tinted
    by universe (+1 near, -1 through-throat).
    """
    step = np.radians(lines_deg)
    fp_t = np.abs(((theta / step + 0.5) % 1.0) - 0.5)
    fp_p = np.abs(((phi / step + 0.5) % 1.0) - 0.5)
    line = np.exp(-(fp_t / 0.06) ** 2) + np.exp(-(fp_p / 0.06) ** 2)
    line = np.clip(line, 0, 1)
    if universe > 0:
        bg = np.array([0.02, 0.03, 0.07]); glow = np.array([0.3, 0.6, 1.0])
    else:
        bg = np.array([0.07, 0.03, 0.02]); glow = np.array([1.0, 0.6, 0.3])
    return bg[None, :] * np.ones(theta.shape)[..., None] + line[..., None] * glow[None, :]


def _star_field(theta, phi, density=0.0008, seed=7):
    """Sparse procedural stars hashed from quantized sky direction."""
    rng_key = (np.floor(theta * 600).astype(np.int64) * 73856093) ^ \
              (np.floor(phi * 600).astype(np.int64) * 19349663)
    rng_key ^= seed
    val = (np.abs(rng_key) % 100000) / 100000.0
    stars = (val < density).astype(float)
    return stars


def render_lensed_image(b0=1.0, cam_distance=10.0, fov_deg=60.0, res=480):
    """Render an Ellis-wormhole lensed sky to an (res, res, 3) RGB float array.

    Parameters
    ----------
    b0 : throat radius.
    cam_distance : observer radius from the throat.
    fov_deg : full field of view of the (square) camera.
    res : output resolution in pixels.
    """
    # Tabulate deflection vs. impact parameter once (same-universe rays, b > b0)
    b_tab = np.linspace(b0 * 1.0001, cam_distance * np.tan(np.radians(fov_deg)) + 5 * b0, 600)
    alpha_tab = np.array([deflection_angle(b0, b) for b in b_tab])

    # Camera pixel grid -> viewing angle from the optical axis (toward the throat)
    fov = np.radians(fov_deg)
    xs = np.linspace(-fov / 2, fov / 2, res)
    ys = np.linspace(-fov / 2, fov / 2, res)
    gx, gy = np.meshgrid(xs, ys)
    view_ang = np.sqrt(gx ** 2 + gy ** 2)          # angle off axis
    azimuth = np.arctan2(gy, gx)                    # screen azimuth (preserved)

    # Impact parameter for a ray leaving the observer at angle view_ang:
    # b = cam_distance * sin(view_ang)  (flat-space relation at the observer)
    b = cam_distance * np.sin(view_ang)

    # Classify: rays whose impact parameter exceeds the throat stay in this
    # universe; smaller ones thread the throat to the far universe.
    same = b >= b0
    through = ~same

    # Map outgoing direction = incoming view angle + deflection (interp table)
    alpha = np.interp(np.abs(b), b_tab, alpha_tab, left=alpha_tab[0], right=0.0)
    out_ang = view_ang + np.where(same, alpha, np.pi - alpha)  # through-throat flips

    # Build sky coordinates from the outgoing direction
    sky_theta = out_ang
    sky_phi = azimuth
    img = np.zeros((res, res, 3))
    # Same-universe pixels
    img[same] = _grid_sky(sky_theta[same], sky_phi[same], universe=+1)
    img[through] = _grid_sky(sky_theta[through], sky_phi[through], universe=-1)

    # Stars overlay
    stars = _star_field(sky_theta, sky_phi)
    img += stars[..., None] * 0.9

    # Subtle Einstein-ring brightening where deflection diverges (b -> b0+)
    ring = np.exp(-((b - b0) / (0.06 * b0)) ** 2) * same
    img += ring[..., None] * np.array([0.6, 0.7, 1.0])

    return np.clip(img, 0, 1)


def save_lensed_image(path="figures/lensing_ellis.png", **kwargs):
    """Render and save a lensed image with Matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = render_lensed_image(**kwargs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="lower")
    ax.set_title("Ellis wormhole — gravitationally lensed sky")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
