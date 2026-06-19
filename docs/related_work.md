# Related Work and Prior Art

This page is deliberately honest about what already exists. Wormhole visualization
and geodesic integration are **well-trodden** areas; the value of `wormhole-sim` is
in *combination and reproducibility*, not in being the first to render a wormhole.

## Existing open-source / published tools

| Project | Scope | Language | Notes |
|---|---|---|---|
| [Silvera0218 / Real-time 3D Wormhole](https://github.com/Silvera0218/Real-time-3D-Wormhole-from-Visualizing-Interstellar-s-Wormhole-) | Real-time Morris–Thorne flythrough | Python + ModernGL/GLSL | Reverse ray-tracing per pixel in a fragment shader; visualization-only |
| [lennrt / Interstellar-Wormhole-Ray-Tracing](https://github.com/lennrt/Interstellar-Wormhole-Ray-Tracing) | Lensing map (parallelized) | Mathematica | Based on the Thorne et al. *Interstellar* method |
| [sirxemic / wormhole](https://github.com/sirxemic/wormhole) | Ellis-throat curvature demo | JS/WebGL | 2D ray integration using spherical symmetry |
| [javirk / Wormhole-simulation](https://github.com/javirk/Wormhole-simulation) | Curved-ray ray tracing (no gravity) | Python | Educational visual demo |
| [bytebat / gros](https://github.com/bytebat/gros) | General-relativity orbit simulator | Python | Numerical particle trajectories from field equations |
| PyGRO (2025) | Geodesic integration for arbitrary metrics | Python | Closest in spirit to our `numerics.geodesic` |

(Links retrieved June 2026 — see the sources list in the project notes.)

## How `wormhole-sim` is positioned

The projects above are mostly **single-purpose**: a shader visualizer, a lensing
notebook, or a generic geodesic integrator. `wormhole-sim` does not out-render the
shader projects and does not claim to. Its distinct contribution is a **unified,
tested, reproducible research scaffold** that puts the following behind one Python
API and one test suite:

1. **Four metric families** (Morris–Thorne, Ellis–Bronnikov, charged, Teo rotating)
   sharing a common interface, where existing repos typically hard-code one metric.
2. **Geometry + matter together** — curvature *and* stress-energy / energy-condition
   diagnostics, which the visualization-focused projects omit.
3. **Geodesics, embeddings, Penrose diagrams and lensing** all driven by the *same*
   metric objects, so a new metric is immediately visualizable and analyzable.
4. **Engineering for research use** — unit tests asserting conserved/known quantities,
   CI across Python versions, and Docker/Conda reproducibility.

In short: the novelty is the **integration and rigor**, suitable as a base others can
extend and cite, rather than a new rendering technique. Any claim stronger than that
would not survive contact with the prior art listed here.
