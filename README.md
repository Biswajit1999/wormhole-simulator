# wormhole-sim

![wormhole-sim — relativistic wormhole simulation](cover.png)

A modular, open-source framework for **relativistic wormhole simulation** — metric
definitions, geodesic integration, energy-condition diagnostics, embedding/Penrose
diagrams, gravitational-lensing ray-tracing, thin-shell stability, quasinormal
modes, semiclassical bounds and real-time interactive 3D — aimed at graduate and
postdoctoral researchers in general and numerical relativity.

> Status: research preview (v0.2.0). MIT licensed. Python-first. 32 unit tests.

---

## What this is (and how it differs from existing tools)

Several wormhole codes exist, but most are single-purpose (a shader visualizer, a
lensing notebook, a generic geodesic integrator; see
[`docs/related_work.md`](docs/related_work.md)). `wormhole-sim` unifies, in one
tested, reproducible package:

- **five metric families** — Morris–Thorne, Ellis–Bronnikov, charged, Teo rotating,
  and a modified-gravity power-law family;
- a **metric-agnostic** numerical core (autodiff/Richardson Christoffels, Riemann,
  Ricci, Einstein tensor) — a new metric needs only one method;
- **energy-condition analysis** (NEC/WEC/SEC/DEC, ANEC, exotic-matter quantifier)
  alongside the geometry;
- **geodesics, conserved quantities, embeddings, Penrose diagrams, lensing images,
  thin-shell stability and quasinormal modes** sharing the same metric objects;
- **interactive 3D** (WebGL/three.js) and apps (Streamlit, Blender);
- **tests + CI + Docker/Conda** for reproducibility.

---

## Quick start

```bash
git clone <your-fork-url> wormhole-sim
cd wormhole-sim
conda env create -f environment.yml        # or: pip install -e .
conda activate wormhole-sim
pytest -q                                   # 32 unit tests
python examples/run_demos.py               # embedding, NEC, lensing curve, wave
python examples/run_advanced.py            # lensed image, thin-shell stability, ringdown
```

Minimal example:

```python
import numpy as np
from core.metrics import EllisBronnikov
from numerics.geodesic import GeodesicSolver, null_initial_velocity
from core import symmetries as sym

wh = EllisBronnikov(b0=1.0)
x0 = np.array([0.0, 6.0, np.pi/2, 0.0])
u0 = null_initial_velocity(wh, x0, [-1.0, 0.0, 0.05])
res = GeodesicSolver(wh).integrate(x0, u0, affine_span=(0, 20))
print("throat crossed:", res.coords[:, 1].min() < 0)
print("conservation drift:", sym.monitor_conservation(wh, res))  # E, L, norm ~1e-8
```

---

## Interactive 3D viewers (mouse: rotate · scroll: zoom · animated)

Open these `.html` files directly in a browser — no build step:

- **[`web/embedding3d.html`](web/embedding3d.html)** — orbit the wormhole embedding
  surface in 3D. **Drag** rotates up/down/left/right, **scroll** zooms, auto-rotate
  animates. Switch between Morris–Thorne and Ellis, vary the throat radius live.
- **[`web/realtime.html`](web/realtime.html)** — a real-time GLSL geodesic ray
  tracer. **Drag** to look around, **scroll** to fly in/out through the throat,
  with an automatic fly-through animation.
- **[`web/index.html`](web/index.html)** — static landing page with the theory,
  figures and Mermaid diagrams.
- **`web/app.py`** — `streamlit run web/app.py` for a parameter dashboard.

---

## Capabilities

| Area | Methods / outputs |
|---|---|
| Differentiation | JAX-autodiff or 4th-order Richardson connections |
| Geodesics | adaptive RK45/RK4 integrator, conserved E & L, equatorial-orbit reduction |
| Energy conditions | NEC/WEC/SEC/DEC, ANEC integral, exotic-matter quantifier |
| Thin shells | Israel junction conditions, linearized V(a) stability |
| Rotation | Teo wormhole: ZAMO, ergoregion, frame dragging |
| Lensing | exact deflection curves and full backward-ray-traced sky images |
| Perturbations | Regge–Wheeler potential and quasinormal-mode extraction |
| Semiclassical | Casimir energy, Casimir wormhole, Ford–Roman quantum inequality |
| Modified gravity | power-law wormhole family; Einstein Toolkit `.par` bridge |
| Visualization | embedding, Penrose, interactive WebGL 3D, Streamlit, Blender |

---

## Repository layout

```
wormhole-sim/
├── core/                      # metrics + GR utilities
│   ├── metrics.py             #   MT, Ellis, charged, Teo rotating
│   ├── backend.py             #   autodiff / Richardson Christoffels
│   ├── symmetries.py          #   conserved E, L
│   ├── stress_energy.py       #   Einstein tensor, NEC/WEC/SEC/DEC/ANEC
│   ├── thinshell.py           #   thin-shell stability
│   ├── semiclassical.py       #   Casimir, Ford–Roman
│   └── modified_gravity.py    #   modified-gravity wormholes
├── numerics/                  # solvers
│   ├── geodesic.py            #   RK45/RK4 geodesic integrator
│   ├── orbits.py              #   conserved-quantity equatorial orbits
│   ├── pde_solver.py          #   finite-difference scalar wave
│   ├── spectral.py            #   Chebyshev pseudospectral
│   ├── qnm.py                 #   quasinormal-mode extraction
│   └── et_bridge.py           #   Einstein Toolkit .par generator
├── visualization/             # figures
│   ├── embed.py · penrose.py · raytrace.py
│   ├── lensing_image.py       #   full lensed-sky renderer
│   └── blender_render.py      #   photoreal Blender pipeline
├── web/                       # index.html, realtime.html, embedding3d.html, app.py
├── examples/                  # run_demos.py, run_advanced.py
├── tests/                     # 32 pytest unit tests
├── docs/                      # theory, methods, references (LaTeX + Mermaid)
├── .github/workflows/ci.yml   # CI
└── Dockerfile · environment.yml · pyproject.toml · LICENSE (MIT)
```

---

## Conventions

Signature **(−,+,+,+)**, geometrized units **G = c = 1**, coordinates **(t, r, θ, φ)**.
A new metric needs only a `components(x)` method (optionally `components_jax(x)` for
autodiff); curvature, geodesics, energy conditions and embeddings then work
automatically.

## Citing

See [`CITATION.cff`](CITATION.cff) and the primary references in
[`docs/mathematical_foundations.md`](docs/mathematical_foundations.md)
(Morris & Thorne 1988; Ellis 1973; Teo 1998; Ford & Roman 1995).

## License

MIT — see [`LICENSE`](LICENSE).
