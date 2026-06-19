# Contributing to wormhole-sim

Thanks for your interest in improving the framework. This project targets
researchers, so correctness and reproducibility take priority over feature count.

## Development setup

```bash
conda env create -f environment.yml
conda activate wormhole-sim
pip install -e ".[dev]"
pytest -q
```

## Coding conventions

- **Style:** PEP 8, enforced with `ruff` and `black` (line length 100).
- **Type hints** on public functions; NumPy-style docstrings.
- **Physics conventions:** signature (−,+,+,+), units G = c = 1, coordinates
  (t, r, θ, φ). State these explicitly in any new module docstring.
- Keep heavy/optional dependencies (JAX, Mayavi, Plotly) behind local imports so
  the core package stays lightweight.

## Adding a new wormhole metric

1. Subclass `core.metrics.Metric` and implement `components(x)` returning the
   4×4 covariant metric. That is the only required method — Christoffel symbols,
   curvature and geodesics work automatically through finite differencing.
2. Add the class to `core/__init__.py` and its `__all__`.
3. Add a parametrization to the `metric` fixture in `tests/test_metrics.py` so it
   inherits the signature/inverse/Christoffel checks.
4. If a closed-form stress-energy or embedding exists, add it and a cross-check test.

## Tests

- Every physics change needs a test asserting a **conserved or known quantity**
  (geodesic norm drift, energy-condition sign at the throat, an analytic limit).
- Run `pytest -q` locally before opening a PR; CI must pass.

## Pull requests

- Branch from `main`, keep PRs focused, reference any related issue.
- Describe the physics/numerics rationale, not just the code change.
- Include a figure or numeric before/after when changing visualization or solvers.

## Reporting issues

Use the templates under `.github/ISSUE_TEMPLATE/`. For numerical discrepancies,
include the metric parameters, integrator tolerances, and the observed vs. expected
values.

## Code of Conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
