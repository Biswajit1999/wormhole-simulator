# Educational & Research Deliverables

A tiered progression from reproducing classic figures to research-grade modules.
Each milestone pairs a demo with a **validation** against a known result.

## Milestone 1 — Beginner demos (weeks 1–4)

| Demo | What it does | Validation |
|---|---|---|
| Embedding diagram | Plot and revolve $z(r)$ for Morris–Thorne; slider on $b_0$ | Reproduce the classic tunnel from the literature |
| Geodesic tutorial | Integrate null/timelike geodesics in the Ellis metric | Null rays satisfy $ds^2=0$; norm drift $<10^{-5}$ |
| Curvature check | Kretschmann scalar across the throat | Finite and bounded for Ellis |

Covered by `examples/run_demos.py` (demos 1, 3) and `tests/test_geodesic.py`.

## Milestone 2 — Intermediate modules (weeks 5–8)

| Demo | What it does | Validation |
|---|---|---|
| Ray-tracing / lensing | Map a star field through the wormhole; locate Einstein rings | Compare deflection curve to James et al. (2015) |
| Energy-condition analyzer | Plot $\rho+p_r$ for arbitrary $b(r),\Phi(r)$; shade NEC-violation region | Sign matches analytic Morris–Thorne at throat |
| Exotic-matter budget | Integrate $\int T_{\hat t\hat t}\,dV$ | Compare to Morris–Thorne estimates |

Covered by `examples/run_demos.py` (demo 2) and `tests/test_energy_conditions.py`.

## Milestone 3 — Advanced research modules (weeks 9–12)

| Demo | What it does | Validation |
|---|---|---|
| Rotating wormhole | Teo metric; frame-dragging $\omega(r)$; equatorial geodesics | Off-diagonal $g_{t\phi}$ sign; ZAMO behaviour |
| Scalar scattering | Wave packet on the wormhole barrier; quasinormal ringing | Energy-drift diagnostic; reflection/transmission |
| Quantum-bound overlay | Evaluate Ford–Roman bound for a given geometry | Reproduce a published bound figure |
| (Optional) dynamical throat | Drive Einstein Toolkit with wormhole initial data | Constraint convergence under refinement |

Rotating geodesics use `core.metrics.TeoRotating`; scattering uses
`numerics.pde_solver.FiniteDifferenceWave`.

## Benchmarks

- **Integrator:** RK45 (adaptive) vs. fixed-step RK4 — cost vs. norm drift.
- **Spectral vs. finite difference:** convergence rate on a smooth static profile.
- **Lensing throughput:** rays/second for SciPy vs. JAX-batched kernels.

## Suggested "good first issues"

- Add a new shape function $b(r)$ (e.g. exponential cut-off) with its embedding test.
- Replace finite-difference Christoffels with a JAX autodiff backend.
- Add a Plotly interactive embedding surface.
- Implement the thin-shell stability potential $V(a)$ and its $V''(a_0)$ test.
