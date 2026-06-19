# Numerical Methods

Simulating wormholes means (a) integrating test-particle and photon **geodesics**
on a fixed background, and optionally (b) evolving **fields or the metric itself**.
This framework focuses on (a) with a metric-agnostic core, and provides
demonstrators for (b).

---

## 1. Geodesic integration

Geodesics obey

$$
\frac{d^2x^\mu}{d\lambda^2} + \Gamma^\mu_{\nu\rho}\frac{dx^\nu}{d\lambda}\frac{dx^\rho}{d\lambda} = 0,
$$

solved as a first-order system in $(x^\mu, u^\mu)$ with $u^\mu = dx^\mu/d\lambda$:

$$
\frac{dx^\mu}{d\lambda}=u^\mu,\qquad \frac{du^\mu}{d\lambda}=-\Gamma^\mu_{\nu\rho}u^\nu u^\rho.
$$

Christoffel symbols are obtained by central finite differencing of the metric, so
**any** metric works without hand-deriving connections:

$$
\Gamma^{\lambda}_{\mu\nu} = \tfrac12 g^{\lambda\sigma}\left(\partial_\mu g_{\sigma\nu} + \partial_\nu g_{\sigma\mu} - \partial_\sigma g_{\mu\nu}\right).
$$

The integrator (`numerics.geodesic.GeodesicSolver`) defaults to adaptive
Dormand–Prince (SciPy `RK45`) and tracks the conserved norm
$\varepsilon = g_{\mu\nu}u^\mu u^\nu$ ($0$ for photons, $-1$ for unit timelike
particles) as the quality metric. A fixed-step RK4 fallback mirrors the pseudocode:

```text
for each ray:
    set initial (x^mu, u^mu)             # u null:  g_{mu nu} u^mu u^nu = 0
    for n = 1..N:
        Gamma = christoffel(metric, x)   # finite-difference connection
        k1 = f(x, u)
        k2 = f(x + h/2 k1_x, u + h/2 k1_u)
        k3 = f(x + h/2 k2_x, u + h/2 k2_u)
        k4 = f(x + h   k3_x, u + h   k3_u)
        x, u += (h/6)(k1 + 2 k2 + 2 k3 + k4)
        store (x, u)
# f(x,u) = ( u ,  -Gamma^mu_{a b} u^a u^b )
```

For a slightly off-axis Ellis photon the measured norm drift is $\lesssim 10^{-6}$
over an affine span of 20 (`tests/test_geodesic.py`).

---

## 2. PDE and time evolution

For dynamical studies (throat collapse, scalar scattering, quasinormal ringing) the
framework ships a **method-of-lines** scalar-wave solver
(`numerics.pde_solver.FiniteDifferenceWave`): 4th-order central spatial stencils plus
RK4 time stepping on the proper-radial chart $\ell\in[-L,L]$, with Sommerfeld
outgoing boundaries and a conserved discrete energy diagnostic. On the Ellis
background the effective potential is the reflectionless Pöschl–Teller-type barrier.

Full **Einstein-equation** evolution (BSSN / 3+1) is out of scope for the pure-Python
core; for that, drive an external solver such as the **Einstein Toolkit**
(Cactus/Carpet with AMR) [ET†L54-L62] using wormhole initial data, and post-process
trajectories here.

---

## 3. Spectral methods

For smooth static profiles, Chebyshev pseudospectral collocation
(`numerics.spectral`) gives exponential convergence and is well suited to the
boundary-value problems of static field equations. The module supplies
Chebyshev–Gauss–Lobatto nodes, the differentiation matrix (Trefethen 2000), and a
Dirichlet BVP demonstrator.

---

## 4. Method comparison

| Method | Advantages | Drawbacks | Notable libraries |
|---|---|---|---|
| Finite difference | Simple, local stencils | Low order; large grids | Einstein Toolkit (Cactus/Carpet) [ET†L54-L62], FiPy |
| Spectral (Chebyshev/Fourier) | Exponential convergence for smooth solutions | Global coupling; hard to AMR | SpEC, NRPy, this package's `spectral.py` |
| Finite element | Flexible/AMR-friendly meshes | Complex for full GR | FEniCS, deal.II |
| Adaptive mesh refinement | Resolves steep throat gradients | Mesh-management overhead | Carpet (Einstein Toolkit), AMReX |
| Geodesic ODE | Lightweight, 1D-per-ray | Stiff near singularities | This package, PyGRO (2025) [PyGRO†L51-L60] |

## 5. Recommended libraries

| Need | Library |
|---|---|
| Array math, ODE integration | **NumPy**, **SciPy** (`solve_ivp`) |
| Autodiff connections / GPU rays | **JAX** (replace finite-diff Christoffels with `jax.jacobian`) |
| Neural / differentiable solvers | **PyTorch** |
| Full numerical relativity, AMR | **Einstein Toolkit** (Cactus/Carpet) [ET†L54-L62] |
| ODE/PDE suites (Julia) | DifferentialEquations.jl, Trixi.jl |
| Spectral GR | NRPy, SpEC |

> **JAX note.** Because the core only needs `components(x)`, swapping the
> finite-difference `christoffel` for a `jax.jacfwd` of the metric upgrades the whole
> stack to machine-precision, GPU-batched connections with no change to the solvers.

## References
- **[ET†L54-L62]** Einstein Toolkit — open software for relativistic astrophysics
  (Cactus framework, Carpet AMR driver). https://einsteintoolkit.org
- **[PyGRO†L51-L60]** PyGRO: a Python geodesic integrator for arbitrary metrics (2025).
