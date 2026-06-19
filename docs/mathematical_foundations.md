# Mathematical Foundations

Conventions: signature $(-,+,+,+)$, geometrized units $G=c=1$, coordinates
$(t,r,\theta,\phi)$. Einstein's equations read $G_{\mu\nu}=8\pi T_{\mu\nu}$.

---

## 1. Static spherically symmetric (Morris–Thorne)

The canonical traversable-wormhole line element is

$$
ds^2 = -e^{2\Phi(r)}\,dt^2 + \frac{dr^2}{1-b(r)/r} + r^2\left(d\theta^2 + \sin^2\theta\,d\phi^2\right),
$$

with two free functions: the **redshift function** $\Phi(r)$ (finite everywhere so
there is no horizon) and the **shape function** $b(r)$. The throat sits at $r=r_0$
where $b(r_0)=r_0$. Defining the proper radial distance

$$
\ell(r) = \pm\int_{r_0}^{r}\frac{dr'}{\sqrt{1-b(r')/r'}},
$$

$\ell$ runs over $(-\infty,\infty)$ and is regular at the throat.

### Flare-out and the Einstein tensor

In the static orthonormal frame the Einstein tensor is diagonal, giving energy
density $\rho$, radial pressure $p_r$, and transverse pressure $p_t$:

$$
\rho = \frac{b'}{8\pi r^2}, \qquad
p_r = \frac{1}{8\pi}\left[-\frac{b}{r^3} + 2\left(1-\frac{b}{r}\right)\frac{\Phi'}{r}\right],
$$
$$
p_t = \frac{1}{8\pi}\left(1-\frac{b}{r}\right)\!\left[\Phi'' + \Phi'^2 - \frac{b'r-b}{2r(r-b)}\Phi' - \frac{b'r-b}{2r^2(r-b)} + \frac{\Phi'}{r}\right].
$$

The **flare-out condition** — that the embedded throat opens outward — is
$b'(r_0) < 1$, equivalently $\dfrac{b-b'r}{b^2} > 0$ at $r_0$. Combined with the
field equations this forces, at the throat,

$$
\rho + p_r = \frac{1}{8\pi}\frac{b'r - b}{r^3} < 0,
$$

i.e. a **violation of the null energy condition** $T_{\mu\nu}k^\mu k^\nu \ge 0$
[MT88†L223-L231]. This is the central, unavoidable feature of classical
traversable wormholes. (Implemented in `core.stress_energy.morris_thorne_components`
and validated in `tests/test_energy_conditions.py`.)

---

## 2. Ellis–Bronnikov drainhole

Setting $\Phi=0$ and $b(r)=b_0^2/r$ yields the massless **Ellis–Bronnikov**
wormhole [EB73†L217-L224]:

$$
ds^2 = -dt^2 + dr^2 + (r^2 + b_0^2)\left(d\theta^2 + \sin^2\theta\,d\phi^2\right),
\qquad r\in(-\infty,\infty).
$$

The areal radius $\rho(r)=\sqrt{r^2+b_0^2}$ has a minimum $b_0$ at the throat
$r=0$; the spatial geometry is a catenoid. The source is a **phantom scalar field**
(negative kinetic term). The Ellis metric is geodesically complete and curvature is
bounded everywhere — its Kretschmann scalar is finite at the throat
(`tests/test_energy_conditions.py::test_ellis_kretschmann_finite`).

Because of spherical symmetry, equatorial photon orbits reduce to a single
quadrature; the exact bending angle for impact parameter $\rho$ is

$$
\alpha(\rho) = 2\int_{r_{\min}}^{\infty}\frac{dr}{\sqrt{(r^2+b_0^2)\big[(r^2+b_0^2)/\rho^2 - 1\big]}} - \pi,
$$

implemented in `visualization.raytrace.deflection_angle`.

---

## 3. Charged wormhole

Adding an electric field $F_{tr}$ to the Morris–Thorne ansatz and solving the
Einstein–Maxwell system gives a Reissner–Nordström-like throat. A convenient closed
form (Kim & Lee 2001; Lemos & Lobo 2005) keeps $\Phi=0$ with a charge-modified
shape function

$$
b(r) = r_0 + \frac{Q^2}{r_0} - \frac{Q^2}{r},
$$

so $b(r_0)=r_0$ is preserved while the charge $Q$ stiffens the throat and shifts the
energy-condition–violating region. Implemented as `core.metrics.ChargedWormhole`.

---

## 4. Rotating (Teo) wormhole

Teo (1998) generalized Morris–Thorne to a stationary, axisymmetric geometry
[Teo98†]:

$$
ds^2 = -N^2\,dt^2 + \frac{dr^2}{1-b/r} + r^2 K^2\left[d\theta^2 + \sin^2\theta\,(d\phi - \omega\,dt)^2\right],
$$

where $N$ (lapse), $K$ (areal factor), $b$ (shape) and $\omega$ (angular velocity of
frame dragging) depend on $r$ (and in general $\theta$). The package uses the
regular, asymptotically flat illustrative choice $N=K=1$, $b=b_0^2/r$,
$\omega = 2J/r^3$ (`core.metrics.TeoRotating`), which captures **frame dragging**
through the off-diagonal $g_{t\phi}=-r^2\sin^2\theta\,\omega$ while remaining
analytic. Stability analysis for rotating throats must account for the dragging of
inertial frames.

---

## 5. Energy conditions

For a null vector $k^\mu$ and timelike observer $u^\mu$:

| Condition | Statement |
|---|---|
| Null (NEC) | $T_{\mu\nu}k^\mu k^\nu \ge 0$ |
| Weak (WEC) | $T_{\mu\nu}u^\mu u^\nu \ge 0$ (i.e. $\rho\ge 0$) and NEC |
| Strong (SEC) | $\left(T_{\mu\nu}-\tfrac12 T g_{\mu\nu}\right)u^\mu u^\nu \ge 0$ |
| Dominant (DEC) | WEC and energy flux non-spacelike |

Traversable wormholes violate at least the NEC at the throat. The framework samples
$T_{\mu\nu}k^\mu k^\nu$ over null directions (`null_energy_condition`) and reports the
minimum; a negative value flags violation.

### Quantum inequalities

Semiclassically, negative energy is allowed but constrained. The Ford–Roman bound
limits the time-averaged energy density seen by an inertial observer:

$$
\int_{-\infty}^{\infty}\langle T_{\mu\nu}u^\mu u^\nu\rangle\,\frac{\tau_0/\pi}{\tau^2+\tau_0^2}\,d\tau \;\ge\; -\frac{C}{\tau_0^4},
$$

with $C=\mathcal{O}(1)$ and $\tau_0$ the sampling time [FR95†L172-L179]. This is the
quantitative obstacle to building macroscopic traversable wormholes from quantum
fields, and motivates Casimir-energy constructions (Sushkov 2000; Garattini 2006).

---

## 6. Thin-shell wormholes and junction conditions

Visser's construction glues two copies of an exterior spacetime at a timelike shell
$r=a(\tau)$, confining the exotic matter to the shell. The Israel/Lanczos junction
conditions give the surface energy density $\sigma$ and pressure $p$:

$$
\sigma = -\frac{1}{4\pi a}\left[\sqrt{1-\frac{b(a)}{a}+\dot a^2}\,\right]^{+}_{-},
\qquad
p = \frac{1}{8\pi a}\left[\frac{(1+\dot a^2 + a\ddot a - \ldots)}{\sqrt{\,\cdots\,}}\right]^{+}_{-}.
$$

Linearizing $a(\tau)=a_0+\delta a$ about a static radius, stability requires the
effective potential to satisfy $V''(a_0)>0$ (Poisson & Visser 1995). The framework's
geometry layer provides the ingredients ($b$, $b'$, metric functions) needed to
assemble this potential.

---

## 7. Boundary conditions and assumptions

- **Asymptotic flatness:** $\Phi\to 0$ and $b/r\to 0$ as $r\to\infty$.
- **Regular throat:** $b(r_0)=r_0$, $\Phi$ finite, $\ell$ regular.
- **Sources:** vacuum + scalar field (Ellis), Einstein–Maxwell (charged), or
  prescribed exotic stress-energy (Morris–Thorne).
- Static/stationary symmetry unless evolving the optional scalar-wave demo.

---

## Annotated references

- **[MT88†L223-L231]** M. Morris & K. Thorne, *Wormholes in spacetime and their use
  for interstellar travel*, Am. J. Phys. **56**, 395 (1988). Defines static
  traversable wormholes; derives the exotic-matter requirement.
- **[EB73†L217-L224]** H. Ellis, *Ether flow through a drainhole*, J. Math. Phys.
  **14**, 104 (1973); K. Bronnikov, Acta Phys. Pol. **B4**, 251 (1973). The
  drainhole / first traversable solution.
- **[ER†L477-L485]** A. Einstein & N. Rosen, *The particle problem in the general
  theory of relativity*, Phys. Rev. **48**, 73 (1935). The non-traversable bridge.
- **[Teo98†]** E. Teo, *Rotating traversable wormholes*, Phys. Rev. D **58**, 024014
  (1998).
- **[FR95†L172-L179]** L. Ford & T. Roman, *Averaged energy conditions and quantum
  inequalities*, Phys. Rev. D **51**, 4277 (1995).
- **[GJW17†L50-L57]** P. Gao, D. Jafferis & A. Wall, *Traversable wormholes via a
  double trace deformation*, JHEP **12**, 151 (2017).
- M. Visser, *Lorentzian Wormholes: From Einstein to Hawking* (AIP, 1995) — the
  standard monograph (thin-shell construction, stability).
- F. Lobo (ed.), *Wormholes, Warp Drives and Energy Conditions* (Springer, 2017) —
  modern review including modified-gravity wormholes.
