# wormhole-sim documentation

A research framework for simulating traversable-wormhole spacetimes: geometry,
geodesics, energy conditions, and lensing.

## Overview of wormhole physics

A **wormhole** is a spacetime geometry connecting two regions — two asymptotic
universes, or two distant parts of one universe — through a **throat**. The idea
traces to Flamm (1916), who first embedded the Schwarzschild geometry, and to the
**Einstein–Rosen bridge** (1935) [ER†L477-L485]. Wheeler coined the term
"wormhole" in 1957, but the Schwarzschild/Einstein–Rosen bridge is **non-traversable**:
its throat is a horizon that pinches off before any signal can cross [MTW†L477-L485].

The modern theory of **traversable** wormholes begins with the Ellis–Bronnikov
"drainhole" (1973) and its rediscovery by **Morris & Thorne (1988)** [MT88†L223-L231].
Morris and Thorne showed that a static, horizonless, two-way-traversable throat in
classical general relativity requires matter that **violates the null energy
condition** (NEC) — so-called *exotic matter* with locally negative energy density
[MT88†L223-L231]. Quantum field theory does permit localized negative energy
(the Casimir effect, squeezed vacua), but **quantum inequalities** (Ford & Roman 1995)
bound its magnitude and duration [FR95†L172-L179].

Recent developments connect wormholes to quantum information: the **ER=EPR**
conjecture (Maldacena & Susskind 2013) and the construction of a quantum-coupled
**traversable** AdS wormhole by **Gao–Jafferis–Wall (2017)** [GJW17†L50-L57].

This project implements the *classical, geometric* side of that story in code: you
specify a metric, and the framework gives you its curvature, its stress-energy and
energy-condition violation, its geodesics, and its visual signature (embedding
surface and gravitational lensing).

## Reading order

1. [`mathematical_foundations.md`](mathematical_foundations.md) — the metrics and
   the physics they must satisfy.
2. [`numerical_methods.md`](numerical_methods.md) — how the equations are solved.
3. [`visualization.md`](visualization.md) — how results are rendered.
4. [`architecture.md`](architecture.md) — how the code is organized.
5. [`deliverables.md`](deliverables.md) — a graded set of demos and validation tests.
6. [`related_work.md`](related_work.md) — existing tools and how this differs.

## Citation labels

Citations use the bracket form `[label†Lx-Ly]` pointing into the annotated
bibliography at the end of `mathematical_foundations.md`. They are human-readable
anchors, not hyperlinks; full references (with DOIs) accompany each label.
