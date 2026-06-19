"""Wormhole metric definitions.

Conventions
-----------
* Signature (-,+,+,+).
* Geometrized units G = c = 1.
* Coordinates ordered (t, r, theta, phi).

Christoffel symbols are obtained through ``core.backend`` (JAX autodiff when
available, else 4th-order Richardson finite differences), so any subclass need
only implement ``components(x)``.  Subclasses may additionally provide
``components_jax(x)`` (a JAX-numpy, in-place-free version) to unlock
machine-precision, GPU-batchable connections.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = [
    "Metric",
    "MorrisThorne",
    "EllisBronnikov",
    "ChargedWormhole",
    "TeoRotating",
]


class Metric:
    """Abstract base class for a metric in coordinates (t, r, theta, phi)."""

    dim = 4

    def components(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.components(x))

    def line_element_norm(self, x: np.ndarray, u: np.ndarray) -> float:
        g = self.components(x)
        return float(u @ g @ u)

    def christoffel(self, x: np.ndarray) -> np.ndarray:
        """Gamma^lambda_{mu nu} (4,4,4) via core.backend (autodiff or Richardson FD)."""
        from . import backend
        return backend.connection(self, np.asarray(x, dtype=float))

    def geodesic_rhs(self, state: np.ndarray) -> np.ndarray:
        x = state[:4]
        u = state[4:]
        gamma = self.christoffel(x)
        du = -np.einsum("mab,a,b->m", gamma, u, u)
        return np.concatenate([u, du])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._params_repr()})"

    def _params_repr(self) -> str:
        return ""


@dataclass
class MorrisThorne(Metric):
    r"""Morris-Thorne traversable wormhole (Morris & Thorne 1988).

    ds^2 = -e^{2 Phi(r)} dt^2 + dr^2/(1 - b(r)/r) + r^2 (dtheta^2 + sin^2 theta dphi^2)
    """

    b0: float = 1.0
    redshift: callable = None
    shape: callable = None

    def __post_init__(self):
        if self.b0 <= 0:
            raise ValueError("throat radius b0 must be positive")
        if self.redshift is None:
            self.redshift = lambda r: 0.0
        if self.shape is None:
            b0 = self.b0
            self.shape = lambda r: b0 * b0 / r

    def components(self, x: np.ndarray) -> np.ndarray:
        _, r, theta, _ = x
        r = max(abs(r), 1e-9)
        Phi = self.redshift(r)
        b = self.shape(r)
        d = 1.0 - b / r
        d = d if abs(d) > 1e-12 else 1e-12
        g = np.zeros((4, 4))
        g[0, 0] = -np.exp(2.0 * Phi)
        g[1, 1] = 1.0 / d
        g[2, 2] = r * r
        g[3, 3] = r * r * np.sin(theta) ** 2
        return g

    def components_jax(self, x):
        import jax.numpy as jnp
        r, theta = x[1], x[2]
        Phi = self.redshift(r)
        b = self.shape(r)
        grr = 1.0 / (1.0 - b / r)
        return jnp.diag(jnp.array([-jnp.exp(2.0 * Phi), grr, r * r, r * r * jnp.sin(theta) ** 2]))

    def embedding_dzdr(self, r: float) -> float:
        b = self.shape(r)
        val = r / b - 1.0
        return 1.0 / np.sqrt(val) if val > 0 else np.inf

    def proper_radial_distance(self, r: float, n: int = 2000) -> float:
        rs = np.linspace(self.b0 * (1 + 1e-6), r, n)
        integrand = 1.0 / np.sqrt(1.0 - self.shape(rs) / rs)
        return float(np.trapz(integrand, rs))

    def _params_repr(self) -> str:
        return f"b0={self.b0}"


@dataclass
class EllisBronnikov(Metric):
    r"""Ellis-Bronnikov massless wormhole (Ellis 1973, Bronnikov 1973).

    ds^2 = -dt^2 + dr^2 + (r^2 + b0^2)(dtheta^2 + sin^2 theta dphi^2), r in (-inf, inf)
    """

    b0: float = 1.0

    def __post_init__(self):
        if self.b0 <= 0:
            raise ValueError("throat radius b0 must be positive")

    def components(self, x: np.ndarray) -> np.ndarray:
        _, r, theta, _ = x
        rho2 = r * r + self.b0 * self.b0
        g = np.zeros((4, 4))
        g[0, 0] = -1.0
        g[1, 1] = 1.0
        g[2, 2] = rho2
        g[3, 3] = rho2 * np.sin(theta) ** 2
        return g

    def components_jax(self, x):
        import jax.numpy as jnp
        r, theta = x[1], x[2]
        rho2 = r * r + self.b0 * self.b0
        return jnp.diag(jnp.array([-1.0, 1.0, rho2, rho2 * jnp.sin(theta) ** 2]))

    def embedding_radius(self, r: float) -> float:
        return float(np.sqrt(r * r + self.b0 * self.b0))

    def _params_repr(self) -> str:
        return f"b0={self.b0}"


@dataclass
class ChargedWormhole(Metric):
    r"""Charged Morris-Thorne-type wormhole: b(r)=r0 + Q^2/r0 - Q^2/r, Phi=0."""

    b0: float = 1.0
    Q: float = 0.3

    def __post_init__(self):
        if self.b0 <= 0:
            raise ValueError("throat radius b0 must be positive")

    def shape(self, r: float) -> float:
        return self.b0 + self.Q ** 2 / self.b0 - self.Q ** 2 / r

    def components(self, x: np.ndarray) -> np.ndarray:
        _, r, theta, _ = x
        r = max(abs(r), 1e-9)
        b = self.shape(r)
        d = 1.0 - b / r
        d = d if abs(d) > 1e-12 else 1e-12
        g = np.zeros((4, 4))
        g[0, 0] = -1.0
        g[1, 1] = 1.0 / d
        g[2, 2] = r * r
        g[3, 3] = r * r * np.sin(theta) ** 2
        return g

    def components_jax(self, x):
        import jax.numpy as jnp
        r, theta = x[1], x[2]
        b = self.b0 + self.Q ** 2 / self.b0 - self.Q ** 2 / r
        grr = 1.0 / (1.0 - b / r)
        return jnp.diag(jnp.array([-1.0, grr, r * r, r * r * jnp.sin(theta) ** 2]))

    def _params_repr(self) -> str:
        return f"b0={self.b0}, Q={self.Q}"


@dataclass
class TeoRotating(Metric):
    r"""Stationary axisymmetric rotating wormhole (Teo 1998).

    ds^2 = -N^2 dt^2 + dr^2/(1-b/r) + r^2 K^2 [dtheta^2 + sin^2 theta (dphi - omega dt)^2]
    with N=K=1, b=b0^2/r, omega=2 J / r^3.
    """

    b0: float = 1.0
    J: float = 0.2

    def __post_init__(self):
        if self.b0 <= 0:
            raise ValueError("throat radius b0 must be positive")

    def omega(self, r: float) -> float:
        return 2.0 * self.J / r ** 3

    def shape(self, r: float) -> float:
        return self.b0 * self.b0 / r

    def components(self, x: np.ndarray) -> np.ndarray:
        _, r, theta, _ = x
        r = max(abs(r), 1e-9)
        st = np.sin(theta)
        b = self.shape(r)
        w = self.omega(r)
        d = 1.0 - b / r
        d = d if abs(d) > 1e-12 else 1e-12
        g = np.zeros((4, 4))
        g[0, 0] = -1.0 + r * r * st * st * w * w
        g[0, 3] = -r * r * st * st * w
        g[3, 0] = g[0, 3]
        g[1, 1] = 1.0 / d
        g[2, 2] = r * r
        g[3, 3] = r * r * st * st
        return g

    def components_jax(self, x):
        import jax.numpy as jnp
        r, theta = x[1], x[2]
        st = jnp.sin(theta)
        b = self.b0 * self.b0 / r
        w = 2.0 * self.J / r ** 3
        grr = 1.0 / (1.0 - b / r)
        g00 = -1.0 + r * r * st * st * w * w
        g03 = -r * r * st * st * w
        row0 = jnp.array([g00, 0.0, 0.0, g03])
        row1 = jnp.array([0.0, grr, 0.0, 0.0])
        row2 = jnp.array([0.0, 0.0, r * r, 0.0])
        row3 = jnp.array([g03, 0.0, 0.0, r * r * st * st])
        return jnp.stack([row0, row1, row2, row3])

    def zamo_angular_velocity(self, r: float) -> float:
        """ZAMO angular velocity Omega = -g_{t phi}/g_{phi phi} = omega(r)."""
        return self.omega(r)

    def in_ergoregion(self, r: float, theta: float = np.pi / 2) -> bool:
        """True where g_{tt} >= 0 (frame-dragging speed r sin(theta) omega >= 1)."""
        st = np.sin(theta)
        return bool(-1.0 + (r * st * self.omega(r)) ** 2 >= 0.0)

    def _params_repr(self) -> str:
        return f"b0={self.b0}, J={self.J}"
