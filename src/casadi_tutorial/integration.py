"""Small explicit integrators that work with SX, MX, DM, and NumPy-like values."""

from collections.abc import Callable
from typing import TypeVar

Vector = TypeVar("Vector")


def euler_step(
    rhs: Callable[[Vector, Vector], Vector], x: Vector, u: Vector, dt: float
) -> Vector:
    """Return one forward-Euler step for ``dx/dt = rhs(x, u)``."""
    return x + dt * rhs(x, u)


def rk4_step(
    rhs: Callable[[Vector, Vector], Vector], x: Vector, u: Vector, dt: float
) -> Vector:
    """Return one classical fourth-order Runge–Kutta step with constant control."""
    k1 = rhs(x, u)
    k2 = rhs(x + 0.5 * dt * k1, u)
    k3 = rhs(x + 0.5 * dt * k2, u)
    k4 = rhs(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
