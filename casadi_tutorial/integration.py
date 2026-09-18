"""Explicit integration formulas that work with CasADi and NumPy-like values."""

from collections.abc import Callable
from typing import TypeVar

# A generic vector type lets the same formula accept SX, MX, DM, or compatible
# numerical arrays without converting away CasADi's symbolic information.
Vector = TypeVar("Vector")


def euler_step(
    rhs: Callable[[Vector, Vector], Vector],
    x: Vector,
    u: Vector,
    dt: float,
) -> Vector:
    """Return one forward-Euler step for ``dx/dt = rhs(x, u)``."""
    derivative = rhs(x, u)
    return x + dt * derivative


def rk4_step(
    rhs: Callable[[Vector, Vector], Vector],
    x: Vector,
    u: Vector,
    dt: float,
) -> Vector:
    """Return one classical fourth-order Runge-Kutta step.

    The control is held constant across all four stages, matching the usual
    zero-order-hold assumption in the optimal-control lessons.
    """
    first_slope = rhs(x, u)
    second_slope = rhs(x + 0.5 * dt * first_slope, u)
    third_slope = rhs(x + 0.5 * dt * second_slope, u)
    fourth_slope = rhs(x + dt * third_slope, u)

    weighted_slope = first_slope + 2 * second_slope + 2 * third_slope + fourth_slope
    return x + (dt / 6.0) * weighted_slope
