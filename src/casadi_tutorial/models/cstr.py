"""Non-isothermal continuous stirred-tank reactor teaching model.

State: ``x = [C_A (kmol/m^3), T (K)]``.
Input: ``u = [T_j (K)]``.
Disturbance: ``d = [C_Af (kmol/m^3), T_f (K)]``.
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np

from casadi_tutorial.integration import rk4_step


@dataclass(frozen=True)
class CSTRParameters:
    """Physical parameters and a nominal operating point for the CSTR."""

    flow_rate: float = 1.0  # m^3 / h
    volume: float = 1.0  # m^3
    pre_exponential: float = 9703.0 * 3600.0  # 1 / h
    activation_energy: float = 11843.0  # kcal / kmol
    heat_of_reaction: float = -5960.0  # kcal / kmol
    volumetric_heat_capacity: float = 500.0  # kcal / (m^3 K)
    heat_transfer: float = 150.0  # kcal / (h K)
    gas_constant: float = 1.98589  # kcal / (kmol K)
    feed_concentration: float = 10.0  # kmol / m^3
    feed_temperature: float = 298.15  # K
    nominal_jacket_temperature: float = 298.15  # K
    nominal_concentration: float = 8.5695  # kmol / m^3
    nominal_temperature: float = 311.267  # K

    @property
    def nominal_state(self) -> np.ndarray:
        return np.array([self.nominal_concentration, self.nominal_temperature])

    @property
    def nominal_input(self) -> np.ndarray:
        return np.array([self.nominal_jacket_temperature])

    @property
    def nominal_disturbance(self) -> np.ndarray:
        return np.array([self.feed_concentration, self.feed_temperature])


def cstr_rhs(x: ca.SX | ca.MX | ca.DM, u, d, params: CSTRParameters | None = None):
    """Return the nonlinear CSTR time derivative for symbolic or numeric inputs."""
    p = params or CSTRParameters()
    concentration, temperature = x[0], x[1]
    jacket_temperature = u[0]
    feed_concentration, feed_temperature = d[0], d[1]

    rate = p.pre_exponential * ca.exp(
        -p.activation_energy / (p.gas_constant * temperature)
    ) * concentration
    dc_dt = p.flow_rate / p.volume * (feed_concentration - concentration) - rate
    dt_dt = (
        p.flow_rate / p.volume * (feed_temperature - temperature)
        + (-p.heat_of_reaction / p.volumetric_heat_capacity) * rate
        + p.heat_transfer
        / (p.volume * p.volumetric_heat_capacity)
        * (jacket_temperature - temperature)
    )
    return ca.vertcat(dc_dt, dt_dt)


def cstr_function(params: CSTRParameters | None = None, symbolic: str = "SX") -> ca.Function:
    """Build a named CasADi function ``f(x, u, d) -> xdot``."""
    symbol = ca.SX if symbolic.upper() == "SX" else ca.MX
    x = symbol.sym("x", 2)
    u = symbol.sym("u", 1)
    d = symbol.sym("d", 2)
    return ca.Function(
        "cstr_rhs",
        [x, u, d],
        [cstr_rhs(x, u, d, params)],
        ["x", "u", "d"],
        ["xdot"],
    )


def steady_state_residual(params: CSTRParameters | None = None) -> np.ndarray:
    """Evaluate the nonlinear residual at the documented nominal point."""
    p = params or CSTRParameters()
    f = cstr_function(p)
    return np.asarray(f(p.nominal_state, p.nominal_input, p.nominal_disturbance).full()).ravel()


def linearize_cstr(
    x_value, u_value, d_value, params: CSTRParameters | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return continuous-time Jacobians ``A=df/dx``, ``B=df/du``, and ``E=df/dd``."""
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    d = ca.SX.sym("d", 2)
    rhs = cstr_rhs(x, u, d, params)
    jac = ca.Function(
        "cstr_jac",
        [x, u, d],
        [ca.jacobian(rhs, x), ca.jacobian(rhs, u), ca.jacobian(rhs, d)],
    )
    values = jac(x_value, u_value, d_value)
    return tuple(np.asarray(value.full()) for value in values)


def cstr_rk4_function(dt: float, params: CSTRParameters | None = None) -> ca.Function:
    """Build the discrete CSTR map with constant input/disturbance over ``dt`` hours."""
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)
    d = ca.MX.sym("d", 2)
    combined = ca.vertcat(u, d)
    next_x = rk4_step(lambda state, ud: cstr_rhs(state, ud[:1], ud[1:], params), x, combined, dt)
    return ca.Function("cstr_rk4", [x, u, d], [next_x], ["x", "u", "d"], ["x_next"])
