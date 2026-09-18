"""Continuous-time equations and graph builders for the CSTR."""

from typing import Literal, cast

import casadi as ca

from casadi_tutorial.models.cstr.parameters import CSTRParameters

SymbolicType = Literal["SX", "MX"]


def normalize_symbolic_type(symbolic_type: str) -> SymbolicType:
    """Validate and normalize a requested CasADi symbolic representation."""
    normalized = symbolic_type.upper()
    if normalized not in {"SX", "MX"}:
        raise ValueError("symbolic_type must be either 'SX' or 'MX'")
    return cast(SymbolicType, normalized)


def reaction_rate(concentration, temperature, params: CSTRParameters | None = None):
    """Return the first-order Arrhenius reaction rate in kmol/(m^3 h)."""
    model_parameters = params or CSTRParameters()

    # The negative exponent makes the reaction accelerate as temperature rises.
    arrhenius_factor = model_parameters.pre_exponential * ca.exp(
        -model_parameters.activation_energy / (model_parameters.gas_constant * temperature)
    )
    return arrhenius_factor * concentration


def cstr_rhs(x, u, d, params: CSTRParameters | None = None):
    """Return the nonlinear CSTR derivative for symbolic or numeric inputs.

    ``state`` is ``[C_A, T]``, ``control`` is jacket temperature ``[T_j]``, and
    ``disturbance`` is ``[C_Af, T_f]``.
    """
    model_parameters = params or CSTRParameters()

    concentration = x[0]
    temperature = x[1]
    jacket_temperature = u[0]
    feed_concentration = d[0]
    feed_temperature = d[1]

    dilution_rate = model_parameters.flow_rate / model_parameters.volume
    rate = reaction_rate(concentration, temperature, model_parameters)

    # Species balance: inlet/outlet dilution minus reactant consumption.
    concentration_derivative = dilution_rate * (feed_concentration - concentration) - rate

    # Energy balance: feed mixing, exothermic heat release, and jacket exchange.
    feed_heat = dilution_rate * (feed_temperature - temperature)
    reaction_heat = (
        -model_parameters.heat_of_reaction / model_parameters.volumetric_heat_capacity
    ) * rate
    jacket_heat = (
        model_parameters.heat_transfer
        / (model_parameters.volume * model_parameters.volumetric_heat_capacity)
        * (jacket_temperature - temperature)
    )
    temperature_derivative = feed_heat + reaction_heat + jacket_heat

    return ca.vertcat(concentration_derivative, temperature_derivative)


def cstr_function(
    params: CSTRParameters | None = None,
    symbolic: str = "SX",
) -> ca.Function:
    """Build ``f(state, control, disturbance) -> state_derivative``."""
    symbolic_type = normalize_symbolic_type(symbolic)
    symbol = ca.SX if symbolic_type == "SX" else ca.MX

    # Named vectors make generated graphs and error messages easier to inspect.
    state = symbol.sym("x", 2)
    control = symbol.sym("u", 1)
    disturbance = symbol.sym("d", 2)
    derivative = cstr_rhs(state, control, disturbance, params)

    return ca.Function(
        "cstr_rhs",
        [state, control, disturbance],
        [derivative],
        ["x", "u", "d"],
        ["xdot"],
    )
