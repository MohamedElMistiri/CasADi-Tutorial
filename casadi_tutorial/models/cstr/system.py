"""Discrete CSTR construction and an object-oriented convenience facade."""

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from casadi_tutorial.integration import rk4_step
from casadi_tutorial.models.cstr.analysis import (
    linearize_cstr,
    steady_state_residual,
)
from casadi_tutorial.models.cstr.dynamics import (
    cstr_function,
    cstr_rhs,
    normalize_symbolic_type,
)
from casadi_tutorial.models.cstr.parameters import CSTRParameters


def cstr_rk4_function(
    dt: float,
    params: CSTRParameters | None = None,
    symbolic: str = "MX",
) -> ca.Function:
    """Build a fixed-step CSTR map with constant input and disturbance."""
    symbolic_type = normalize_symbolic_type(symbolic)
    symbol = ca.SX if symbolic_type == "SX" else ca.MX

    state = symbol.sym("x", 2)
    control = symbol.sym("u", 1)
    disturbance = symbol.sym("d", 2)

    # RK4 accepts one held input vector, so concatenate the manipulated input and
    # measured disturbance and split them inside the model callback.
    held_values = ca.vertcat(control, disturbance)
    next_state = rk4_step(
        lambda current_state, values: cstr_rhs(
            current_state,
            values[:1],
            values[1:],
            params,
        ),
        state,
        held_values,
        dt,
    )

    return ca.Function(
        "cstr_rk4",
        [state, control, disturbance],
        [next_state],
        ["x", "u", "d"],
        ["x_next"],
    )


@dataclass(frozen=True)
class CSTRSystem:
    """Convenience facade that constructs fresh CSTR functions on demand.

    No CasADi graph is cached. This is intentional: after IPython autoreloads an
    edited module, the next method call reflects the updated equations.
    """

    parameters: CSTRParameters = field(default_factory=CSTRParameters)
    symbolic_type: str = "SX"

    def __post_init__(self) -> None:
        """Reject unsupported symbolic representations at construction time."""
        normalize_symbolic_type(self.symbolic_type)

    def continuous_dynamics(self) -> ca.Function:
        """Build the continuous map ``f(x, u, d) -> xdot``."""
        return cstr_function(self.parameters, self.symbolic_type)

    def discrete_dynamics(self, step_size: float) -> ca.Function:
        """Build an RK4 map ``F(x, u, d) -> x_next``."""
        return cstr_rk4_function(
            step_size,
            self.parameters,
            symbolic=self.symbolic_type,
        )

    def steady_state_residual(self) -> np.ndarray:
        """Evaluate the nominal continuous derivative."""
        return steady_state_residual(self.parameters)

    def linearize(
        self,
        state,
        control,
        disturbance,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearize the continuous dynamics at supplied values."""
        return linearize_cstr(
            state,
            control,
            disturbance,
            self.parameters,
        )
