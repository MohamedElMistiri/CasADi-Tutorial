"""Steady-state and local linear analysis for the nonlinear CSTR."""

import casadi as ca
import numpy as np

from casadi_tutorial.models.cstr.dynamics import cstr_function, cstr_rhs
from casadi_tutorial.models.cstr.parameters import CSTRParameters


def steady_state_residual(params: CSTRParameters | None = None) -> np.ndarray:
    """Evaluate the nonlinear derivative at the documented nominal point."""
    model_parameters = params or CSTRParameters()
    dynamics = cstr_function(model_parameters)
    derivative = dynamics(
        model_parameters.nominal_state,
        model_parameters.nominal_input,
        model_parameters.nominal_disturbance,
    )
    return np.asarray(derivative.full()).ravel()


def linearize_cstr(
    x_value,
    u_value,
    d_value,
    params: CSTRParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return continuous Jacobians ``A=df/dx``, ``B=df/du``, and ``E=df/dd``."""
    state = ca.SX.sym("state", 2)
    control = ca.SX.sym("control", 1)
    disturbance = ca.SX.sym("disturbance", 2)
    derivative = cstr_rhs(state, control, disturbance, params)

    # Automatic differentiation preserves the exact symbolic derivatives of the
    # nonlinear balances; no finite-difference step size is required.
    jacobian_function = ca.Function(
        "cstr_jacobians",
        [state, control, disturbance],
        [
            ca.jacobian(derivative, state),
            ca.jacobian(derivative, control),
            ca.jacobian(derivative, disturbance),
        ],
    )
    matrices = jacobian_function(x_value, u_value, d_value)
    return tuple(np.asarray(matrix.full()) for matrix in matrices)
