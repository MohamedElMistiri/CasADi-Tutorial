"""Composable construction tools for the non-isothermal CSTR model.

The package exposes both small functional building blocks and ``CSTRSystem``, a
convenience facade. All builders create fresh CasADi graphs on each call so that
edits made during a notebook session can be reloaded immediately.
"""

from casadi_tutorial.models.cstr.analysis import (
    linearize_cstr,
    steady_state_residual,
)
from casadi_tutorial.models.cstr.dynamics import (
    cstr_function,
    cstr_rhs,
    reaction_rate,
)
from casadi_tutorial.models.cstr.parameters import CSTRParameters
from casadi_tutorial.models.cstr.system import CSTRSystem, cstr_rk4_function

__all__ = [
    "CSTRParameters",
    "CSTRSystem",
    "cstr_function",
    "cstr_rhs",
    "cstr_rk4_function",
    "linearize_cstr",
    "reaction_rate",
    "steady_state_residual",
]
