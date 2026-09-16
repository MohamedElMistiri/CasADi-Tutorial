"""Models shared by the tutorial capstones."""

from casadi_tutorial.models.cstr import CSTRParameters, cstr_function, cstr_rhs
from casadi_tutorial.models.ur5e import UR5eParameters, forward_kinematics

__all__ = [
    "CSTRParameters",
    "UR5eParameters",
    "cstr_function",
    "cstr_rhs",
    "forward_kinematics",
]
