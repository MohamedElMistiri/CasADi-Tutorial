"""Self-contained UR5e kinematics and joint-space surrogate dynamics.

The modified control model is ``qdot = v, vdot = a``. It intentionally omits
rigid-body torque dynamics. DH values follow Universal Robots' published UR5e
kinematic table and are expressed in metres and radians:
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
"""

from dataclasses import dataclass, field

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class UR5eParameters:
    """UR5e standard-DH geometry and conservative teaching limits."""

    a: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
    )
    d: np.ndarray = field(
        default_factory=lambda: np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
    )
    alpha: np.ndarray = field(
        default_factory=lambda: np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
    )
    q_min: np.ndarray = field(default_factory=lambda: np.full(6, -2 * np.pi))
    q_max: np.ndarray = field(default_factory=lambda: np.full(6, 2 * np.pi))
    v_max: np.ndarray = field(default_factory=lambda: np.full(6, np.pi))
    a_max: np.ndarray = field(default_factory=lambda: np.full(6, 2 * np.pi))


def dh_transform(theta, d: float, a: float, alpha: float):
    """Return a standard Denavit–Hartenberg homogeneous transform."""
    ct, st = ca.cos(theta), ca.sin(theta)
    ca_, sa = np.cos(alpha), np.sin(alpha)
    return ca.vertcat(
        ca.horzcat(ct, -st * ca_, st * sa, a * ct),
        ca.horzcat(st, ct * ca_, -ct * sa, a * st),
        ca.horzcat(0, sa, ca_, d),
        ca.horzcat(0, 0, 0, 1),
    )


def link_transforms(q, params: UR5eParameters | None = None) -> list:
    """Return base-to-link transforms, including the identity base transform."""
    p = params or UR5eParameters()
    transforms = [ca.DM.eye(4)]
    current = transforms[0]
    for index in range(6):
        current = ca.mtimes(current, dh_transform(q[index], p.d[index], p.a[index], p.alpha[index]))
        transforms.append(current)
    return transforms


def forward_kinematics(q, params: UR5eParameters | None = None):
    """Return the 4x4 base-to-tool transform."""
    return link_transforms(q, params)[-1]


def joint_positions(q, params: UR5eParameters | None = None):
    """Return a 3x7 matrix containing the base and each link-frame origin."""
    return ca.horzcat(*[transform[:3, 3] for transform in link_transforms(q, params)])


def geometric_jacobian(q, params: UR5eParameters | None = None):
    """Return the 6x6 geometric tool Jacobian in the base frame."""
    transforms = link_transforms(q, params)
    tool = transforms[-1][:3, 3]
    columns = []
    for index in range(6):
        axis = transforms[index][:3, 2]
        origin = transforms[index][:3, 3]
        columns.append(ca.vertcat(ca.cross(axis, tool - origin), axis))
    return ca.horzcat(*columns)


def joint_space_rhs(state, acceleration):
    """Return ``[qdot, vdot]`` for the 12-state double-integrator surrogate."""
    return ca.vertcat(state[6:12], acceleration)
