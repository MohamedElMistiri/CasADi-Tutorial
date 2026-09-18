"""Transparent UR5e kinematics and joint-space surrogate dynamics.

The control model is ``qdot = velocity`` and ``velocity_dot = acceleration``.
It intentionally omits torque-level rigid-body dynamics. Standard-DH geometry
is expressed in metres and radians using Universal Robots' published table.
"""

from dataclasses import dataclass, field

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class UR5eParameters:
    """Store UR5e standard-DH geometry and conservative teaching limits."""

    # Each six-vector follows the base-to-tool joint order.
    a: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]))
    d: np.ndarray = field(
        default_factory=lambda: np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
    )
    alpha: np.ndarray = field(
        default_factory=lambda: np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
    )

    # Symmetric limits keep the OCP example readable; they are tutorial values,
    # not a replacement for model-specific hardware safety limits.
    q_min: np.ndarray = field(default_factory=lambda: np.full(6, -2 * np.pi))
    q_max: np.ndarray = field(default_factory=lambda: np.full(6, 2 * np.pi))
    v_max: np.ndarray = field(default_factory=lambda: np.full(6, np.pi))
    a_max: np.ndarray = field(default_factory=lambda: np.full(6, 2 * np.pi))


def dh_transform(theta, d: float, a: float, alpha: float):
    """Return one standard Denavit-Hartenberg homogeneous transform."""
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # Rows combine a rotation and translation into one 4x4 frame transform.
    return ca.vertcat(
        ca.horzcat(
            cos_theta,
            -sin_theta * cos_alpha,
            sin_theta * sin_alpha,
            a * cos_theta,
        ),
        ca.horzcat(
            sin_theta,
            cos_theta * cos_alpha,
            -cos_theta * sin_alpha,
            a * sin_theta,
        ),
        ca.horzcat(0, sin_alpha, cos_alpha, d),
        ca.horzcat(0, 0, 0, 1),
    )


def link_transforms(q, params: UR5eParameters | None = None) -> list:
    """Return base-to-link transforms, including the identity base frame."""
    model_parameters = params or UR5eParameters()
    transforms = [ca.DM.eye(4)]
    current_transform = transforms[0]

    # Premultiplying each relative transform expresses every frame in the base
    # coordinate system, which is also the frame used by the Jacobian.
    for joint_index in range(6):
        relative_transform = dh_transform(
            q[joint_index],
            model_parameters.d[joint_index],
            model_parameters.a[joint_index],
            model_parameters.alpha[joint_index],
        )
        current_transform = ca.mtimes(current_transform, relative_transform)
        transforms.append(current_transform)

    return transforms


def forward_kinematics(q, params: UR5eParameters | None = None):
    """Return the 4x4 base-to-tool transform."""
    return link_transforms(q, params)[-1]


def joint_positions(q, params: UR5eParameters | None = None):
    """Return a 3x7 matrix containing the base and link-frame origins."""
    transforms = link_transforms(q, params)
    return ca.horzcat(*[transform[:3, 3] for transform in transforms])


def geometric_jacobian(q, params: UR5eParameters | None = None):
    """Return the 6x6 geometric tool Jacobian in the base frame."""
    transforms = link_transforms(q, params)
    tool_position = transforms[-1][:3, 3]
    columns = []

    for joint_index in range(6):
        # For a revolute joint, z_i is its rotation axis. The upper cross product
        # gives linear velocity and the lower axis gives angular velocity.
        joint_axis = transforms[joint_index][:3, 2]
        joint_origin = transforms[joint_index][:3, 3]
        linear_velocity = ca.cross(joint_axis, tool_position - joint_origin)
        columns.append(ca.vertcat(linear_velocity, joint_axis))

    return ca.horzcat(*columns)


def joint_space_rhs(state, acceleration):
    """Return ``[qdot, velocity_dot]`` for the 12-state surrogate model."""
    joint_velocity = state[6:12]
    return ca.vertcat(joint_velocity, acceleration)
