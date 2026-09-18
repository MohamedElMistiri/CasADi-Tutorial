"""Kinematic and surrogate-dynamics tests for the editable UR5e model."""

import casadi as ca
import numpy as np

from casadi_tutorial.models.ur5e import (
    UR5eParameters,
    forward_kinematics,
    geometric_jacobian,
    joint_positions,
    joint_space_rhs,
)


def test_kinematic_shapes_and_homogeneous_row() -> None:
    """Kinematic outputs should have their documented matrix structures."""
    q = ca.DM.zeros(6)
    transform = np.asarray(forward_kinematics(q).full())
    assert transform.shape == (4, 4)
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=1e-14)
    assert joint_positions(q).shape == (3, 7)
    assert geometric_jacobian(q).shape == (6, 6)


def test_translational_jacobian_matches_finite_difference() -> None:
    """Validate the analytic translational Jacobian along one joint direction."""
    q_symbol = ca.MX.sym("q", 6)
    position = ca.Function("test_position", [q_symbol], [forward_kinematics(q_symbol)[:3, 3]])
    jacobian = ca.Function("test_jacobian", [q_symbol], [geometric_jacobian(q_symbol)[:3, :]])
    q = np.array([0.1, -0.5, 0.6, -0.2, 0.3, 0.1])
    direction = np.arange(1, 7) / 10
    epsilon = 1e-6
    # A centered difference removes the leading one-sided truncation error.
    finite_difference = (position(q + epsilon * direction) - position(q - epsilon * direction)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(
        finite_difference.full().ravel(),
        (jacobian(q) @ direction).full().ravel(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_parameters_and_surrogate_dynamics() -> None:
    """Parameter vectors and the double-integrator state ordering stay consistent."""
    p = UR5eParameters()
    assert all(array.shape == (6,) for array in (p.a, p.d, p.alpha, p.q_min, p.q_max))
    rhs = joint_space_rhs(ca.DM.zeros(12), ca.DM.ones(6))
    np.testing.assert_allclose(rhs.full().ravel(), [0] * 6 + [1] * 6)
