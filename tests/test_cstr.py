"""Behavioral tests for the functional and object-oriented CSTR builders."""

import casadi as ca
import numpy as np
import pytest

from casadi_tutorial.models.cstr import (
    CSTRParameters,
    CSTRSystem,
    cstr_function,
    cstr_rhs,
    cstr_rk4_function,
    linearize_cstr,
    steady_state_residual,
)


def test_nominal_point_is_steady() -> None:
    """The documented operating point should satisfy both material balances."""
    assert np.linalg.norm(steady_state_residual(), ord=np.inf) < 2e-4


@pytest.mark.parametrize("symbolic_type", ["SX", "MX", "sx", "mx"])
def test_continuous_builder_supports_symbolic_types(symbolic_type: str) -> None:
    """Both CasADi graph representations should expose the same 2-1-2 map."""
    dynamics = cstr_function(symbolic=symbolic_type)

    assert dynamics.size1_in(0) == 2
    assert dynamics.size1_in(1) == 1
    assert dynamics.size1_in(2) == 2
    assert dynamics.size1_out(0) == 2
    assert dynamics.name_in() == ["x", "u", "d"]
    assert dynamics.name_out() == ["xdot"]


def test_invalid_symbolic_type_is_rejected() -> None:
    """A misspelled graph type should fail before constructing a model graph."""
    with pytest.raises(ValueError, match="SX.*MX"):
        CSTRSystem(symbolic_type="DM")


def test_linearization_shapes_and_finite_values() -> None:
    """Linearization should produce finite state, input, and disturbance blocks."""
    p = CSTRParameters()
    a, b, e = linearize_cstr(p.nominal_state, p.nominal_input, p.nominal_disturbance)

    assert a.shape == (2, 2)
    assert b.shape == (2, 1)
    assert e.shape == (2, 2)
    assert all(np.isfinite(matrix).all() for matrix in (a, b, e))


def test_discrete_map_preserves_nominal_point() -> None:
    """An RK4 step should leave a continuous equilibrium almost unchanged."""
    p = CSTRParameters()
    step = cstr_rk4_function(0.01, p)
    next_state = np.asarray(
        step(p.nominal_state, p.nominal_input, p.nominal_disturbance).full()
    ).ravel()

    np.testing.assert_allclose(next_state, p.nominal_state, atol=2e-6)


def test_system_facade_matches_functional_api() -> None:
    """The facade should remain a convenience layer over the visible functions."""
    p = CSTRParameters()
    system = CSTRSystem(parameters=p, symbolic_type="MX")

    continuous_value = system.continuous_dynamics()(
        p.nominal_state,
        p.nominal_input,
        p.nominal_disturbance,
    )
    functional_value = cstr_function(p, symbolic="MX")(
        p.nominal_state,
        p.nominal_input,
        p.nominal_disturbance,
    )
    np.testing.assert_allclose(continuous_value.full(), functional_value.full())

    object_step = system.discrete_dynamics(0.01)
    functional_step = cstr_rk4_function(0.01, p, symbolic="MX")
    object_next = object_step(p.nominal_state, p.nominal_input, p.nominal_disturbance)
    functional_next = functional_step(
        p.nominal_state,
        p.nominal_input,
        p.nominal_disturbance,
    )
    np.testing.assert_allclose(object_next.full(), functional_next.full())

    object_jacobians = system.linearize(
        p.nominal_state,
        p.nominal_input,
        p.nominal_disturbance,
    )
    functional_jacobians = linearize_cstr(
        p.nominal_state,
        p.nominal_input,
        p.nominal_disturbance,
        p,
    )
    for object_matrix, functional_matrix in zip(
        object_jacobians,
        functional_jacobians,
        strict=True,
    ):
        np.testing.assert_allclose(object_matrix, functional_matrix)

    np.testing.assert_allclose(
        system.steady_state_residual(),
        steady_state_residual(p),
    )


def test_symbolic_parameter_remains_editable() -> None:
    """Parameters may contain CasADi decisions for estimation formulations."""
    heat_transfer = ca.MX.sym("heat_transfer")
    p = CSTRParameters(heat_transfer=heat_transfer)
    state = ca.MX.sym("state", 2)
    control = ca.MX.sym("control")
    disturbance = ca.MX.sym("disturbance", 2)
    derivative = cstr_rhs(state, control, disturbance, p)

    # The free heat-transfer symbol should remain part of the graph rather than
    # being coerced to a float by the parameter container.
    assert ca.depends_on(derivative, heat_transfer)
