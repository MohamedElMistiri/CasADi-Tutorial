import numpy as np

from casadi_tutorial.models.cstr import (
    CSTRParameters,
    cstr_rk4_function,
    linearize_cstr,
    steady_state_residual,
)


def test_nominal_point_is_steady() -> None:
    assert np.linalg.norm(steady_state_residual(), ord=np.inf) < 2e-4


def test_linearization_shapes_and_finite_values() -> None:
    p = CSTRParameters()
    a, b, e = linearize_cstr(p.nominal_state, p.nominal_input, p.nominal_disturbance)
    assert a.shape == (2, 2)
    assert b.shape == (2, 1)
    assert e.shape == (2, 2)
    assert all(np.isfinite(matrix).all() for matrix in (a, b, e))


def test_discrete_map_preserves_nominal_point() -> None:
    p = CSTRParameters()
    step = cstr_rk4_function(0.01, p)
    next_state = np.asarray(
        step(p.nominal_state, p.nominal_input, p.nominal_disturbance).full()
    ).ravel()
    np.testing.assert_allclose(next_state, p.nominal_state, atol=2e-6)
