"""Checks against analytic KKT points, invalid candidates and missing statistics."""

import numpy as np
import pytest

from casadi_tutorial.diagnostics import (
    bound_report,
    kkt_residuals,
    max_bound_violation,
    solver_summary,
)


def test_bound_violation() -> None:
    assert max_bound_violation([0, 3], [0, 0], [2, 2]) == 1.0
    assert max_bound_violation([0, 2], [0, 0], [2, 2]) == 0.0
    assert max_bound_violation([], -np.inf, np.inf) == 0.0


@pytest.mark.parametrize("candidate", [np.nan, np.inf, -np.inf])
def test_nonfinite_candidate_is_never_feasible(candidate) -> None:
    assert np.isinf(max_bound_violation([candidate], -np.inf, np.inf))
    assert not bound_report([candidate], -np.inf, np.inf)[0]["finite"]


def test_matrix_bounds_use_casadi_column_order() -> None:
    values = [[1, 8], [3, 4]]
    report = bound_report(values, np.zeros((2, 2)), [[2, 5], [3, 4]])
    assert [row["value"] for row in report] == [1, 3, 8, 4]
    assert [row["violation"] for row in report] == [0, 0, 3, 0]
    assert report[2]["index"] == 2
    assert max_bound_violation(values, 0, [[2, 5], [3, 4]]) == 3


@pytest.mark.parametrize("lower,upper", [(2, 1), (np.nan, 1), (np.inf, np.inf), (0, -np.inf)])
def test_malformed_bounds_rejected(lower, upper) -> None:
    with pytest.raises(ValueError):
        max_bound_violation([0], lower, upper)


def test_bound_shapes_and_labels_checked() -> None:
    with pytest.raises(ValueError):
        max_bound_violation([0, 1], [0, 0, 0], 1)
    with pytest.raises(ValueError):
        bound_report([0, 1], 0, 1, labels=["only one"])


def test_kkt_equality_preserves_original_positional_api() -> None:
    residuals = kkt_residuals([-1, -1], [[1, 1]], [1], [2], [2], [2])
    assert set(residuals) == {
        "stationarity_inf",
        "primal_inf",
        "complementarity_inf",
        "dual_feasibility_inf",
    }
    assert all(value == 0 for value in residuals.values())


@pytest.mark.parametrize(
    "value,gradient,multiplier,lower,upper",
    [(0, 4, -4, 0, 1), (1, -4, 4, 0, 1), (0.5, 0, 0, 0, 1), (0, -7, 7, 0, 0)],
)
def test_kkt_active_inactive_and_fixed_decision_bounds(
    value, gradient, multiplier, lower, upper
) -> None:
    # Quadratic minima at either endpoint, an interior minimum and a fixed variable.
    residuals = kkt_residuals(
        [gradient],
        np.empty((0, 1)),
        [],
        [],
        [],
        [],
        variable=[value],
        lower_x=lower,
        upper_x=upper,
        lambda_x=[multiplier],
    )
    assert all(metric == 0 for metric in residuals.values())


@pytest.mark.parametrize(
    "value,lower,upper,multiplier,dual,complementarity",
    [
        (0, 0, np.inf, 2, 2, 0),
        (1, -np.inf, 1, -2, 2, 0),
        (0.5, 0, 1, 2, 0, 1),
        (0, -np.inf, np.inf, 2, 2, 0),
    ],
)
def test_signed_constraint_multipliers(value, lower, upper, multiplier, dual, complementarity):
    residuals = kkt_residuals([-multiplier], [[1]], [multiplier], [value], [lower], [upper])
    assert residuals["stationarity_inf"] == 0
    assert residuals["dual_feasibility_inf"] == dual
    assert residuals["complementarity_inf"] == complementarity


def test_primal_accounts_for_both_bound_kinds() -> None:
    residuals = kkt_residuals(
        [0],
        [[1]],
        [0],
        [2],
        [0],
        [1],
        variable=[3],
        lower_x=0,
        upper_x=1,
        lambda_x=[0],
    )
    assert residuals["primal_inf"] == 2


def test_kkt_rejects_partial_bounds_and_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        kkt_residuals([0], [[1]], [0], [0], [0], [1], variable=[0])
    with pytest.raises(ValueError):
        kkt_residuals([0, 0], [[1]], [0], [0], [0], [1])


def test_kkt_nonfinite_inputs_are_explicitly_invalid() -> None:
    residuals = kkt_residuals([np.nan], [[1]], [0], [0], [0], [1])
    assert all(np.isinf(value) for value in residuals.values())


def test_solver_summary_tolerates_missing_fields() -> None:
    result = solver_summary({"success": np.bool_(True), "return_status": "Solve_Succeeded"})
    assert result["success"] is True
    assert result["iterations"] is None
    assert solver_summary({})["success"] is None
    result = solver_summary({"n_call_nlp_f": 12}, elapsed_s=0.25)
    assert result["iterations"] is None
    assert result["objective_evaluations"] == 12
    assert result["wall_time_s"] == 0.25


@pytest.mark.parametrize("elapsed", [np.nan, np.inf, -1])
def test_solver_summary_rejects_invalid_measured_time(elapsed) -> None:
    with pytest.raises(ValueError):
        solver_summary({}, elapsed_s=elapsed)
