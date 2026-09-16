import numpy as np

from casadi_tutorial.diagnostics import kkt_residuals, max_bound_violation, solver_summary


def test_bound_violation() -> None:
    assert max_bound_violation([0, 3], [0, 0], [2, 2]) == 1.0
    assert max_bound_violation([0, 2], [0, 0], [2, 2]) == 0.0


def test_kkt_residuals() -> None:
    residuals = kkt_residuals([-1, -1], [[1, 1]], [1], [2], [2], [2])
    assert residuals == {"stationarity_inf": 0.0, "primal_inf": 0.0}


def test_solver_summary_tolerates_missing_fields() -> None:
    result = solver_summary({"success": np.bool_(True), "return_status": "Solve_Succeeded"})
    assert result["success"] is True
    assert result["iterations"] is None
