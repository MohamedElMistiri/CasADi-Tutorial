"""Solver-independent diagnostic calculations used by the debugging lessons."""

from typing import Any

import numpy as np


def max_bound_violation(value, lower, upper) -> float:
    """Return the largest violation of elementwise lower and upper bounds."""
    value_array = np.asarray(value, dtype=float).ravel()
    lower_array = np.broadcast_to(np.asarray(lower, dtype=float), value_array.shape)
    upper_array = np.broadcast_to(np.asarray(upper, dtype=float), value_array.shape)
    return float(
        max(
            0.0,
            np.max(lower_array - value_array, initial=-np.inf),
            np.max(value_array - upper_array, initial=-np.inf),
        )
    )


def kkt_residuals(gradient_f, jacobian_g, lambda_g, constraint, lower, upper) -> dict[str, float]:
    """Compute stationarity and primal-feasibility infinity norms.

    This compact helper diagnoses equality/bounded constraints. Complementarity
    for variable bounds is solver-specific and should be read from solver stats.
    """
    gradient = np.asarray(gradient_f, dtype=float).reshape(-1)
    jacobian = np.asarray(jacobian_g, dtype=float)
    multipliers = np.asarray(lambda_g, dtype=float).reshape(-1)
    stationarity = gradient + jacobian.T @ multipliers
    return {
        "stationarity_inf": float(np.linalg.norm(stationarity, ord=np.inf)),
        "primal_inf": max_bound_violation(constraint, lower, upper),
    }


def solver_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Extract stable, useful fields from a CasADi solver statistics dictionary."""
    return {
        "success": bool(stats.get("success", False)),
        "return_status": stats.get("return_status", "unknown"),
        "iterations": stats.get("iter_count", stats.get("n_call_nlp_f", None)),
        "wall_time_s": stats.get("t_wall_total", None),
    }
