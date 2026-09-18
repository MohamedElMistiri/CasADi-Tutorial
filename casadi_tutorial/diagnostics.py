"""Small, solver-independent calculations used by the debugging lessons."""

from typing import Any

import numpy as np


def max_bound_violation(value, lower, upper) -> float:
    """Return the largest violation of elementwise lower and upper bounds."""
    # Flatten all values so scalar and matrix-shaped inputs share one comparison
    # path. Bounds are broadcast to support scalar bounds on vector decisions.
    value_array = np.asarray(value, dtype=float).ravel()
    lower_array = np.broadcast_to(np.asarray(lower, dtype=float), value_array.shape)
    upper_array = np.broadcast_to(np.asarray(upper, dtype=float), value_array.shape)

    lower_violation = np.max(lower_array - value_array, initial=-np.inf)
    upper_violation = np.max(value_array - upper_array, initial=-np.inf)
    return float(max(0.0, lower_violation, upper_violation))


def kkt_residuals(
    gradient_f,
    jacobian_g,
    lambda_g,
    constraint,
    lower,
    upper,
) -> dict[str, float]:
    """Compute stationarity and primal-feasibility infinity norms.

    This compact helper diagnoses equality and bounded constraints.
    Complementarity for variable bounds is solver-specific and should be read
    from the solver statistics instead.
    """
    gradient = np.asarray(gradient_f, dtype=float).reshape(-1)
    jacobian = np.asarray(jacobian_g, dtype=float)
    multipliers = np.asarray(lambda_g, dtype=float).reshape(-1)

    # For L = f + lambda^T g, stationarity requires this vector to be zero.
    stationarity = gradient + jacobian.T @ multipliers
    return {
        "stationarity_inf": float(np.linalg.norm(stationarity, ord=np.inf)),
        "primal_inf": max_bound_violation(constraint, lower, upper),
    }


def solver_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Extract stable, useful fields from a CasADi statistics dictionary."""
    # CasADi plugins do not all report identical keys, so each field has a safe
    # fallback that keeps the tutorial diagnostics plugin-independent.
    return {
        "success": bool(stats.get("success", False)),
        "return_status": stats.get("return_status", "unknown"),
        "iterations": stats.get("iter_count", stats.get("n_call_nlp_f")),
        "wall_time_s": stats.get("t_wall_total"),
    }
