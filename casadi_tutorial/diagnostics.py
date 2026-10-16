"""Solver-independent checks in the original problem's units.

Vectors follow CasADi's column-major ``vec`` convention. These calculations
assess a candidate point; a small residual is not a global-optimality proof.
"""

from typing import Any

import numpy as np


def _vector(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1, order="F")


def _bounds(value, lower, upper):
    values = _vector(value)

    def expand(bound):
        result = _vector(bound)
        if result.size == 1:
            return np.full(values.size, result.item())
        if result.size != values.size:
            raise ValueError("Bounds must be scalar or contain one entry per value.")
        return result

    lo, hi = expand(lower), expand(upper)
    if (
        np.isnan(lo).any()
        or np.isnan(hi).any()
        or np.isposinf(lo).any()
        or np.isneginf(hi).any()
        or (lo > hi).any()
    ):
        raise ValueError("Bounds must satisfy lower <= upper with valid open sides.")
    return values, lo, hi


def bound_report(value, lower, upper, *, labels=None) -> list[dict[str, Any]]:
    """Return one labelled record per entry, in column-major order.

    Bounds can be scalar or have exactly as many elements as ``value``.
    A NaN or infinite candidate is invalid even on an unbounded row, and has
    infinite violation. Infinite bounds denote open sides, never candidates.
    """
    values, lo, hi = _bounds(value, lower, upper)
    names = [f"row[{i}]" for i in range(values.size)] if labels is None else list(labels)
    if len(names) != values.size:
        raise ValueError("Supply one label per flattened value.")
    with np.errstate(invalid="ignore"):
        violations = np.maximum(0.0, np.maximum(lo - values, values - hi))
    violations[~np.isfinite(values)] = np.inf
    return [
        {
            "index": i,
            "label": str(names[i]),
            "value": float(values[i]),
            "lower": float(lo[i]),
            "upper": float(hi[i]),
            "violation": float(violations[i]),
            "finite": bool(np.isfinite(values[i])),
        }
        for i in range(values.size)
    ]


def max_bound_violation(value, lower, upper) -> float:
    """Return the largest bound violation; invalid candidates return infinity."""
    return max((row["violation"] for row in bound_report(value, lower, upper)), default=0.0)


def _norm_inf(value) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _bound_optimality(values, lower, upper, multipliers):
    # CasADi combines upper-minus-lower multipliers into one signed vector.
    equality = lower == upper
    has_lower = np.isfinite(lower) & ~equality
    has_upper = np.isfinite(upper) & ~equality
    mu_lower = np.maximum(-multipliers, 0.0)
    mu_upper = np.maximum(multipliers, 0.0)
    # Mask before multiplying: zero times an infinite slack would be NaN.
    complementarity = max(
        _norm_inf(mu_lower[has_lower] * (values[has_lower] - lower[has_lower])),
        _norm_inf(mu_upper[has_upper] * (upper[has_upper] - values[has_upper])),
    )
    wrong_lower = multipliers[has_lower & ~has_upper]
    wrong_upper = -multipliers[has_upper & ~has_lower]
    unbounded = multipliers[~has_lower & ~has_upper & ~equality]
    dual = max(
        float(np.max(wrong_lower, initial=0.0)),
        float(np.max(wrong_upper, initial=0.0)),
        _norm_inf(unbounded),
    )
    return complementarity, dual


def kkt_residuals(
    gradient_f,
    jacobian_g,
    lambda_g,
    constraint,
    lower,
    upper,
    *,
    variable=None,
    lower_x=None,
    upper_x=None,
    lambda_x=None,
) -> dict[str, float]:
    """Compute primal, stationarity, complementarity and dual-sign residuals.

    The original six positional arguments remain supported. Supply all four
    decision-bound keywords to include variable bounds and ``lam_x``; omitting
    them means the candidate has no decision bounds. Stationarity uses
    ``gradient_f + jacobian_g.T @ lambda_g + lambda_x``.

    Negative signed multipliers act on lower bounds, positive ones on upper
    bounds. Equality multipliers are unrestricted. Complementarity uses the
    positive/negative split of net signed multipliers, not IPOPT's separate
    interior barrier multipliers. Residuals use original units and are not
    directly comparable to a solver's internally scaled stopping criteria.
    """
    gradient = _vector(gradient_f)
    values, lo, hi = _bounds(constraint, lower, upper)
    multipliers = _vector(lambda_g)
    jacobian = np.asarray(jacobian_g, dtype=float)
    if jacobian.ndim == 0 and (values.size, gradient.size) == (1, 1):
        jacobian = jacobian.reshape(1, 1)
    if jacobian.shape != (values.size, gradient.size) or multipliers.size != values.size:
        raise ValueError("Jacobian and constraint multipliers must match the problem dimensions.")

    optional = (variable, lower_x, upper_x, lambda_x)
    if any(item is not None for item in optional) and not all(
        item is not None for item in optional
    ):
        raise ValueError("Supply variable, lower_x, upper_x and lambda_x together.")
    if variable is None:
        x_values = x_lower = x_upper = x_multipliers = np.empty(0)
        bound_stationarity = np.zeros_like(gradient)
    else:
        x_values, x_lower, x_upper = _bounds(variable, lower_x, upper_x)
        x_multipliers = _vector(lambda_x)
        if x_values.size != gradient.size or x_multipliers.size != gradient.size:
            raise ValueError("Decision values and bound multipliers must match the gradient.")
        bound_stationarity = x_multipliers

    names = ("stationarity_inf", "primal_inf", "complementarity_inf", "dual_feasibility_inf")
    if not all(
        np.isfinite(item).all()
        for item in (gradient, jacobian, multipliers, values, x_values, x_multipliers)
    ):
        return dict.fromkeys(names, float("inf"))

    g_complementarity, g_dual = _bound_optimality(values, lo, hi, multipliers)
    x_complementarity, x_dual = _bound_optimality(x_values, x_lower, x_upper, x_multipliers)
    return {
        "stationarity_inf": _norm_inf(gradient + jacobian.T @ multipliers + bound_stationarity),
        "primal_inf": max(
            max_bound_violation(values, lo, hi),
            max_bound_violation(x_values, x_lower, x_upper),
        ),
        "complementarity_inf": max(g_complementarity, x_complementarity),
        "dual_feasibility_inf": max(g_dual, x_dual),
    }


def solver_summary(stats: dict[str, Any], *, elapsed_s=None) -> dict[str, Any]:
    """Extract status without confusing missing data with failure or iterations.

    Pass elapsed wall time measured around the solve with ``perf_counter`` if
    available. Plugins may omit their aggregate timing and evaluation counts.
    """
    if elapsed_s is not None and (not np.isfinite(elapsed_s) or elapsed_s < 0):
        raise ValueError("Elapsed time must be finite and nonnegative.")
    success = stats.get("success")
    return {
        "success": None if success is None else bool(success),
        "return_status": stats.get("return_status") or "unknown",
        "iterations": stats.get("iter_count"),
        "wall_time_s": elapsed_s if elapsed_s is not None else stats.get("t_wall_total"),
        "objective_evaluations": stats.get("n_call_nlp_f"),
        "constraint_evaluations": stats.get("n_call_nlp_g"),
        "unified_return_status": stats.get("unified_return_status"),
    }
