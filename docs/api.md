# Editable helper API and conventions

The notebooks keep optimization transcriptions visible. Capstones define their physical models, estimation, and control locally; shared
helpers remove repetition from integration, plots, widgets, and diagnostics.
They live in the repository-level `casadi_tutorial/` directory and do not need
to be installed. Notebook setup cells add the checkout to the import path and
enable autoreload for interactive model development.

## Conventions

- Import CasADi as `import casadi as ca`; avoid wildcard imports.
- State vectors use `x`, manipulated inputs use `u`, disturbances use `d`,
  and fixed model parameters use `p` or `theta`. All are column vectors.
- Dynamic lessons plot the state history `x(t)` and manipulated-input history
  `u(t)` so transient behavior and held-input timing remain visible.
- CasADi matrices use column-major structural sparsity.
- Convert final numerical results with `np.asarray(dm.full())`.
- SI units are used unless a model explicitly states otherwise.
- Random examples use `np.random.default_rng(seed)`.

## Modules

- `casadi_tutorial.models.cstr.parameters`: physical constants and nominal values.
- `casadi_tutorial.models.cstr.dynamics`: reaction rate, nonlinear RHS, and
  continuous CasADi function construction.
- `casadi_tutorial.models.cstr.analysis`: steady-state residuals and Jacobian
  linearization.
- `casadi_tutorial.models.cstr.system`: RK4 construction and the `CSTRSystem`
  convenience facade. The facade builds fresh graphs on every call so edits are
  visible after notebook autoreload.
- `casadi_tutorial.models.ur5e`: DH transforms, forward kinematics, geometric
  Jacobian, points along the arm, and joint double-integrator dynamics.
- `casadi_tutorial.integration`: Euler and RK4 graph builders.
- `casadi_tutorial.plotting`: structural sparsity, numeric histories, and robot
  frame-origin visualization. `plot_series(time, series, ...)` accepts labeled
  arrays; `plot_robot(points, ...)` accepts a notebook-computed `3 x N` array.
  `display_plot` presents Plotly figures or Matplotlib output. The compatibility
  `plot_ur5e` wrapper imports its packaged model only when explicitly called.
- `casadi_tutorial.interactive`: `experiment(run, controls, title=...)` displays
  explicit Run/Reset controls without automatically invoking the calculation.
  Deterministic defaults belong in ordinary cells outside callbacks.
- `casadi.tools.graph`: CasADi's optional Graphviz utilities. The graph-tools
  lesson uses `dotdraw` to render an SX or MX expression; the Conda environment
  supplies its `pydot` and Graphviz dependencies.
- `casadi_tutorial.diagnostics`: solver summaries, bound violations, and KKT
  residual calculations.

The `casadi_tutorial.models.cstr` package re-exports the complete functional API,
so existing imports of `CSTRParameters`, `cstr_rhs`, `cstr_function`,
`cstr_rk4_function`, `steady_state_residual`, and `linearize_cstr` remain valid.


## Diagnostic contracts

`max_bound_violation(value, lower, upper)` accepts scalar bounds or matching
element counts. Matrix inputs flatten by columns, like CasADi `vec`. Invalid
bounds raise `ValueError`; a nonfinite candidate has infinite violation.
`bound_report(..., labels=...)` retains row indices for stage/variable labels.

`kkt_residuals(gradient_f, jacobian_g, lambda_g, constraint, lower, upper, ...)`
preserves its six positional arguments. Supply `variable`, `lower_x`, `upper_x`,
and `lambda_x` together to include decision bounds. Results report primal
feasibility, stationarity, complementarity, and dual feasibility. With CasADi's
net signed multipliers, stationarity is
`gradient_f + jacobian_g.T @ lambda_g + lambda_x`; negative multipliers belong
to lower bounds and positive multipliers to upper bounds. Equality multipliers
are unrestricted. These checks do not reproduce internal barrier complementarity.

`solver_summary(stats, elapsed_s=...)` separates iterations from objective and
constraint evaluations. Missing fields remain unknown. Measure time with
`time.perf_counter()` rather than assuming all plugins report wall-clock time.

## Presentation modes and standalone capstones

Set `CASADI_TUTORIAL_WIDGETS=0` to omit Python controls; set
`CASADI_TUTORIAL_PLOTS=static` for Matplotlib. These modes do not skip defaults.
Documentation builds disable Python controls and execute staged copies with
errors fatal. Plotly interaction on precomputed results works in the browser;
changing solver inputs needs a live kernel.

The CSTR and UR5e lessons do not import the models package, execute neighboring
notebooks, or require prior result files. They include all required physical
parameters, equations, discretization, and optimization transcriptions. The
packaged models remain available for compatibility and independent comparisons.
