# Editable helper API and conventions

The notebooks keep optimization transcriptions visible. Shared helpers only
remove repetition from models, numerical integration, plots, and diagnostics.
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
- `casadi_tutorial.plotting`: sparsity-pattern and robot visualization.
- `casadi.tools.graph`: CasADi's optional Graphviz utilities. The graph-tools
  lesson uses `dotdraw` to render an SX or MX expression; the Conda environment
  supplies its `pydot` and Graphviz dependencies.
- `casadi_tutorial.diagnostics`: solver summaries, bound violations, and KKT
  residual calculations.

The `casadi_tutorial.models.cstr` package re-exports the complete functional API,
so existing imports of `CSTRParameters`, `cstr_rhs`, `cstr_function`,
`cstr_rk4_function`, `steady_state_residual`, and `linearize_cstr` remain valid.
