# Helper API and conventions

The notebooks keep optimization transcriptions visible. Shared helpers only
remove repetition from models, numerical integration, plots, and diagnostics.

## Conventions

- Import CasADi as `import casadi as ca`; avoid wildcard imports.
- State, control, and parameter vectors are column vectors.
- CasADi matrices use column-major structural sparsity.
- Convert final numerical results with `np.asarray(dm.full())`.
- SI units are used unless a model explicitly states otherwise.
- Random examples use `np.random.default_rng(seed)`.

## Modules

- `casadi_tutorial.models.cstr`: parameters, nonlinear RHS, steady-state residual,
  Jacobian linearization, and one-step integrators.
- `casadi_tutorial.models.ur5e`: DH transforms, forward kinematics, geometric
  Jacobian, points along the arm, and joint double-integrator dynamics.
- `casadi_tutorial.integration`: Euler and RK4 graph builders.
- `casadi_tutorial.plotting`: sparsity, trajectory, and robot visualization.
- `casadi_tutorial.diagnostics`: solver summaries, bound violations, and KKT
  residual calculations.
