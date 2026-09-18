# CasADi Python Tutorial

A notebook-first course for learning [CasADi](https://web.casadi.org/) from symbolic
fundamentals through nonlinear optimization, optimal control, NMPC, sparse
derivatives, and solver debugging.

The course targets **Python 3.10–3.13** and **CasADi 3.7.2**. Readers should know
basic Python, NumPy, plotting, and single-variable calculus; optimization and
control theory are introduced as needed.

## Start here

```bash
conda env create -f environment.yml
conda activate casadi-tutorial
jupyter lab
```

Run these commands from the repository root. The notebooks import the editable
`casadi_tutorial/` modules directly from the checkout and enable IPython
autoreload, so model changes do not require a package installation or kernel
restart. Update an existing environment with
`conda env update -f environment.yml --prune`.

Open [`notebooks/00_course_orientation.ipynb`](notebooks/00_course_orientation.ipynb),
then follow the numbered notebooks. The concise route is 00–05, 08–11, 14–17;
the complete route includes every chapter.

## Course map

| Part | Chapters | Outcome |
|---|---:|---|
| Foundations | 00–05 | Build symbolic graphs, functions, sparse matrices, and exact derivatives |
| Numerical tools | 06–10 | Solve roots, ODEs/DAEs, NLPs, QPs, and use `Opti` |
| Optimal control | 11–13 | Understand KKT systems and implement shooting/collocation transcriptions |
| Capstones | 14–17 | Estimate/control a CSTR and plan collision-aware UR5e motion |
| Expert/reference | 18–20 | Diagnose solvers, improve performance, and use codegen/callback/FMI tools |

Exercises live in [`exercises/`](exercises/) and complete worked versions in
[`solutions/`](solutions/). Editable teaching modules are in
[`casadi_tutorial/`](casadi_tutorial/); they are intentionally not configured as
a distributable Python project while the course is evolving.

## Quality checks

```bash
ruff check .
ruff format --check casadi_tutorial scripts tests
pytest
pytest --nbmake notebooks solutions
mkdocs build --strict
```

Notebooks use small deterministic problems so they run on a laptop. Larger meshes
are described where they improve accuracy, but are not required for CI.

## Important modeling note

The UR5e capstone uses the robot's six-joint kinematics with a joint-space
double-integrator model. It is an intentionally transparent OCP teaching model,
not a torque-level rigid-body model and not a substitute for a safety-rated robot
controller.

## References

- [CasADi user guide](https://web.casadi.org/docs/)
- [CasADi Python API](https://web.casadi.org/python-api/)
- Andersson et al., *CasADi – A software framework for nonlinear optimization and optimal control*, 2018.

CasADi itself is LGPL-licensed; consult its project license when distributing
applications that use it.
