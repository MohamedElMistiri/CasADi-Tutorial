# CasADi Python Tutorial

A notebook-first course for learning [CasADi](https://web.casadi.org/) from symbolic
fundamentals through nonlinear optimization, optimal control, NMPC, sparse
derivatives, and solver debugging.

The course targets **Python 3.10–3.13** and **CasADi 3.8.x**. Readers should know
basic Python, NumPy, plotting, and single-variable calculus; optimization and
control theory are introduced as needed.

## Start here

```bash
uv sync --all-extras
uv run jupyter lab
```

Alternatively, use `python -m pip install -e ".[tutorial,test,docs]"` in an
activated virtual environment. The committed `uv.lock` is the reproducible course
and CI environment.

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
[`solutions/`](solutions/). Reusable, typed examples are in
[`src/casadi_tutorial/`](src/casadi_tutorial/).

## Quality checks

```bash
uv run pytest
uv run pytest --nbmake notebooks solutions
uv run mkdocs build --strict
uv run ruff check .
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
