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

Run these commands from the repository root. Notebooks import presentation,
integration, and diagnostic helpers directly from the checkout and enable
IPython autoreload. CSTR and UR5e lessons define their physical models,
estimation problems, and control transcriptions inside each notebook; they do
not import models or depend on another notebook's kernel state. Update with
`conda env update -f environment.yml --prune`.

Open [`notebooks/00_course_orientation.ipynb`](notebooks/00_course_orientation.ipynb),
then follow the numbered notebooks. The concise route is 00–05, 08–11, 14–17;
the complete route includes every chapter and the linked 14a, 16a, 16b, and 18a
labs. Each notebook runs independently; prerequisites describe concepts rather
than shared kernel state.

## Explore the experiments

Each chapter explains its equations, array dimensions, numerical assumptions,
and expected results before asking you to change them. Default results run
without touching any controls. Plotly plots support hover, zoom, trace toggles,
and rotatable robot views. Python experiments use explicit **Run experiment**
buttons; **Reset controls** restores the original settings without solving.

The capstones include:

- CSTR balances, linearization, heat-transfer fitting, temperature-only moving
  horizon estimation, and estimated-state versus ideal-state NMPC.
- UR5e DH kinematics, geometric calibration, joint-state estimation, an
  obstacle-avoiding reference, and estimated-state feedback control.
- Solver failure labs with initial-value checks, named constraint violations,
  KKT residuals, convergence histories, and verifiable repairs.

Set `CASADI_TUTORIAL_WIDGETS=0` to omit Python controls in headless runs. Set
`CASADI_TUTORIAL_PLOTS=static` for Matplotlib instead of Plotly. These settings
change presentation only; default calculations still run. Static documentation
retains computed results and browser-native plot interaction. Lesson pages can
start an on-demand Binder kernel through Thebe to run Python and experiment
controls directly in the page. Public Binder requires a public tutorial
repository, and its sessions are temporary. Local JupyterLab remains available.

## Course map

| Part | Chapters | Outcome |
|---|---:|---|
| Foundations | 00–05 | Build symbolic graphs, functions, sparse matrices, and exact derivatives |
| Numerical tools | 06–10 | Solve roots, ODEs/DAEs, NLPs, QPs, and use `Opti` |
| Optimal control | 11–13 | Understand KKT systems and implement shooting/collocation transcriptions |
| Capstones | 14–17, 14a, 16a–16b | Derive, calibrate, estimate, and control the CSTR and UR5e |
| Expert/reference | 18–20, 18a | Diagnose failures, measure performance, and use codegen/model-building tools |

Exercises live in [`exercises/`](exercises/) and complete worked versions in
[`solutions/`](solutions/). Editable teaching modules are in
[`casadi_tutorial/`](casadi_tutorial/); they are intentionally not configured as
a distributable Python project while the course is evolving.

## Quality checks

```bash
ruff check .
ruff format --check casadi_tutorial scripts tests
pytest
pytest --nbmake --nbmake-timeout=600 notebooks solutions
mkdocs build --strict
```

## Personal website integration

The course and `MohamedElMistiri.github.io` remain separate repositories. The
website's **Deploy site** workflow checks out this course, executes the lessons,
and includes the built pages at `/tutorials/casadi/`. The same course commit is
embedded in each page's Binder configuration. Publish the course changes before
running that website workflow; public Binder needs a publicly accessible course
repository. Repository visibility is managed separately from local development.

After updating the course, rerun **Deploy site** in the website repository. Its
optional `tutorial_ref` input accepts a branch, tag, or commit. The workflow also
accepts a `casadi-tutorial-updated` repository dispatch event for external automation.
No access token is embedded in the published tutorial. `binder/environment.yml`
contains the runtime environment; development/build tools stay in `environment.yml`.

For a local preview, build the Jekyll website first (using its Docker setup or
`bundle exec jekyll build`), then run these from this checkout's course environment:

```bash
python scripts/build_tutorial.py --output ../MohamedElMistiri.github.io/_site/tutorials/casadi --site-url http://127.0.0.1:8090/tutorials/casadi/ --personal-site http://127.0.0.1:8090/
python scripts/serve_website_preview.py --site-dir ../MohamedElMistiri.github.io/_site
```

Open `http://127.0.0.1:8090/`. Interactive sessions in this preview use a temporary
local Jupyter server instead of public Binder. Credentials are injected into
responses in memory only. Stop the preview with Ctrl+C. Normal production pages
start Binder only when **Enable interactive session** is clicked; saved results
remain available without a live session.

With Playwright installed, `python scripts/validate_website_preview.py` checks
the actual website navigation and the three representative interactive lessons.
The earlier standalone validation scripts remain available for numerical checks.

Notebooks use small deterministic problems so they run on a laptop. Mesh
comparisons and independent residual checks distinguish convergence from
discretization accuracy. Documentation builds execute staged copies of lessons
and solutions without overwriting source notebooks. Exercise scaffolds render
without execution. CI executes all lessons and solutions; exact iteration
counts and runtimes are deliberately not acceptance criteria.

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
