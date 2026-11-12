# CasADi Python Tutorial

This course teaches CasADi as composable numerical building blocks, from sparse
symbolic graphs and exact derivatives to estimation and feedback control.

Use the navigation in order on a first pass. Each chapter explains its equations,
dimensions, assumptions, and numerical checks before guided experiments. Run a
notebook from a fresh kernel before changing controls, then predict a result and
test one change at a time.

The CSTR path derives balances, fits heat transfer, reconstructs states from
temperature measurements, and compares estimated-state NMPC with ideal feedback.
The UR5e path derives DH kinematics, calibrates geometry and encoder offsets,
estimates joint states, and closes a feedback loop around an obstacle-avoiding
reference. Every capstone defines its own model and optimization formulations.

Browser plots support hover, zoom, and 3D rotation immediately. Each lesson also
has an **Enable interactive session** button: start a free Python session, edit
code, and use **Run experiment** to recompute results within the page. The first
startup may take a few minutes; saved results remain available while you wait.
Sessions are temporary, so download notebooks to keep your work. **Open in
JupyterLab** provides a full notebook workspace for longer experiments.

You can also download a notebook from its page and run it locally from the
repository checkout. See the installation instructions in the README.

If a solve fails, start with chapter 18 and the 18a failure labs: inspect initial
evaluations, identify the variable or constraint, repair the formulation, and
validate the result independently. A solver success flag alone is insufficient.

!!! note "What CasADi is"
    CasADi constructs differentiable expression graphs and connects them to
    numerical algorithms. It is neither a general computer algebra system nor a
    one-call globally optimal OCP solver.

See the [repository README](https://github.com/MohamedElMistiri/CasADi-Tutorial#readme)
for installation and validation commands.
