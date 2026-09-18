"""Generate a small self-contained CasADi C function for CI compilation."""

import os
from pathlib import Path

import casadi as ca


def main() -> None:
    """Generate source and header files under the ignored ``generated`` folder."""
    # The expression is deliberately small but includes nonlinear operations and
    # an exact Jacobian, covering the code-generation path used in chapter 19.
    x = ca.MX.sym("x", 4)
    p = ca.MX.sym("p", 4)
    expression = ca.sin(x) + p * x**2
    function = ca.Function("generated_dynamics", [x, p], [expression, ca.jacobian(expression, x)])
    output = Path("generated").resolve()
    output.mkdir(exist_ok=True)
    original = Path.cwd()

    # CasADi writes generated files to the process working directory. Restore it
    # even if generation fails so callers are not left in a surprising location.
    try:
        os.chdir(output)
        function.generate("generated_dynamics.c", {"with_header": True})
    finally:
        os.chdir(original)


if __name__ == "__main__":
    # Keeping execution behind a main guard makes this module safe to import.
    main()
