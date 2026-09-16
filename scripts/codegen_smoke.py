"""Generate a small self-contained CasADi C function for CI compilation."""

import os
from pathlib import Path

import casadi as ca


def main() -> None:
    x = ca.MX.sym("x", 4)
    p = ca.MX.sym("p", 4)
    expression = ca.sin(x) + p * x**2
    function = ca.Function(
        "generated_dynamics", [x, p], [expression, ca.jacobian(expression, x)]
    )
    output = Path("generated").resolve()
    output.mkdir(exist_ok=True)
    original = Path.cwd()
    try:
        os.chdir(output)
        function.generate("generated_dynamics.c", {"with_header": True})
    finally:
        os.chdir(original)


if __name__ == "__main__":
    main()
