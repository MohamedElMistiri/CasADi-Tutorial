import casadi as ca
import numpy as np


def test_ipopt_solves_parameterized_nlp() -> None:
    x = ca.MX.sym("x", 2)
    parameter = ca.MX.sym("parameter")
    problem = {
        "x": x,
        "p": parameter,
        "f": (x[0] - parameter) ** 2 + (x[1] - 1) ** 2,
        "g": x[0] + x[1],
    }
    solver = ca.nlpsol(
        "test_ipopt", "ipopt", problem, {"ipopt.print_level": 0, "print_time": False}
    )
    result = solver(x0=[0, 0], p=2, lbx=0, ubx=3, lbg=1, ubg=ca.inf)
    assert solver.stats()["success"]
    assert np.isfinite(result["x"].full()).all()
    assert float(result["g"]) >= 1 - 1e-8
