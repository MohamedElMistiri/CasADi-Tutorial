import casadi as ca
import numpy as np

from casadi_tutorial.integration import euler_step, rk4_step


def test_rk4_is_more_accurate_than_euler() -> None:
    def rhs(x, _u):
        return -x

    euler = float(euler_step(rhs, ca.DM(1), ca.DM(), 0.2))
    rk4 = float(rk4_step(rhs, ca.DM(1), ca.DM(), 0.2))
    exact = np.exp(-0.2)
    assert abs(rk4 - exact) < abs(euler - exact)
    assert abs(rk4 - exact) < 1e-5


def test_rk4_symbolic_jacobian() -> None:
    x = ca.MX.sym("x")
    step = rk4_step(lambda state, _u: -2 * state, x, ca.MX(), 0.1)
    derivative = ca.Function("step_derivative", [x], [ca.jacobian(step, x)])
    assert 0 < float(derivative(1)) < 1
