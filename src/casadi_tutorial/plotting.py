"""Plotting helpers that accept CasADi and NumPy values."""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from casadi_tutorial.models.ur5e import joint_positions


def plot_sparsity(matrix, *, ax=None, title: str | None = None):
    """Plot structural nonzeros, which differ from numerically nonzero entries."""
    target = ax or plt.gca()
    sparsity = matrix if isinstance(matrix, ca.Sparsity) else matrix.sparsity()
    rows, cols = sparsity.get_triplet()
    target.scatter(cols, rows, marker="s", s=18)
    target.set_xlim(-0.5, sparsity.size2() - 0.5)
    target.set_ylim(sparsity.size1() - 0.5, -0.5)
    target.set_aspect("equal")
    target.set_xlabel("column")
    target.set_ylabel("row")
    target.set_title(title or f"{sparsity.size1()}x{sparsity.size2()}, nnz={sparsity.nnz()}")
    return target


def plot_ur5e(q, *, ax=None, obstacle: tuple[np.ndarray, float] | None = None):
    """Plot UR5e link-frame origins as a 3-D stick model."""
    target = ax or plt.figure(figsize=(7, 6)).add_subplot(projection="3d")
    points = np.asarray(joint_positions(ca.DM(q)).full())
    target.plot(points[0], points[1], points[2], "o-", linewidth=3)
    if obstacle is not None:
        center, radius = obstacle
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        target.plot_wireframe(
            center[0] + radius * np.cos(u) * np.sin(v),
            center[1] + radius * np.sin(u) * np.sin(v),
            center[2] + radius * np.cos(v),
            color="tab:red",
            alpha=0.25,
        )
    target.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    target.set_box_aspect((1, 1, 1))
    return target
