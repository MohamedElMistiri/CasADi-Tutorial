"""Plotting helpers that accept both CasADi and NumPy values."""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from casadi_tutorial.models.ur5e import joint_positions


def plot_sparsity(matrix, *, ax=None, title: str | None = None):
    """Plot structural nonzeros rather than only numerically nonzero entries."""
    target = ax or plt.gca()

    # A CasADi matrix owns a Sparsity object, while callers may also provide the
    # Sparsity directly. Triplets expose the structural row and column indices.
    sparsity = matrix if isinstance(matrix, ca.Sparsity) else matrix.sparsity()
    rows, columns = sparsity.get_triplet()
    pattern = np.zeros(sparsity.size(), dtype=bool)
    pattern[rows, columns] = True
    target.spy(pattern)

    # Reverse the vertical axis so the plot uses matrix-style row ordering.
    target.set_xlim(-0.5, sparsity.size2() - 0.5)
    target.set_ylim(sparsity.size1() - 0.5, -0.5)
    target.set_aspect("equal")
    target.set_xlabel("column")
    target.set_ylabel("row")
    target.set_title(title or f"{sparsity.size1()}x{sparsity.size2()}, nnz={sparsity.nnz()}")
    return target


def plot_ur5e(q, *, ax=None, obstacle: tuple[np.ndarray, float] | None = None):
    """Plot UR5e link-frame origins as a three-dimensional stick model."""
    target = ax or plt.figure(figsize=(7, 6)).add_subplot(projection="3d")

    # Converting at the plotting boundary keeps the kinematics symbolic while
    # giving Matplotlib ordinary numeric arrays.
    points = np.asarray(joint_positions(ca.DM(q)).full())
    target.plot(points[0], points[1], points[2], "o-", linewidth=3)

    if obstacle is not None:
        center, radius = obstacle
        azimuth, polar = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        target.plot_wireframe(
            center[0] + radius * np.cos(azimuth) * np.sin(polar),
            center[1] + radius * np.sin(azimuth) * np.sin(polar),
            center[2] + radius * np.cos(polar),
            color="tab:red",
            alpha=0.25,
        )

    target.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    target.set_box_aspect((1, 1, 1))
    return target
