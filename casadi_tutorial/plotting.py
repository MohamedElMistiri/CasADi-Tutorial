"""Model-independent plots with optional browser interaction.

Plotly is selected when installed. Set ``CASADI_TUTORIAL_PLOTS=static`` or pass
``backend='matplotlib'`` for ordinary Matplotlib output. Helpers construct and
return figures/axes; ``display_plot`` presents either kind in a notebook.
"""

import json
import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def _plotly_backend(backend, ax=None):
    if backend not in {"auto", "plotly", "matplotlib"}:
        raise ValueError("backend must be auto, plotly or matplotlib")
    if ax is not None or backend == "matplotlib" or os.getenv("CASADI_TUTORIAL_PLOTS") == "static":
        return None
    # Lazy imports keep basic lessons usable without optional interactive tools.
    try:
        import plotly.graph_objects as go
    except ImportError:
        if backend == "plotly":
            raise
        return None
    return go


def display_plot(figure):
    """Display a Plotly figure or Matplotlib figure/axes without opening a browser.

    The combined renderer supports JupyterLab and HTML exports. Matplotlib PNGs
    come directly from Matplotlib, so a browser screenshot engine is unnecessary.
    """
    if hasattr(figure, "to_plotly_json"):
        from IPython.display import display
        from plotly.offline import get_plotlyjs_version

        # Native MIME data serves JupyterLab; HTML serves exports and Thebe.
        # Share one library per page and use the site's existing MathJax.
        body = figure.to_html(
            full_html=False, include_plotlyjs=False, include_mathjax=False, default_height=525
        )
        loader = """
window.casadiPlotlyReady = window.casadiPlotlyReady || new Promise((resolve, reject) => {
  if (window.Plotly) { resolve(); return; }
  const script = document.createElement('script');
  script.src = 'https://cdn.plot.ly/plotly-VERSION.min.js';
  script.onload = resolve;
  script.onerror = () => {
    window.casadiPlotlyReady = null;
    reject(Error('Plotly could not load'));
  };
  document.head.append(script);
});
window.casadiPlotlyReady.then(() => {
""".replace("VERSION", get_plotlyjs_version())
        # Plotly versions emit either a plain script tag or one with type=.
        opening = body.index(">", body.index("<script")) + 1
        body = body[:opening] + loader + body[opening:]
        body = body.replace("</script>", "});</script>", 1)
        display(
            {"application/vnd.plotly.v1+json": json.loads(figure.to_json()), "text/html": body},
            raw=True,
        )
    else:
        from IPython.display import display

        target = figure.figure if hasattr(figure, "figure") else figure
        display(target)
        plt.close(target)


def plot_series(
    time,
    series,
    *,
    title="",
    ylabel="",
    xlabel="time",
    labels=None,
    ax=None,
    backend="auto",
    steps=False,
    log_y=False,
):
    """Plot named numeric histories, with hover/zoom when Plotly is available.

    ``series`` is a label-to-array dictionary, one vector, or a channels-by-time
    matrix. Every channel must have ``len(time)`` entries. ``steps=True`` means
    sample-and-hold inputs; ``log_y=True`` is useful for positive residual norms.
    An explicit Matplotlib axes always selects Matplotlib.
    """
    times = np.asarray(time, dtype=float).reshape(-1)
    if isinstance(series, dict):
        names = list(series)
        channels = [np.asarray(values, dtype=float).reshape(-1) for values in series.values()]
    else:
        values = np.asarray(series, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("series must be a vector, channels-by-time matrix or dictionary")
        channels = list(values)
        names = (
            list(labels) if labels is not None else [f"series {i}" for i in range(len(channels))]
        )
    if len(names) != len(channels) or any(len(channel) != len(times) for channel in channels):
        raise ValueError("Supply one label per channel and one value per time.")
    go = _plotly_backend(backend, ax)
    if go is not None:
        figure = go.Figure()
        for name, channel in zip(names, channels, strict=True):
            figure.add_trace(
                go.Scatter(
                    x=times,
                    y=channel,
                    name=str(name),
                    mode="lines",
                    line={"shape": "hv" if steps else "linear"},
                )
            )
        figure.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
        if log_y:
            figure.update_yaxes(type="log")
        return figure
    target = ax if ax is not None else plt.subplots(figsize=(8, 4))[1]
    for name, channel in zip(names, channels, strict=True):
        if steps:
            target.step(times, channel, where="post", label=str(name))
        else:
            target.plot(times, channel, label=str(name))
    target.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if log_y:
        target.set_yscale("log")
    if channels:
        target.legend()
    target.grid(alpha=0.25)
    return target


def plot_sparsity(matrix, *, ax=None, title: str | None = None):
    """Plot structural nonzeros rather than only numerically nonzero entries."""
    target = ax if ax is not None else plt.gca()
    sparsity = matrix if isinstance(matrix, ca.Sparsity) else matrix.sparsity()
    rows, columns = sparsity.get_triplet()
    pattern = np.zeros(sparsity.size(), dtype=bool)
    pattern[rows, columns] = True
    target.spy(pattern)
    target.set_xlim(-0.5, sparsity.size2() - 0.5)
    target.set_ylim(sparsity.size1() - 0.5, -0.5)
    target.set_aspect("equal")
    target.set_xlabel("column")
    target.set_ylabel("row")
    target.set_title(title or f"{sparsity.size1()}x{sparsity.size2()}, nnz={sparsity.nnz()}")
    return target


def plot_robot(points, *, ax=None, obstacle=None, title="", backend="auto"):
    """Plot notebook-computed link origins as a 3-by-N array in metres.

    ``obstacle`` is an optional ``(center, radius)`` sphere. No model is imported:
    learners can supply their own nominal or calibrated forward kinematics.
    """
    positions = np.asarray(points, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 1:
        raise ValueError("points must be a finite 3-by-N array")
    if not np.isfinite(positions).all():
        raise ValueError("points must be finite")
    sphere = None
    if obstacle is not None:
        center, radius = obstacle
        center = np.asarray(center, dtype=float).reshape(-1)
        if (
            center.size != 3
            or not np.isfinite(center).all()
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("An obstacle needs a finite 3-vector center and positive radius.")
        azimuth, polar = np.mgrid[0 : 2 * np.pi : 25j, 0 : np.pi : 15j]
        sphere = (
            center[0] + radius * np.cos(azimuth) * np.sin(polar),
            center[1] + radius * np.sin(azimuth) * np.sin(polar),
            center[2] + radius * np.cos(polar),
        )
    go = _plotly_backend(backend, ax)
    if go is not None:
        figure = go.Figure(
            go.Scatter3d(
                x=positions[0],
                y=positions[1],
                z=positions[2],
                mode="lines+markers",
                name="arm",
                line={"width": 7},
            )
        )
        if sphere is not None:
            figure.add_trace(
                go.Surface(
                    x=sphere[0],
                    y=sphere[1],
                    z=sphere[2],
                    opacity=0.25,
                    showscale=False,
                    name="obstacle",
                    colorscale=[[0, "red"], [1, "red"]],
                    hoverinfo="skip",
                )
            )
        figure.update_layout(
            title=title,
            scene={
                "xaxis_title": "x [m]",
                "yaxis_title": "y [m]",
                "zaxis_title": "z [m]",
                "aspectmode": "data",
            },
        )
        return figure
    target = ax if ax is not None else plt.figure(figsize=(7, 6)).add_subplot(projection="3d")
    target.plot(positions[0], positions[1], positions[2], "o-", linewidth=3)
    if sphere is not None:
        target.plot_wireframe(*sphere, color="tab:red", alpha=0.25)
    target.set(title=title, xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    extent_points = positions
    if obstacle is not None:
        extent_points = np.column_stack((positions, center - radius, center + radius))
    lower, upper = extent_points.min(axis=1), extent_points.max(axis=1)
    midpoint = (lower + upper) / 2
    half_span = max(float(np.max(upper - lower)) * 0.55, 0.05)
    # Equal axis ranges make one metre occupy the same distance in every axis.
    target.set_xlim(midpoint[0] - half_span, midpoint[0] + half_span)
    target.set_ylim(midpoint[1] - half_span, midpoint[1] + half_span)
    target.set_zlim(midpoint[2] - half_span, midpoint[2] + half_span)
    target.set_box_aspect((1, 1, 1))
    return target


def plot_ur5e(q, *, ax=None, obstacle=None):
    """Compatibility wrapper returning Matplotlib axes for the packaged UR5e.

    New lessons should construct their own kinematics and call ``plot_robot``.
    The legacy model is imported only when this wrapper is explicitly called.
    """
    from casadi_tutorial.models.ur5e import joint_positions

    return plot_robot(
        joint_positions(ca.DM(q)).full(), ax=ax, obstacle=obstacle, backend="matplotlib"
    )
