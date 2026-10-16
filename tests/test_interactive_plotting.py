"""Validate frontend-independent data contracts and explicit-run behavior."""

import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from casadi_tutorial.interactive import experiment
from casadi_tutorial.plotting import plot_robot, plot_series, plot_ur5e


def test_static_series_preserves_input_timing(monkeypatch):
    monkeypatch.setenv("CASADI_TUTORIAL_PLOTS", "static")
    axes = plot_series([0, 1, 2], {"input": [1, 2, 2]}, steps=True)
    np.testing.assert_equal(axes.lines[0].get_xdata(), [0, 1, 2])
    assert axes.lines[0].get_drawstyle() == "steps-post"
    plt.close("all")


def test_robot_accepts_local_points_and_legacy_wrapper(monkeypatch):
    monkeypatch.setenv("CASADI_TUTORIAL_PLOTS", "static")
    points = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0.5]])
    axes = plot_robot(points, obstacle=(np.array([0, 0, 0]), 0.2))
    np.testing.assert_allclose(axes.lines[0].get_data_3d(), points)
    spans = [
        np.diff(limits).item() for limits in (axes.get_xlim(), axes.get_ylim(), axes.get_zlim())
    ]
    np.testing.assert_allclose(spans, [spans[0]] * 3)
    assert plot_ur5e(np.zeros(6)).name == "3d"
    plt.close("all")


def test_plot_data_shape_errors():
    with pytest.raises(ValueError):
        plot_series([0, 1], {"wrong length": [1]}, backend="matplotlib")
    with pytest.raises(ValueError):
        plot_robot(np.zeros((2, 4)))


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="optional Plotly absent")
def test_plotly_default_can_be_exported_without_kernel(monkeypatch):
    monkeypatch.delenv("CASADI_TUTORIAL_PLOTS", raising=False)
    figure = plot_series([0, 1], {"state": [0, 1]})
    assert figure.to_plotly_json()["data"][0]["name"] == "state"
    assert "Plotly.newPlot" in figure.to_html(full_html=False, include_plotlyjs=False)
    robot = plot_robot(np.zeros((3, 2)))
    assert robot.to_plotly_json()["data"][0]["type"] == "scatter3d"


def test_widget_opt_out_never_runs(monkeypatch):
    monkeypatch.setenv("CASADI_TUTORIAL_WIDGETS", "0")
    called = []
    assert experiment(lambda **kwargs: called.append(kwargs), {"x": 1}) is None
    assert called == []


@pytest.mark.skipif(
    importlib.util.find_spec("ipywidgets") is None, reason="optional widgets absent"
)
def test_widgets_only_run_on_click_and_reset_without_running(monkeypatch):
    monkeypatch.delenv("CASADI_TUTORIAL_WIDGETS", raising=False)
    called = []
    panel = experiment(lambda value: called.append(value), {"value": 2})
    control = panel.children[1]
    run_button, reset_button = panel.children[2].children
    assert called == []
    control.value = 5
    run_button.click()
    assert called == [5]
    reset_button.click()
    assert control.value == 2
    assert called == [5]
    panel.close()


@pytest.mark.skipif(
    importlib.util.find_spec("ipywidgets") is None, reason="optional widgets absent"
)
def test_widget_exception_restores_buttons(monkeypatch):
    monkeypatch.delenv("CASADI_TUTORIAL_WIDGETS", raising=False)

    def failing(value):
        raise RuntimeError("intentional failure")

    panel = experiment(failing, {"value": 1})
    run_button, reset_button = panel.children[2].children
    run_button.click()
    assert not run_button.disabled and not reset_button.disabled
    assert "failed" in panel.children[3].value
    panel.close()
