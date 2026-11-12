"""Execute three website pilot lessons in fresh kernels with live widgets.

Run with the course environment, including Plotly::

    python scripts/validate_web_lessons.py

Executed copies, HTML exports, and a JSON report go to build/web-validation.
Canonical notebooks are never overwritten. This checks the native runtime;
it does not certify Binder availability or WebAssembly compatibility.
"""

import argparse
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ("01_symbolic_types", "09_nlpsol", "14_cstr_estimation")

CAPTURE = """
import functools as _validation_functools
from casadi_tutorial import interactive as _validation_interactive
_validation_panels = []
_validation_original = _validation_interactive.experiment
def _validation_capture(run, controls, *, title='Experiment'):
    calls = []
    @_validation_functools.wraps(run)
    def record(**kwargs):
        calls.append(kwargs)
        return run(**kwargs)
    panel = _validation_original(record, controls, title=title)
    assert panel is not None, 'Widgets must be enabled for this validation'
    _validation_panels.append((title, panel, calls))
    return panel
_validation_interactive.experiment = _validation_capture
"""

CHECKS = {
    "01_symbolic_types": """
np.testing.assert_allclose(affine_map([2, 3]).full().ravel(), [9, 19])
_validation_metrics = {'affine_changed_input': [9, 19]}
_validation_changes = [{'rows': 4, 'columns': 3, 'family': 'MX'}]
""",
    "09_nlpsol": """
np.testing.assert_allclose(x_optimal, [4 / np.sqrt(5), 2 / np.sqrt(5)], atol=1e-6)
changed = solver(x0=[1.25, .5], p=3, lbx=[0, 0], ubx=[3, 3],
                 lbg=[1, -ca.inf], ubg=[ca.inf, 4])
assert solver.stats()['success']
np.testing.assert_allclose(changed['x'].full().ravel(),
                           [6 / np.sqrt(10), 2 / np.sqrt(10)], atol=1e-6)
assert max_bound_violation(changed['g'], [1, -np.inf], [np.inf, 4]) < 1e-6
_validation_metrics = {'default_optimizer': x_optimal.tolist(),
                       'changed_optimizer': changed['x'].full().ravel().tolist()}
_validation_changes = [{'target_value': 3.0, 'initial_x': 1.25}]
""",
    "14_cstr_estimation": """
assert calibration['success'] and calibration['primal_inf'] < 1e-6
assert abs(FITTED_UA - TRUE_UA) / TRUE_UA < .05
assert heldout_rmse < .1
changed_fit = fit_heat_transfer(calibration_inputs, calibration_measurements,
                                measurement_sigma, initial_guess=180,
                                step_size=CALIBRATION_DT)
assert changed_fit['success'] and changed_fit['primal_inf'] < 1e-6
assert abs(changed_fit['ua'] - FITTED_UA) < 1e-3
_validation_metrics = {'fitted_ua': FITTED_UA,
                       'primal_inf': calibration['primal_inf'],
                       'heldout_rmse_kelvin': heldout_rmse,
                       'changed_initial_guess_ua': changed_fit['ua']}
_validation_changes = [{'step_amplitude': 10.0}, {'noise': .1, 'initial_guess': 180.0}]
""",
}

WIDGET_CHECKS = """
import json as _validation_json
assert len(_validation_panels) == len(_validation_changes)
_validation_widget_results = []
for (title, panel, calls), changes in zip(_validation_panels, _validation_changes):
    controls = panel.children[1:-3]
    original_values = [control.value for control in controls]
    run_button, reset_button = panel.children[-3].children
    status = panel.children[-2]
    assert calls == [], 'Constructing controls must not trigger a solver'
    for control, (name, value) in zip(controls, changes.items()):
        control.value = value
    assert calls == [], 'Editing controls must not trigger a solver'
    run_button.click()
    assert calls == [changes], (title, calls, changes)
    assert status.value == 'Experiment completed.', (title, status.value)
    assert not run_button.disabled and not reset_button.disabled
    reset_button.click()
    assert [control.value for control in controls] == original_values
    assert len(calls) == 1, 'Reset must not trigger a solver'
    _validation_widget_results.append({'title': title, 'changed_inputs': changes,
                                       'run_and_reset': 'passed'})
print('WEB_VALIDATION_JSON=' + _validation_json.dumps(
    {'metrics': _validation_metrics, 'widgets': _validation_widget_results}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/web-validation")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep Jupyter runtime files and the pilot kernelspec inside the output tree.
    kernelspec = output / "jupyter/kernels/web-validation"
    kernelspec.mkdir(parents=True, exist_ok=True)
    (kernelspec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Website validation",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        [str(output / "jupyter"), str(Path(sys.base_prefix) / "share/jupyter")]
    )
    os.environ["JUPYTER_RUNTIME_DIR"] = str(output / "runtime")
    os.environ["IPYTHONDIR"] = str(output / "ipython")
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ["CASADI_TUTORIAL_WIDGETS"] = "1"
    os.environ.pop("CASADI_TUTORIAL_PLOTS", None)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    report = {
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("casadi", "plotly", "ipywidgets", "nbclient")},
        "scope": "Native kernels and HTML export; browser and hosted launch checked separately",
        "lessons": [],
    }
    for lesson in LESSONS:
        print(f"Executing {lesson} with live widgets...", flush=True)
        notebook = nbformat.read(ROOT / "notebooks" / f"{lesson}.ipynb", as_version=4)
        original_code_cells = sum(c.cell_type == "code" for c in notebook.cells)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(CAPTURE))
        notebook.cells.append(nbformat.v4.new_code_cell(CHECKS[lesson] + WIDGET_CHECKS))
        client = NotebookClient(
            notebook,
            kernel_name="web-validation",
            timeout=300,
            resources={"metadata": {"path": str(ROOT / "notebooks")}},
        )
        start = perf_counter()
        entry = {"lesson": lesson, "original_code_cells": original_code_cells}
        try:
            client.execute()
            mime_counts = {}
            details = None
            for cell in notebook.cells:
                for item in cell.get("outputs", []):
                    for mime in item.get("data", {}):
                        mime_counts[mime] = mime_counts.get(mime, 0) + 1
                    if item.output_type == "stream":
                        for line in item.text.splitlines():
                            if line.startswith("WEB_VALIDATION_JSON="):
                                details = json.loads(line.split("=", 1)[1])
            assert details is not None, "Missing validation results"
            assert mime_counts.get("application/vnd.jupyter.widget-view+json", 0)
            if lesson != "01_symbolic_types":
                assert mime_counts.get("application/vnd.plotly.v1+json", 0)
            html, _ = HTMLExporter().from_notebook_node(notebook)
            if lesson != "01_symbolic_types":
                assert "Plotly.newPlot" in html
            (output / f"{lesson}.html").write_text(html, encoding="utf-8")
            entry.update(status="passed", mime_counts=mime_counts, **details)
        except Exception as error:
            entry.update(status="failed", error=str(error))
        finally:
            nbformat.write(notebook, output / f"{lesson}.ipynb")
        entry["seconds"] = round(perf_counter() - start, 2)
        report["lessons"].append(entry)
        (output / "native-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{lesson}: {entry['status']} ({entry['seconds']} s)", flush=True)
    return int(any(entry["status"] != "passed" for entry in report["lessons"]))


if __name__ == "__main__":
    raise SystemExit(main())
