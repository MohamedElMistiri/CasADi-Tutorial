"""Explicit-run notebook experiments; numerical defaults live outside widgets."""

import copy
import html
import numbers
import os
import traceback


def experiment(run, controls, *, title="Experiment"):
    """Display Run/Reset controls without executing ``run`` automatically.

    ``controls`` maps keyword argument names to ipywidgets value widgets or
    simple bool/int/float/string values. Reset restores construction-time values
    and does not solve again. A non-None callback result is displayed; callbacks
    can instead display their own figures and return None. Exceptions stay in
    the panel with their traceback and never masquerade as successful results.

    Returns the VBox panel, or None when ``CASADI_TUTORIAL_WIDGETS=0`` or widgets
    are unavailable. This opt-out never skips numerical calculations outside
    this function. No callback runs during notebook import/execution.
    """
    if os.getenv("CASADI_TUTORIAL_WIDGETS") == "0":
        return None
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("Interactive controls unavailable; the default results above remain usable.")
        return None

    inputs = {}
    for name, value in controls.items():
        if isinstance(value, widgets.ValueWidget):
            control = value
        elif isinstance(value, bool):
            control = widgets.Checkbox(value=value)
        elif isinstance(value, numbers.Integral):
            control = widgets.IntText(value=int(value))
        elif isinstance(value, numbers.Real):
            control = widgets.FloatText(value=float(value))
        elif isinstance(value, str):
            control = widgets.Text(value=value)
        else:
            raise TypeError(f"Control {name!r} must be a value widget or bool/int/float/string.")
        if hasattr(control, "description") and not control.description:
            control.description = name.replace("_", " ")
        if hasattr(control, "continuous_update"):
            control.continuous_update = False
        inputs[name] = control

    initial = {name: copy.deepcopy(control.value) for name, control in inputs.items()}
    run_button = widgets.Button(description="Run experiment", button_style="primary")
    reset_button = widgets.Button(description="Reset controls")
    status = widgets.HTML(value="Defaults are shown above. Change controls, then run.")
    output = widgets.Output()

    def execute(_):
        run_button.disabled = reset_button.disabled = True
        status.value = "Running…"
        with output:
            output.clear_output(wait=True)
            try:
                result = run(**{name: control.value for name, control in inputs.items()})
                if result is not None:
                    display(result)
            except Exception:
                traceback.print_exc()
                status.value = "Experiment failed; inspect the traceback below."
            else:
                status.value = "Experiment completed."
            finally:
                run_button.disabled = reset_button.disabled = False

    def reset(_):
        for name, control in inputs.items():
            control.value = copy.deepcopy(initial[name])
        status.value = "Controls reset. Run to compute the restored experiment."

    run_button.on_click(execute)
    reset_button.on_click(reset)
    panel = widgets.VBox(
        [
            widgets.HTML(value=f"<b>{html.escape(title)}</b>"),
            *inputs.values(),
            widgets.HBox([run_button, reset_button]),
            status,
            output,
        ]
    )
    display(panel)
    return panel
