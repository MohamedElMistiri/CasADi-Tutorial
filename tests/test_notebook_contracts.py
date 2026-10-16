"""Keep capstone formulations visible and explanations adjacent to code."""

import ast
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
TEACHING_NOTEBOOKS = sorted(
    path
    for directory in ("notebooks", "exercises", "solutions")
    for path in (REPOSITORY / directory).glob("*.ipynb")
)
CAPSTONES = [path for path in TEACHING_NOTEBOOKS if "cstr" in path.stem or "ur5" in path.stem]


def notebook_cells(path):
    """Read source JSON without executing a notebook or requiring Jupyter."""
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def dotted_name(node):
    """Recover ordinary dotted call names for the execution-policy check."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


@pytest.mark.parametrize(
    "path", TEACHING_NOTEBOOKS, ids=lambda path: str(path.relative_to(REPOSITORY))
)
def test_each_code_cell_has_preceding_explanatory_markdown(path):
    """A lesson or exercise must explain a computation before presenting it."""
    cells = notebook_cells(path)
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        assert index > 0, f"Code cell {index} has no introduction"
        explanation = cells[index - 1]
        assert explanation["cell_type"] == "markdown", f"Code cell {index} lacks preceding prose"
        assert "".join(explanation["source"]).strip(), f"Code cell {index} has an empty explanation"


@pytest.mark.parametrize("path", CAPSTONES, ids=lambda path: str(path.relative_to(REPOSITORY)))
def test_capstones_formulate_models_locally_without_running_siblings(path):
    """Do not hide capstone models in imports or execute another notebook."""
    for index, cell in enumerate(notebook_cells(path)):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"{path.name}:cell-{index}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                modules += [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                modules = []
            for module in modules:
                assert not module.startswith("casadi_tutorial.models"), (
                    f"Cell {index} imports hidden model {module}"
                )
                assert not module.startswith(("nbclient", "nbconvert.preprocessors")), (
                    f"Cell {index} imports notebook execution machinery {module}"
                )
            if not isinstance(node, ast.Call):
                continue
            name = dotted_name(node.func)
            assert name not in {"exec", "eval"}, f"Cell {index} dynamically executes source"
            assert not name.endswith((".run_path", ".run_module", ".run_cell")), (
                f"Cell {index} executes another source through {name}"
            )
            if name.endswith("run_line_magic") and node.args:
                magic = node.args[0]
                assert not (isinstance(magic, ast.Constant) and magic.value == "run"), (
                    f"Cell {index} executes a sibling through %run"
                )
            if name.endswith(("import_module", "__import__")) and node.args:
                module = node.args[0]
                assert not (
                    isinstance(module, ast.Constant)
                    and isinstance(module.value, str)
                    and module.value.startswith("casadi_tutorial.models")
                ), f"Cell {index} dynamically imports a hidden model"
