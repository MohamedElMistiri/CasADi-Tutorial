"""Checks that teaching modules load directly from the repository checkout."""

from pathlib import Path

import casadi_tutorial


def test_tutorial_modules_are_repository_local() -> None:
    """Guard against accidentally restoring an installed ``src`` distribution."""
    repository_root = Path(__file__).resolve().parents[1]
    expected_module_root = repository_root / "casadi_tutorial"
    loaded_module = Path(casadi_tutorial.__file__).resolve()

    assert loaded_module.is_relative_to(expected_module_root)
    assert not (repository_root / "src" / "casadi_tutorial").exists()
