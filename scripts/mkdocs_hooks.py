"""Stage canonical notebooks under MkDocs' docs directory before each build."""

import os
import shutil
from pathlib import Path


def on_config(config):
    """Copy canonical notebooks into MkDocs' source tree before each build."""
    # Documentation retains computed results and browser-native Plotly controls;
    # Python callbacks need a live kernel and are omitted from the static pages.
    os.environ["CASADI_TUTORIAL_WIDGETS"] = "0"
    # Newer mkdocs-jupyter releases cache execution; disable it so helper edits
    # are checked too. Version 0.25 (used on Python 3.10) has no cache option.
    notebook_plugin = config["plugins"]["mkdocs-jupyter"]
    if "cache" in notebook_plugin.config:
        notebook_plugin.config["cache"] = False
    repository = Path(__file__).resolve().parents[1]
    docs_root = (repository / "docs").resolve()
    destination = (docs_root / "generated").resolve()
    # Resolve both paths before the safety check so symbolic ``..`` components
    # cannot make the cleanup target escape the documentation directory.
    if docs_root not in destination.parents:
        raise RuntimeError(f"Refusing to stage notebooks outside {docs_root}")

    # The staged tree is generated content; replacing it prevents stale chapters
    # from surviving a rename or deletion in the canonical notebook directory.
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage linked exercises/solutions so relative notebook links also work.
    for folder in ("notebooks", "exercises", "solutions"):
        shutil.copytree(repository / folder, destination / folder)
    return config
