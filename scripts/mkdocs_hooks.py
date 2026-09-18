"""Stage canonical notebooks under MkDocs' docs directory before each build."""

import shutil
from pathlib import Path


def on_config(config):
    """Copy canonical notebooks into MkDocs' source tree before each build."""
    repository = Path(__file__).resolve().parents[1]
    source = repository / "notebooks"
    docs_root = (repository / "docs").resolve()
    destination = (docs_root / "generated" / "notebooks").resolve()
    # Resolve both paths before the safety check so symbolic ``..`` components
    # cannot make the cleanup target escape the documentation directory.
    if docs_root not in destination.parents:
        raise RuntimeError(f"Refusing to stage notebooks outside {docs_root}")

    # The staged tree is generated content; replacing it prevents stale chapters
    # from surviving a rename or deletion in the canonical notebook directory.
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return config
