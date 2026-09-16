"""Stage canonical notebooks under MkDocs' docs directory before each build."""

import shutil
from pathlib import Path


def on_config(config):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "notebooks"
    docs_root = (repository / "docs").resolve()
    destination = (docs_root / "generated" / "notebooks").resolve()
    if docs_root not in destination.parents:
        raise RuntimeError(f"Refusing to stage notebooks outside {docs_root}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return config
