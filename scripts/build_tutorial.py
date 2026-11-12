"""Build executed tutorial pages for the personal website or a local preview.

    python scripts/build_tutorial.py --output ../MohamedElMistiri.github.io/_site/tutorials/casadi

Run in the environment.yml environment. A build-local kernelspec ensures that
notebook execution uses this interpreter, not a different system Python.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mkdocs.commands.build import build
from mkdocs.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    parser.add_argument(
        "--site-url", default="https://MohamedElMistiri.github.io/tutorials/casadi/"
    )
    parser.add_argument("--personal-site", default="https://MohamedElMistiri.github.io/")
    args = parser.parse_args()
    data = ROOT / "build/jupyter"
    spec = data / "kernels/casadi-tutorial-build"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "CasADi documentation",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        [str(data), str(Path(sys.base_prefix) / "share/jupyter"), os.getenv("JUPYTER_PATH", "")]
    )
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "build/jupyter-runtime")
    os.environ["IPYTHONDIR"] = str(ROOT / "build/ipython")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    config = load_config(
        str(ROOT / "mkdocs.yml"),
        site_dir=str(args.output.resolve()),
        site_url=args.site_url,
        strict=True,
    )
    config["plugins"]["mkdocs-jupyter"].config["kernel_name"] = "casadi-tutorial-build"
    config["extra"]["personal_site"] = args.personal_site
    config["nav"][0] = {"Back to Mohamed's website": args.personal_site}
    build(config)


if __name__ == "__main__":
    main()
