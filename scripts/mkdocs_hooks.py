"""Stage canonical notebooks under MkDocs' docs directory before each build."""

import hashlib
import html
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup


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


def tutorial_revision():
    """Use the same revision for the rendered lessons and Binder's checkout."""
    if revision := os.getenv("CASADI_TUTORIAL_REF"):
        return revision
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Set CASADI_TUTORIAL_REF when building outside a Git checkout")
    return result.stdout.strip()


def on_page_content(content, page, config, files):
    """Attach exact notebook sources without loading a runtime during reading."""
    source = page.file.src_uri
    if not source.startswith("generated/") or not source.endswith(".ipynb"):
        return content
    notebook_path = source.removeprefix("generated/")
    repository = Path(__file__).resolve().parents[1]
    notebook = json.loads((repository / notebook_path).read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    soup = BeautifulSoup(content, "html.parser")
    # mkdocs-jupyter uses InputArea for rendered Markdown too; only code areas
    # contain the editor wrapper. Nested CodeCell divs must not be counted twice.
    inputs = [
        area for area in soup.select(".jp-InputArea") if area.select_one(".jp-InputArea-editor")
    ]
    if len(inputs) != len(code):
        raise RuntimeError(f"Notebook HTML/source mismatch in {notebook_path}")
    for index, input_area in enumerate(inputs):
        cell = input_area.find_parent(class_="jp-CodeCell")
        cell["data-casadi-cell"] = str(index)
        input_area["data-casadi-input"] = ""
    repo = "MohamedElMistiri/CasADi-Tutorial"
    revision = tutorial_revision()
    launch_url = (
        f"https://mybinder.org/v2/gh/{repo}/{quote(revision, safe='')}"
        f"?urlpath={quote('lab/tree/' + notebook_path, safe='')}"
    )
    payload = json.dumps(
        {"repo": repo, "ref": revision, "path": notebook_path, "code": code},
        ensure_ascii=True,
    ).replace("<", "\\u003c")
    is_exercise = notebook_path.startswith("exercises/")
    notice = (
        "Exercise cells contain unfinished tasks. Run them individually as you work."
        if is_exercise
        else "Your session will run this lesson's cells in order before enabling its experiments."
    )
    toolbar = f"""
<section class="casadi-session" aria-label="Interactive Python">
  <div class="casadi-session-heading"><strong>Try this lesson</strong>
    <span class="casadi-session-badge">Python · CasADi</span></div>
  <p>Read the saved results below, or start a free Python session to edit code
     and run experiments here. Startup may take a few minutes.</p>
  <div class="casadi-session-actions">
    <button type="button" class="casadi-button" data-casadi-start>
      Enable interactive session</button>
    <button type="button" class="casadi-button secondary" data-casadi-run hidden>
      Run all cells</button>
    <button type="button" class="casadi-button secondary" data-casadi-stop hidden>
      Disconnect</button>
    <a href="{html.escape(launch_url, quote=True)}" target="_blank" rel="noopener">
      Open in JupyterLab ↗</a>
  </div>
  <p class="casadi-session-status" role="status" aria-live="polite" data-casadi-status>
    No session started. Saved results are ready to explore.</p>
  <details><summary>About interactive sessions</summary>
    <p>{notice} Sessions use public Binder resources and end after inactivity.
    Download your notebook to keep your work.</p></details>
</section>
<script type="application/json" id="casadi-notebook">{payload}</script>
"""
    return toolbar + str(soup)


def on_post_build(config):
    """Share repeated plot libraries in older saved outputs and use one MathJax."""
    site = Path(config["site_dir"])
    for path in site.rglob("*.html"):
        content = path.read_text(encoding="utf-8")
        if "plotly.js v" not in content and "init_mathjax" not in content:
            continue
        soup = BeautifulSoup(content, "html.parser")
        seen = set()
        for script in soup.find_all("script"):
            source = script.get("src", "").lower()
            text = script.string or ""
            if "mathjax/2" in source or ("init_mathjax" in text and "MathJax.Hub" in text):
                script.decompose()
            elif "* plotly.js v" in text and len(text) > 100_000:
                digest = hashlib.sha256(text.encode()).hexdigest()[:12]
                asset = site / "assets/vendor" / f"plotly-{digest}.min.js"
                if not asset.exists():
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    asset.write_text(text, encoding="utf-8")
                if digest in seen:
                    script.decompose()
                else:
                    seen.add(digest)
                    script.clear()
                    script["src"] = os.path.relpath(asset, path.parent).replace(os.sep, "/")
        temporary = path.with_suffix(".html.tmp")
        temporary.write_text(str(soup), encoding="utf-8")
        temporary.replace(path)
