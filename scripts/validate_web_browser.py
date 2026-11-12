"""Validate live Thebe cells, widgets, and plots in a headless browser.

Run validate_web_lessons.py first in the course environment, then run this
script with Playwright installed. The pinned Thebe bundle is downloaded into
.cache on first use. Network access is required for widget frontend modules.
Set WEB_VALIDATION_BROWSER to a Chromium executable, or install Playwright's
Chromium with ``python -m playwright install chromium``. Edge is used when
available on Windows. Reports and screenshots go to build/web-validation.

This is a local Jupyter runtime test, not a public Binder or Pyodide test.
"""

import functools
import html
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nbformat
from playwright.sync_api import sync_playwright
from plotly.offline import get_plotlyjs
from validate_web_lessons import LESSONS, ROOT

OUT = ROOT / "build/web-validation"
WEB = OUT / "web"
WEB.mkdir(parents=True, exist_ok=True)
bundle = ROOT / ".cache/web-validation/thebe.js"
if not bundle.exists():
    bundle.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://unpkg.com/thebe@0.9.2/lib/index.js", bundle)
shutil.copyfile(bundle, WEB / "thebe.js")
(WEB / "plotly.js").write_text(get_plotlyjs(), encoding="utf-8")
TOKEN = secrets.token_hex(24)
SERVER_PORT, STATIC_PORT = 8875, 8876
BASE = f"http://127.0.0.1:{SERVER_PORT}/"
ORIGIN = f"http://127.0.0.1:{STATIC_PORT}"
env = os.environ.copy()
env.update(
    JUPYTER_PATH=os.pathsep.join(
        [str(OUT / "jupyter"), str(Path(sys.base_prefix) / "share/jupyter")]
    ),
    JUPYTER_CONFIG_DIR=str(OUT / "config"),
    JUPYTER_DATA_DIR=str(OUT / "jupyter"),
    JUPYTER_RUNTIME_DIR=str(OUT / ("runtime-browser-" + secrets.token_hex(4))),
    IPYTHONDIR=str(OUT / "ipython"),
    PYTHONPATH=str(ROOT),
    CASADI_TUTORIAL_WIDGETS="1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MPLBACKEND="module://matplotlib_inline.backend_inline",
)
env.pop("CASADI_TUTORIAL_PLOTS", None)

for lesson in LESSONS:
    nb = nbformat.read(ROOT / "notebooks" / f"{lesson}.ipynb", as_version=4)
    cells = "\n".join(
        '<section><pre data-executable data-language="python">'
        f"{html.escape(c.source)}</pre></section>"
        for c in nb.cells
        if c.cell_type == "code"
    )
    (WEB / f"{lesson}.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
    <title>{lesson} — live Python pilot</title>
    <script src="plotly.js"></script><script src="thebe.js"></script>
    <link rel="icon" href="data:,">
    <style>body{{font:16px sans-serif;margin:2rem;max-width:1150px}}
    section{{margin:1rem 0;border:1px solid #ddd;padding:1rem}}
    pre{{white-space:pre-wrap}} .thebe-output{{min-height:1rem}}
    .widget-vbox{{margin:1rem 0}}</style></head><body><h1>{lesson}</h1>
    <p>Local integration validation with original lesson cells.</p>{cells}</body></html>""",
        encoding="utf-8",
    )


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


static = ThreadingHTTPServer(
    ("127.0.0.1", STATIC_PORT), functools.partial(QuietHandler, directory=str(WEB))
)
threading.Thread(target=static.serve_forever, daemon=True).start()
log = (OUT / "jupyter-server.log").open("w", encoding="utf-8")
server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "jupyter_server",
        "--no-browser",
        f"--ServerApp.root_dir={ROOT}",
        f"--ServerApp.port={SERVER_PORT}",
        "--ServerApp.port_retries=0",
        f"--IdentityProvider.token={TOKEN}",
        f"--ServerApp.allow_origin={ORIGIN}",
        "--ServerApp.ip=127.0.0.1",
    ],
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
)
report = {
    "frontend": "Thebe 0.9.2, original notebook cells",
    "backend": "Local native Jupyter server; not a public Binder launch",
    "lessons": [],
}
try:
    for _ in range(120):
        try:
            req = urllib.request.Request(BASE + "api", headers={"Authorization": "token " + TOKEN})
            urllib.request.urlopen(req, timeout=2).close()
            break
        except Exception as error:
            if server.poll() is not None:
                raise RuntimeError("Jupyter server exited; see jupyter-server.log") from error
            time.sleep(0.5)
    else:
        raise RuntimeError("Jupyter server did not become ready")
    with sync_playwright() as p:
        edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
        executable = os.getenv("WEB_VALIDATION_BROWSER") or (str(edge) if edge.is_file() else None)
        browser = p.chromium.launch(executable_path=executable, headless=True)
        report["browser"] = browser.version
        for lesson in LESSONS:
            print("Browser:", lesson, flush=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors, console_errors = [], []
            page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
            page.on(
                "requestfailed",
                lambda request, log=console_errors: log.append(
                    request.url + ": " + str(request.failure)
                ),
            )
            page.on(
                "console",
                lambda message, log=console_errors: (
                    log.append(message.text) if message.type == "error" else None
                ),
            )
            entry = {"lesson": lesson}
            start = time.perf_counter()
            try:
                page.goto(f"{ORIGIN}/{lesson}.html", wait_until="load")
                config = {
                    "useBinder": False,
                    "requestKernel": True,
                    "mountActivateWidget": False,
                    "mountStatusWidget": False,
                    "serverSettings": {"baseUrl": BASE, "token": TOKEN},
                    "kernelOptions": {"kernelName": "web-validation", "path": "notebooks"},
                    "savedSessionOptions": {"enabled": False},
                }
                print("Connecting Thebe...", flush=True)
                page.evaluate(
                    """async config => {
                  window.pilot = await Promise.race([thebe.bootstrap(config),
                    new Promise((_, reject) => setTimeout(() =>
                      reject(Error('Kernel startup timeout')), 60000))]);
                  return Boolean(window.pilot.session);
                }""",
                    config,
                )
                print("Thebe connected; executing cells...", flush=True)
                results = []
                count = page.evaluate("window.thebe.notebook.cells.length")
                for index in range(count):
                    print("Cell", index, flush=True)
                    result = page.evaluate(
                        """async index => {
                      const c = window.thebe.notebook.cells[index];
                      await Promise.race([c.execute(c.source), new Promise((_,reject) =>
                        setTimeout(() => reject(Error('Cell execution timeout')), 45000))]);
                      return {id:c.id, outputs:c.outputs};
                    }""",
                        index,
                    )
                    results.append(result)
                (OUT / f"{lesson}-browser-outputs.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )
                cell_errors = [
                    o
                    for c in results
                    for o in (c.get("outputs") or [])
                    if o.get("output_type") == "error"
                ]
                assert not cell_errors, cell_errors
                expected = 2 if lesson == "14_cstr_estimation" else 1
                buttons = page.get_by_role("button", name="Run experiment", exact=True)
                buttons.first.wait_for(state="visible", timeout=30000)
                assert buttons.count() == expected, (buttons.count(), expected)
                entry["plot_count"] = page.locator(".js-plotly-plot").count()
                assert (
                    entry["plot_count"]
                    == {"01_symbolic_types": 0, "09_nlpsol": 1, "14_cstr_estimation": 12}[lesson]
                )
                changes = {
                    "01_symbolic_types": [[4, 3]],
                    "09_nlpsol": [[3, 1.25]],
                    "14_cstr_estimation": [[10], [0.1, 180]],
                }[lesson]
                entry["experiments"] = []
                for index in range(expected):
                    button = buttons.nth(index)
                    panel = button.locator("xpath=../..")
                    readouts = panel.locator(".widget-readout")
                    defaults = readouts.all_text_contents()
                    button.click()
                    panel.get_by_text("Experiment completed.", exact=True).wait_for(timeout=60000)
                    assert panel.get_by_role(
                        "button", name="Run experiment", exact=True
                    ).is_enabled()
                    panel.get_by_role("button", name="Reset controls", exact=True).click()
                    panel.get_by_text(
                        "Controls reset. Run to compute the restored experiment.", exact=True
                    ).wait_for()
                    baseline = panel.locator(".widget-output").inner_text()
                    for control_index, value in enumerate(changes[index]):
                        readouts.nth(control_index).fill(str(value))
                        readouts.nth(control_index).press("Enter")
                    if lesson == "01_symbolic_types":
                        panel.locator("select").select_option(label="MX")
                    # Editing controls must leave both status and output unchanged.
                    assert panel.locator(".widget-output").inner_text() == baseline
                    panel.get_by_text(
                        "Controls reset. Run to compute the restored experiment.", exact=True
                    ).wait_for()
                    button.click()
                    panel.get_by_text("Experiment completed.", exact=True).wait_for(timeout=60000)
                    changed_output = panel.locator(".widget-output").inner_text()
                    if lesson == "01_symbolic_types":
                        assert "MX: A (4, 3)" in changed_output, changed_output
                    elif lesson == "14_cstr_estimation" and index == 1:
                        assert changed_output != baseline, "Changing noise must change the fit"
                    panel.get_by_role("button", name="Reset controls", exact=True).click()
                    panel.get_by_text(
                        "Controls reset. Run to compute the restored experiment.", exact=True
                    ).wait_for()
                    assert readouts.all_text_contents() == defaults
                    assert panel.locator(".widget-output").inner_text() == changed_output
                    entry["experiments"].append(
                        {"changed_readouts": changes[index], "run_and_reset": "passed"}
                    )
                if lesson != "01_symbolic_types":
                    plot = page.locator(".js-plotly-plot").first
                    old_range = plot.evaluate("e => Array.from(e._fullLayout.xaxis.range)")
                    plot.locator('[data-title="Zoom in"]').click(force=True)
                    page.wait_for_function(
                        """({oldRange}) => {
                      const r=document.querySelector('.js-plotly-plot')._fullLayout.xaxis.range;
                      return r[1]-r[0] < oldRange[1]-oldRange[0];
                    }""",
                        arg={"oldRange": old_range},
                    )
                    entry["plot_zoom"] = "passed"
                assert not errors, errors
                entry["status"] = "passed"
                entry["widget_panels"] = expected
                page.screenshot(path=str(OUT / f"{lesson}-browser.png"), full_page=True)
            except Exception as error:
                entry.update(status="failed", error=str(error))
                page.screenshot(path=str(OUT / f"{lesson}-browser-failure.png"), full_page=True)
            finally:
                entry.update(
                    seconds=round(time.perf_counter() - start, 2),
                    page_errors=errors,
                    console_errors=console_errors,
                )
                report["lessons"].append(entry)
                (OUT / "browser-report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(lesson, entry["status"], entry.get("error", ""), flush=True)
                # Dispose this pilot's session before moving on to another lesson.
                page.evaluate(
                    "async () => { if (window.pilot?.session) "
                    "await window.pilot.session.shutdown(); }"
                )
                page.close()
        browser.close()
finally:
    static.shutdown()
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    log.close()
raise SystemExit(int(any(e["status"] != "passed" for e in report["lessons"])))
