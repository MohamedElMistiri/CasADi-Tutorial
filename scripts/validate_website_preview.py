"""Check the real personal-site pages and live course in a running local preview.

    python scripts/validate_website_preview.py --url http://127.0.0.1:8090

Start serve_website_preview.py first. Playwright and a Chromium browser are
required. Reports and screenshots are saved under build/website-validation.
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from playwright.sync_api import sync_playwright
from validate_web_lessons import LESSONS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    output = ROOT / "build/website-validation"
    output.mkdir(parents=True, exist_ok=True)
    report = {"url": base, "lessons": []}
    with sync_playwright() as p:
        edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
        executable = os.getenv("WEB_VALIDATION_BROWSER") or (str(edge) if edge.is_file() else None)
        browser = p.chromium.launch(executable_path=executable, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(base + "/", wait_until="domcontentloaded")
        page.get_by_role("navigation", name="Main navigation").get_by_role(
            "link", name="Teaching", exact=True
        ).click()
        page.wait_for_url("**/teaching/")
        page.screenshot(path=str(output / "teaching.png"), full_page=True)
        page.get_by_role("link", name="Start the CasADi tutorial").click()
        page.wait_for_url("**/tutorials/casadi/")
        page.screenshot(path=str(output / "course.png"), full_page=True)
        report["website_navigation"] = "passed"
        for lesson in LESSONS:
            entry = {"lesson": lesson}
            errors, requests = [], []
            tab = browser.new_page(viewport={"width": 1440, "height": 1000})
            tab.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
            tab.on("request", lambda request, requests=requests: requests.append(request.url))
            started = perf_counter()
            print("Integrated lesson:", lesson, flush=True)
            try:
                tab.goto(
                    base + f"/tutorials/casadi/generated/notebooks/{lesson}/",
                    wait_until="networkidle",
                )
                expected_plots = {"01_symbolic_types": 0, "09_nlpsol": 1, "14_cstr_estimation": 12}[
                    lesson
                ]
                if expected_plots:
                    tab.wait_for_function(
                        "count => document.querySelectorAll('.js-plotly-plot').length >= count",
                        arg=expected_plots,
                    )
                assert tab.locator(".js-plotly-plot:visible").count() == expected_plots
                assert tab.evaluate("typeof window.thebe") == "undefined"
                assert not any(
                    "/api/sessions" in url or "mybinder.org/build/" in url for url in requests
                )
                entry["no_runtime_before_click"] = "passed"
                tab.get_by_role("button", name="Enable interactive session", exact=True).click()
                tab.locator("[data-casadi-status]").filter(has_text="Session ready.").wait_for(
                    timeout=180000
                )
                assert tab.locator("[data-casadi-saved]:visible").count() == 0
                assert tab.locator(".js-plotly-plot:visible").count() == expected_plots
                buttons = tab.get_by_role("button", name="Run experiment", exact=True)
                expected_panels = 2 if lesson == "14_cstr_estimation" else 1
                assert buttons.count() == expected_panels
                changes = {
                    "01_symbolic_types": [[4, 3]],
                    "09_nlpsol": [[3, 1.25]],
                    "14_cstr_estimation": [[10], [0.1, 180]],
                }[lesson]
                for index in range(expected_panels):
                    panel = buttons.nth(index).locator("xpath=../..")
                    readouts = panel.locator(".widget-readout")
                    defaults = readouts.all_text_contents()
                    for control, value in zip(readouts.all(), changes[index], strict=True):
                        control.fill(str(value))
                        control.press("Enter")
                    if lesson == "01_symbolic_types":
                        panel.locator("select").select_option(label="MX")
                    panel.get_by_text(
                        "Defaults are shown above. Change controls, then run.", exact=True
                    ).wait_for()
                    buttons.nth(index).click()
                    panel.get_by_text("Experiment completed.", exact=True).wait_for(timeout=60000)
                    result = panel.locator(".widget-output").inner_text()
                    if lesson == "01_symbolic_types":
                        assert "MX: A (4, 3)" in result
                    panel.get_by_role("button", name="Reset controls", exact=True).click()
                    panel.get_by_text(
                        "Controls reset. Run to compute the restored experiment.", exact=True
                    ).wait_for()
                    assert readouts.all_text_contents() == defaults
                    assert panel.locator(".widget-output").inner_text() == result
                entry["changed_inputs_run_reset"] = "passed"
                tab.locator(".widget-vbox").first.screenshot(
                    path=str(output / f"{lesson}-experiment.png")
                )
                tab.get_by_role("button", name="Disconnect", exact=True).click()
                tab.locator("[data-casadi-status]").filter(
                    has_text="Session disconnected."
                ).wait_for()
                assert tab.locator(".casadi-live-cell").count() == 0
                assert tab.locator(".js-plotly-plot:visible").count() == expected_plots
                entry["disconnect_restores_saved_results"] = "passed"
                tab.set_viewport_size({"width": 390, "height": 844})
                assert tab.evaluate("document.documentElement.scrollWidth <= innerWidth + 5")
                tab.screenshot(path=str(output / f"{lesson}-mobile.png"))
                entry["mobile_layout"] = "passed"
                assert not errors, errors
                entry["status"] = "passed"
            except Exception as error:
                entry.update(status="failed", error=str(error))
                tab.screenshot(path=str(output / f"{lesson}-failure.png"), full_page=True)
            finally:
                entry.update(seconds=round(perf_counter() - started, 2), page_errors=errors)
                report["lessons"].append(entry)
                (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(lesson, entry["status"], entry.get("error", ""), flush=True)
                tab.close()
        # Losing the optional frontend must leave the static lesson usable.
        page.goto(base + "/tutorials/casadi/generated/notebooks/01_symbolic_types/")
        page.route("https://unpkg.com/thebe@0.9.2/lib/index.js", lambda route: route.abort())
        page.get_by_role("button", name="Enable interactive session", exact=True).click()
        page.locator("[data-casadi-status]").filter(has_text="Could not download").wait_for()
        assert page.locator("[data-casadi-input]:visible").count() == 5
        assert page.get_by_role("button", name="Retry interactive session").is_enabled()
        report["runtime_failure_preserves_reading"] = "passed"
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        browser.close()
    return int(any(entry["status"] != "passed" for entry in report["lessons"]))


if __name__ == "__main__":
    raise SystemExit(main())
