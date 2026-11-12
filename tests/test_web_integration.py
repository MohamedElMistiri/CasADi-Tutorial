"""Check source/HTML alignment and safe on-demand notebook metadata."""

import json
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from scripts import mkdocs_hooks


def render_lesson(tmp_path, monkeypatch, source, content):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks/pilot.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": [source]}]}), encoding="utf-8"
    )
    monkeypatch.setattr(mkdocs_hooks, "__file__", str(tmp_path / "scripts/mkdocs_hooks.py"))
    monkeypatch.setenv("CASADI_TUTORIAL_REF", "a" * 40)
    page = SimpleNamespace(file=SimpleNamespace(src_uri="generated/notebooks/pilot.ipynb"))
    return mkdocs_hooks.on_page_content(content, page, None, None)


def test_metadata_preserves_source_and_marks_only_code(tmp_path, monkeypatch):
    content = """
    <div class="jp-InputArea"><div class="jp-RenderedMarkdown">Explanation</div></div>
    <div class="jp-CodeCell"><div class="jp-CodeCell">
      <div class="jp-InputArea"><div class="jp-InputArea-editor"><pre>print(1)</pre></div></div>
      <div class="jp-Cell-outputWrapper">Saved result</div>
    </div></div>
    """
    source = 'print("</script><script>unsafe()</script>")\n'
    rendered = render_lesson(tmp_path, monkeypatch, source, content)
    soup = BeautifulSoup(rendered, "html.parser")
    assert len(soup.select("[data-casadi-cell]")) == 1
    assert len(soup.select("[data-casadi-input]")) == 1
    assert len(soup.find_all("script")) == 1
    payload = json.loads(soup.select_one("#casadi-notebook").string)
    assert payload["code"] == [source]
    assert payload["ref"] == "a" * 40
    assert payload["path"] == "notebooks/pilot.ipynb"
    assert "Saved result" in rendered
    # No executable library or runtime credentials are embedded in the page.
    assert not soup.select("script[src]")
    assert "token" not in payload


def test_missing_code_area_fails_the_build(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="HTML/source mismatch"):
        render_lesson(tmp_path, monkeypatch, "print(1)", "<p>No code area</p>")


def test_regular_pages_are_unchanged():
    page = SimpleNamespace(file=SimpleNamespace(src_uri="index.md"))
    assert mkdocs_hooks.on_page_content("<p>Overview</p>", page, None, None) == "<p>Overview</p>"


def test_old_exports_share_one_library_and_keep_plot_data(tmp_path):
    page = tmp_path / "generated/pilot/index.html"
    page.parent.mkdir(parents=True)
    library = "/* plotly.js v3.7.0 */" + " " * 100_001
    page.write_text(
        '<script src="https://cdn.example/mathjax/2/MathJax.js"></script>'
        f"<script>{library}</script><script>{library}</script>"
        '<div id="chart"></div><script>Plotly.newPlot("chart", []);</script>',
        encoding="utf-8",
    )
    mkdocs_hooks.on_post_build({"site_dir": str(tmp_path)})
    soup = BeautifulSoup(page.read_text(encoding="utf-8"), "html.parser")
    assert len(soup.select("script[src]")) == 1
    assert "MathJax.js" not in str(soup)
    assert "Plotly.newPlot" in str(soup)
    assert len(list((tmp_path / "assets/vendor").glob("*.js"))) == 1
