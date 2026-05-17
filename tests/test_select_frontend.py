"""Smoke tests for the static frontend of the `aftermovie select` web GUI.

These tests don't spin up the backend; they only sanity-check that the three
static files exist, are non-empty UTF-8 text, and contain a few anchor strings
that prove the asset is the right one (title tag, fetch call, grid CSS rule).
"""
from __future__ import annotations

from pathlib import Path

STATIC_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "aftermovie"
    / "select"
    / "static"
)


def _read(name: str) -> str:
    path = STATIC_DIR / name
    assert path.is_file(), f"missing static file: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"static file is empty: {path}"
    return text


def test_static_files_exist_and_parse() -> None:
    html = _read("index.html")
    css = _read("style.css")
    js = _read("app.js")

    # index.html must declare a <title> (any title) — proves it parses as the page.
    assert "<title>" in html and "</title>" in html, "index.html missing <title>"

    # app.js must use fetch() to call the backend.
    assert "fetch" in js, "app.js does not reference fetch()"

    # style.css must define the responsive thumbnail grid.
    assert ".grid" in css, "style.css missing .grid rule"
    assert "grid-template-columns" in css, "style.css missing grid-template-columns"
