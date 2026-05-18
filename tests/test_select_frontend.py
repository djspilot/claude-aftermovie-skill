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


def test_plan_timeline_panel_wired() -> None:
    """Issue #5: read-only Plan timeline below the source grid.

    Static check — we don't boot the GUI here; we only assert the DOM hooks
    and JS plumbing exist so the front-end can render Plan Entries once a
    parallel agent ships /api/plan.
    """
    html = _read("index.html")
    css = _read("style.css")
    js = _read("app.js")

    # DOM hooks: the panel container, the timeline strip, and a tile template.
    assert 'id="plan-panel"' in html, "index.html missing #plan-panel container"
    assert 'id="plan-timeline"' in html, "index.html missing #plan-timeline strip"
    assert 'id="plan-tile-template"' in html, "index.html missing #plan-tile-template"

    # CSS: timeline must support horizontal scroll when entries overflow.
    assert ".plan-timeline" in css, "style.css missing .plan-timeline rule"
    assert "overflow-x" in css, "style.css missing horizontal overflow for timeline"

    # JS: must fetch /api/plan and expose a Plan-tile builder.
    assert "/api/plan" in js, "app.js does not call /api/plan"
    assert "renderPlanTimeline" in js, "app.js missing renderPlanTimeline function"
    # Must build tiles per entry and read the documented PlanEntry shape.
    assert "buildPlanTile" in js, "app.js missing buildPlanTile function"
    assert "transition_in" in js, "app.js does not handle transition_in badge"
    assert "audio_interest" in js, "app.js does not handle audio_interest indicator"
    # Robustness: must tolerate the array-or-object response shape.
    assert "extractPlanEntries" in js, (
        "app.js missing extractPlanEntries (entries-or-array fallback)"
    )
