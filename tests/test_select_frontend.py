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


def test_render_preview_button_wired() -> None:
    """Issue #3: GUI exposes a separate `Render Preview` button.

    Static check — confirms the DOM has the preview button, app.js sends
    `preview: true` in the POST payload, and the preview status badge has
    CSS rules. We don't boot the GUI here; only that the wiring is present.
    """
    html = _read("index.html")
    css = _read("style.css")
    js = _read("app.js")

    # DOM: a dedicated preview button distinct from the final Render button.
    assert 'id="render-preview-btn"' in html, (
        "index.html missing #render-preview-btn button"
    )
    assert "Render Preview" in html, "index.html missing 'Render Preview' label"

    # DOM: a preview status row with badge + message + (optional) cache hint.
    assert 'id="preview-status"' in html, "index.html missing #preview-status row"
    assert 'id="preview-badge"' in html, "index.html missing #preview-badge"
    assert 'id="cache-indicator"' in html, (
        "index.html missing #cache-indicator (Reuse analysis hint)"
    )

    # JS: payload assembly must include `preview: true` for the preview path.
    assert "preview: true" in js, (
        "app.js does not send preview: true in /api/render payload"
    )
    # JS: a dedicated renderPreview entry point so the wiring is greppable.
    assert "renderPreview" in js, "app.js missing renderPreview function"
    # JS: cache_hit handling must be defensive (only acts when present).
    assert "cache_hit" in js, "app.js does not read cache_hit from /api/status"

    # CSS: the new button + preview badge must have styles using existing tokens.
    assert ".btn.secondary" in css, "style.css missing .btn.secondary rule"
    assert ".preview-badge" in css, "style.css missing .preview-badge rule"
    assert ".cache-indicator" in css, "style.css missing .cache-indicator rule"


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


def test_song_picker_section_wired() -> None:
    """Phase D: GUI exposes a song picker section above the source grid.

    Static check — confirms the DOM hooks (song-section, file picker, path
    input, candidate / recent chip containers), the JS entry points
    (loadSongInfo, setSong, loadCandidateSongs, loadRecentSongs), and the
    documented endpoint names exist. The backend is wired by the same
    Phase D batch; this only verifies the front-end stays in lockstep.
    """
    html = _read("index.html")
    css = _read("style.css")
    js = _read("app.js")

    # DOM: the picker section, the current-path display, and the two pick
    # mechanisms (file picker + text path + Use button).
    assert 'id="song-section"' in html, "index.html missing #song-section"
    assert 'id="song-current-path"' in html, (
        "index.html missing #song-current-path"
    )
    assert 'id="song-file"' in html, "index.html missing #song-file picker"
    assert 'id="song-path"' in html, "index.html missing #song-path input"
    assert 'id="song-use-path-btn"' in html, (
        "index.html missing #song-use-path-btn"
    )
    # DOM: candidate + recent chip containers and the sparkline element.
    assert 'id="song-candidates"' in html, "index.html missing #song-candidates"
    assert 'id="song-recents"' in html, "index.html missing #song-recents"
    assert 'id="song-sparkline"' in html, "index.html missing #song-sparkline"

    # CSS: the picker has dedicated styles using existing tokens.
    assert ".song-section" in css, "style.css missing .song-section rule"
    assert ".song-chip" in css, "style.css missing .song-chip rule"

    # JS: the documented entry points must be greppable.
    assert "loadSongInfo" in js, "app.js missing loadSongInfo function"
    assert "setSong" in js, "app.js missing setSong function"
    assert "uploadSong" in js, "app.js missing uploadSong function"
    assert "loadCandidateSongs" in js, "app.js missing loadCandidateSongs"
    assert "loadRecentSongs" in js, "app.js missing loadRecentSongs"
    assert "loadCurrentSong" in js, "app.js missing loadCurrentSong"

    # JS: the documented endpoints must be referenced.
    assert "/api/song" in js, "app.js does not call /api/song"
    assert "/api/song-info" in js, "app.js does not call /api/song-info"
    assert "/api/set-song-path" in js, "app.js does not call /api/set-song-path"
    assert "/api/upload-song" in js, "app.js does not call /api/upload-song"
    assert "/api/candidate-songs" in js, (
        "app.js does not call /api/candidate-songs"
    )
    assert "/api/recent-songs" in js, "app.js does not call /api/recent-songs"


def test_import_from_devices_panel_wired() -> None:
    """Issue: Import-from-devices panel above the source grid.

    Static check — confirms the DOM hooks, JS entry points, CSS class, and
    referenced HTTP endpoints exist. The backend (aftermovie.import_sources +
    /api/import* routes) is wired by parallel agents; this only verifies the
    front-end stays in lockstep with the documented contracts.
    """
    html = _read("index.html")
    css = _read("style.css")
    js = _read("app.js")

    # DOM: the panel container, two date inputs, primary + secondary buttons,
    # and a status area for live job updates.
    assert 'id="import-panel"' in html, "index.html missing #import-panel"
    assert html.count('type="date"') >= 2, (
        "index.html needs at least two date inputs inside #import-panel"
    )
    assert 'id="import-btn"' in html, "index.html missing #import-btn"
    assert 'id="dry-run-btn"' in html, "index.html missing #dry-run-btn"
    assert 'id="import-status"' in html, "index.html missing #import-status area"

    # CSS: a dedicated panel rule using existing tokens (no new colors).
    assert ".import-panel" in css, "style.css missing .import-panel rule"

    # JS: greppable entry points and the documented endpoint names.
    assert "loadImportSources" in js, "app.js missing loadImportSources"
    assert "startImport" in js, "app.js missing startImport"
    assert "pollImportStatus" in js, "app.js missing pollImportStatus"
    assert "/api/import-sources" in js, "app.js does not call /api/import-sources"
    assert "/api/import" in js, "app.js does not call /api/import"
    assert "/api/import-status" in js, "app.js does not call /api/import-status"
