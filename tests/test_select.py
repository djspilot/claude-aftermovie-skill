"""Tests for the `aftermovie select` web GUI backend.

These tests exercise the stdlib HTTP server end-to-end (real bind on a
random port, real HTTP requests) so the wiring between the JSON endpoints
and the on-disk `.aftermovie-selection.json` is verified holistically.
"""
from __future__ import annotations

import json
import shutil
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from aftermovie.analyze.preferences import (
    PREFERENCES_FILENAME,
    clear_cache as clear_preferences_cache,
    load_preferences,
)
from aftermovie.analyze.selection import (
    SELECTION_FILENAME,
    clear_cache,
    load_excluded,
)
from aftermovie.select.server import SelectServer
from aftermovie.select.thumbnails import thumb_path_for


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def _free_port() -> int:
    """Bind ephemeral port; return the OS-assigned port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _seed_folder(tmp_path: Path, fixtures_dir: Path) -> Path:
    """Copy a couple of fixture clips + a generated still into tmp_path."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    shutil.copy(fixtures_dir / "clip_a.mp4", clips_dir / "clip_a.mp4")
    shutil.copy(fixtures_dir / "clip_b.mp4", clips_dir / "clip_b.mp4")
    Image.new("RGB", (200, 150), (180, 30, 30)).save(clips_dir / "still.jpg")
    return clips_dir


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read(), dict(resp.headers)


def _http_post(url: str, body: dict, timeout: float = 5.0) -> tuple[int, bytes]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def test_sources_endpoint_lists_files(tmp_path: Path, fixtures_dir: Path) -> None:
    """`GET /api/sources` returns the folder's clips with selected=true by default."""
    clear_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        status, body, _ = _http_get(f"{srv.url}api/sources")
    assert status == 200
    data = json.loads(body)
    assert isinstance(data, list)
    names = {item["name"] for item in data}
    assert {"clip_a.mp4", "clip_b.mp4", "still.jpg"}.issubset(names), (
        f"expected the three seeded files in sources, got {names}"
    )
    # All items default to selected=true when no sidecar exists.
    assert all(item["selected"] is True for item in data)
    kinds = {item["name"]: item["kind"] for item in data}
    assert kinds["clip_a.mp4"] == "video"
    assert kinds["still.jpg"] == "still"
    # Thumb URLs are stable + look like /thumbs/<hex>.jpg
    for item in data:
        assert item["thumb_url"].startswith("/thumbs/") and item["thumb_url"].endswith(".jpg")


def test_selection_round_trip(tmp_path: Path, fixtures_dir: Path) -> None:
    """POSTing excluded paths persists them and flips `selected` on /api/sources."""
    clear_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    excluded_path = str((clips_dir / "clip_b.mp4").resolve())
    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        status, _ = _http_post(f"{srv.url}api/selection",
                               {"excluded": [excluded_path]})
        assert status == 200

        # Sidecar landed on disk with the expected schema.
        sidecar = clips_dir / SELECTION_FILENAME
        assert sidecar.is_file()
        payload = json.loads(sidecar.read_text())
        assert payload["excluded"] == [excluded_path]
        assert payload["generated_by"] == "aftermovie-select"
        assert payload["version"] == 1

        # /api/sources reflects the new selection state.
        clear_cache()  # force a fresh read on the next request
        status, body, _ = _http_get(f"{srv.url}api/sources")
        assert status == 200
        data = json.loads(body)
        by_name = {item["name"]: item for item in data}
        assert by_name["clip_b.mp4"]["selected"] is False
        assert by_name["clip_a.mp4"]["selected"] is True


def test_thumbnail_is_jpeg(tmp_path: Path) -> None:
    """`thumb_path_for` produces a JPEG (FF D8 FF magic) for a still image."""
    src = tmp_path / "thumb_src.jpg"
    Image.new("RGB", (640, 480), (50, 120, 200)).save(src)

    out = thumb_path_for(src)
    assert out is not None and out.is_file()
    head = out.read_bytes()[:4]
    # JPEG magic: SOI (FF D8) followed by another segment marker (FF xx).
    assert head[:3] == b"\xff\xd8\xff", f"not a JPEG, got bytes {head!r}"
    # The next byte names the segment type (E0=JFIF, E1=Exif, DB=quant table,
    # EE=Adobe, FE=comment). PIL with optimize=True can emit any of these.
    assert head[3] in (0xE0, 0xE1, 0xDB, 0xEE, 0xFE), (
        f"unexpected JPEG segment marker after SOI: {head[3]:#x}"
    )

    # Also exercise the HTTP path: GET /thumbs/<hash>.jpg should round-trip.
    clips_dir = tmp_path
    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        # Hit /api/sources first so the server knows about our file.
        status, body, _ = _http_get(f"{srv.url}api/sources")
        assert status == 200
        data = json.loads(body)
        # Find the entry for our still and request its thumb_url.
        row = next(r for r in data if r["name"] == "thumb_src.jpg")
        status, body, headers = _http_get(f"{srv.url.rstrip('/')}{row['thumb_url']}")
        assert status == 200
        assert headers.get("Content-Type") == "image/jpeg"
        assert body[:2] == b"\xff\xd8"


def test_api_plan_returns_404_when_no_plan(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """`GET /api/plan` returns 404 + `{"error":"no_plan"}` when nothing in
    state.plan_dir() is tagged with the current clips_root's catalog_id."""
    clear_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)

    # Point state's data_dir at an empty tmp dir so we don't see leftovers
    # from other test runs on this machine.
    from aftermovie import config, state
    fake_data = tmp_path / "state"
    fake_data.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake_data)
    monkeypatch.setattr(state, "data_dir", lambda: fake_data)

    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        req = urllib.request.Request(f"{srv.url}api/plan", method="GET")
        try:
            urllib.request.urlopen(req, timeout=5.0)
            raise AssertionError("expected 404 from /api/plan with no plans on disk")
        except urllib.error.HTTPError as e:
            assert e.code == 404
            body = json.loads(e.read())
            assert body == {"error": "no_plan"}


def test_api_plan_returns_matching_plan(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """When a plan tagged with the current catalog_id is on disk, /api/plan
    returns its JSON verbatim."""
    clear_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)

    # Redirect state dirs into tmp_path so this test is hermetic.
    from aftermovie import config, state
    fake_data = tmp_path / "state"
    fake_data.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake_data)
    monkeypatch.setattr(state, "data_dir", lambda: fake_data)

    catalog_id = state.catalog_id_for(clips_dir)
    plan = {
        "entries": [{"source": "x.mp4", "start_s": 0.0, "end_s": 1.0}],
        "aspect": "16:9",
        "_aftermovie": {"catalog_id": catalog_id},
    }
    state.save_plan("test-plan-1234", plan)

    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        status, body, _ = _http_get(f"{srv.url}api/plan")
    assert status == 200
    payload = json.loads(body)
    assert payload["entries"] == plan["entries"]
    assert payload["_aftermovie"]["catalog_id"] == catalog_id


def test_preferences_post_writes_sidecar(tmp_path: Path, fixtures_dir: Path) -> None:
    """POST /api/preferences persists favorited+banned to the sidecar."""
    clear_cache()
    clear_preferences_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    fav = str((clips_dir / "clip_a.mp4").resolve())
    ban = str((clips_dir / "clip_b.mp4").resolve())
    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        status, body = _http_post(f"{srv.url}api/preferences",
                                  {"favorited": [fav], "banned": [ban]})
    assert status == 200
    assert json.loads(body) == {"ok": True}

    sidecar = clips_dir / PREFERENCES_FILENAME
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["favorited"] == [fav]
    assert payload["banned"] == [ban]
    # pinned_entries is reserved — defaults to [] when the client omits it.
    assert payload["pinned_entries"] == []
    assert payload["generated_by"] == "aftermovie-select"
    assert payload["version"] == 1


def test_preferences_get_returns_dict(tmp_path: Path, fixtures_dir: Path) -> None:
    """GET /api/preferences returns the persisted favorited/banned/pinned dict."""
    clear_cache()
    clear_preferences_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    fav = str((clips_dir / "clip_a.mp4").resolve())
    ban = str((clips_dir / "clip_b.mp4").resolve())

    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        # First call before any POST → empty defaults, not 404.
        status, body, _ = _http_get(f"{srv.url}api/preferences")
        assert status == 200
        assert json.loads(body) == {"favorited": [], "banned": [], "pinned_entries": []}

        # Persist a state, then GET should reflect it.
        _http_post(f"{srv.url}api/preferences",
                   {"favorited": [fav], "banned": [ban]})
        clear_preferences_cache()  # force a fresh disk read
        status, body, _ = _http_get(f"{srv.url}api/preferences")
    assert status == 200
    data = json.loads(body)
    assert data["favorited"] == [fav]
    assert data["banned"] == [ban]
    assert data["pinned_entries"] == []

    # Sanity: the helper sees the same state we wrote.
    clear_preferences_cache()
    prefs = load_preferences(clips_dir)
    assert prefs["favorited"] == [fav]
    assert prefs["banned"] == [ban]


def test_excluded_files_skipped_in_discover_sources(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """`.aftermovie-selection.json` excludes files from `discover_sources`."""
    from aftermovie.analyze.clip import discover_sources

    clear_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    excluded_path = str((clips_dir / "clip_a.mp4").resolve())
    (clips_dir / SELECTION_FILENAME).write_text(json.dumps({
        "excluded": [excluded_path],
        "generated_by": "aftermovie-select",
        "version": 1,
    }))

    # Sanity: load_excluded sees what we wrote.
    excluded = load_excluded(clips_dir)
    assert excluded_path in excluded

    sources = discover_sources(clips_dir, include_stills=False)
    abs_sources = {str(p) for p in sources}
    assert excluded_path not in abs_sources, (
        f"excluded clip leaked into discover_sources: {abs_sources}"
    )
    # And the unexcluded clip is still there.
    assert any(p.name == "clip_b.mp4" for p in sources), (
        f"expected clip_b.mp4 in remaining sources, got {[p.name for p in sources]}"
    )
