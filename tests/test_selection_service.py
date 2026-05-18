"""Tests for `SelectionService` — the GUI's domain Interface, sans HTTP.

These tests verify the service directly (no `SelectServer`, no sockets):
that's the whole point of the refactor — the Interface that used to be
trapped inside `_Handler` is now testable without spinning a real server.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from PIL import Image

from aftermovie.analyze.preferences import (
    PREFERENCES_FILENAME,
    clear_cache as clear_preferences_cache,
)
from aftermovie.analyze.selection import (
    SELECTION_FILENAME,
    clear_cache as clear_selection_cache,
)
from aftermovie.select.service import (
    ALLOWED_RENDER_OVERRIDES,
    ImportJob,
    RenderJob,
    SelectionService,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def _seed_folder(tmp_path: Path, fixtures_dir: Path) -> Path:
    """Copy a couple of fixture clips + a generated still into tmp_path."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    shutil.copy(fixtures_dir / "clip_a.mp4", clips_dir / "clip_a.mp4")
    shutil.copy(fixtures_dir / "clip_b.mp4", clips_dir / "clip_b.mp4")
    Image.new("RGB", (200, 150), (180, 30, 30)).save(clips_dir / "still.jpg")
    return clips_dir


# ---- list_sources ----------------------------------------------------------

def test_list_sources_returns_expected_rows(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """`list_sources` mirrors `/api/sources` — three rows for three fixture files."""
    clear_selection_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)

    rows = svc.list_sources()
    names = {r.name for r in rows}
    assert {"clip_a.mp4", "clip_b.mp4", "still.jpg"}.issubset(names)
    # All items default to selected=True when no sidecar exists.
    assert all(r.selected for r in rows)
    kinds = {r.name: r.kind for r in rows}
    assert kinds["clip_a.mp4"] == "video"
    assert kinds["still.jpg"] == "still"
    # Thumb URLs are stable + look like /thumbs/<hex>.jpg
    for r in rows:
        assert r.thumb_url.startswith("/thumbs/") and r.thumb_url.endswith(".jpg")


def test_list_sources_handles_missing_folder(tmp_path: Path) -> None:
    """A non-existent clips_root yields an empty list, not an exception."""
    svc = SelectionService(tmp_path / "does_not_exist")
    assert svc.list_sources() == []


# ---- selection round-trip --------------------------------------------------

def test_save_selection_round_trips_through_get_selection(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """`save_selection` persists; `get_selection` reads it back."""
    clear_selection_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    excluded_path = str((clips_dir / "clip_b.mp4").resolve())

    svc = SelectionService(clips_dir)
    result = svc.save_selection([excluded_path])

    # SaveResult points at the sidecar on disk.
    assert result.sidecar == clips_dir / SELECTION_FILENAME
    assert result.filename == SELECTION_FILENAME
    assert result.n_items == 1
    assert result.sidecar.is_file()

    payload = json.loads(result.sidecar.read_text())
    assert payload["excluded"] == [excluded_path]
    assert payload["generated_by"] == "aftermovie-select"

    # get_selection sees what we wrote — and list_sources reflects it.
    clear_selection_cache()
    assert svc.get_selection() == [excluded_path]
    by_name = {r.name: r for r in svc.list_sources()}
    assert by_name["clip_b.mp4"].selected is False
    assert by_name["clip_a.mp4"].selected is True


# ---- preferences round-trip ------------------------------------------------

def test_save_preferences_round_trips_through_get_preferences(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """`save_preferences` persists; `get_preferences` reads back the same dict."""
    clear_selection_cache()
    clear_preferences_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    fav = str((clips_dir / "clip_a.mp4").resolve())
    ban = str((clips_dir / "clip_b.mp4").resolve())

    svc = SelectionService(clips_dir)
    # Default state before any save: all three list fields empty.
    assert svc.get_preferences() == {
        "favorited": [], "banned": [], "pinned_entries": [],
    }

    svc.save_preferences({"favorited": [fav], "banned": [ban]})

    sidecar = clips_dir / PREFERENCES_FILENAME
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["favorited"] == [fav]
    assert payload["banned"] == [ban]
    assert payload["pinned_entries"] == []  # reserved, default-empty

    clear_preferences_cache()
    prefs = svc.get_preferences()
    assert prefs["favorited"] == [fav]
    assert prefs["banned"] == [ban]
    assert prefs["pinned_entries"] == []


# ---- latest_plan -----------------------------------------------------------

def test_latest_plan_returns_none_when_no_plan_on_disk(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """No matching plan on disk → `latest_plan` returns None (not an exception)."""
    clear_selection_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)

    from aftermovie import config, state
    fake_data = tmp_path / "state"
    fake_data.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake_data)
    monkeypatch.setattr(state, "data_dir", lambda: fake_data)

    svc = SelectionService(clips_dir)
    assert svc.latest_plan() is None


def test_latest_plan_returns_matching_plan(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """When a plan tagged with this folder's catalog_id is on disk, return it."""
    clear_selection_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)

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
    state.save_plan("test-plan-svc-1", plan)

    svc = SelectionService(clips_dir)
    out = svc.latest_plan()
    assert out is not None
    assert out["entries"] == plan["entries"]
    assert out["_aftermovie"]["catalog_id"] == catalog_id


# ---- available_options -----------------------------------------------------

def test_available_options_returns_documented_keys(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """The options dict has the keys the frontend dropdowns rely on."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)

    opts = svc.available_options()
    for key in ("luts", "themes", "audio_mix", "pace",
                "transitions", "aspect", "resolution"):
        assert key in opts, f"missing key {key}"
    assert "ducked" in opts["audio_mix"]
    assert "16:9" in opts["aspect"]


# ---- thumb_for_key ---------------------------------------------------------

def test_thumb_for_key_returns_bytes_for_known_source(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """`thumb_for_key` returns raw JPG bytes for a key produced by `list_sources`."""
    clear_selection_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)

    rows = svc.list_sources()
    row = next(r for r in rows if r.name == "still.jpg")
    key = row.thumb_url.removeprefix("/thumbs/").removesuffix(".jpg")

    data = svc.thumb_for_key(key)
    assert data is not None
    assert data[:2] == b"\xff\xd8"  # JPEG SOI


def test_thumb_for_key_returns_none_for_unknown_key(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """An unknown key yields None (the HTTP layer renders that as 404)."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)
    assert svc.thumb_for_key("deadbeefdeadbeefdeadbeef") is None


# ---- start_render + status -------------------------------------------------

def _stub_runner(job: RenderJob, clips: Path, song, output, overrides) -> None:
    """A drop-in for `_run_render_job` used by tests — no ffmpeg, no librosa.

    Marks the job as `done` with a fake output path so we can verify the
    job lifecycle (queue → start → finish) end-to-end without rendering.
    """
    job.output_path = str(clips / "fake-output.mp4")
    job.log_tail.append(f"stub ran with song={song} overrides={sorted(overrides)}")
    job.state = "done"
    job.finished_at = time.time()


def test_start_render_then_status_happy_path(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """`start_render` queues a job; `status` reflects it through completion."""
    clear_selection_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    song = fixtures_dir / "tone.wav"

    svc = SelectionService(clips_dir, song_default=song,
                           render_runner=_stub_runner)

    body = {"theme": "cinematic", "preview": True,
            "excluded": [str((clips_dir / "clip_b.mp4").resolve())]}
    job = svc.start_render(body)
    assert job.job_id

    # Wait for the worker to finish (stub is near-instant; cap at 2s).
    deadline = time.time() + 2.0
    while time.time() < deadline:
        snap = svc.status(job.job_id)
        assert snap is not None
        if snap["state"] != "running":
            break
        time.sleep(0.02)

    snap = svc.status(job.job_id)
    assert snap is not None
    assert snap["state"] == "done"
    assert snap["output_path"] == str(clips_dir / "fake-output.mp4")
    assert snap["error"] is None
    assert "stub ran with song=" in snap["log_tail"]
    assert snap["finished_at"] is not None

    # The excluded path was also persisted as part of start_render.
    sidecar = clips_dir / SELECTION_FILENAME
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["excluded"] == [str((clips_dir / "clip_b.mp4").resolve())]


def test_status_returns_none_for_unknown_job_id(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """An unknown job_id yields None (renders as 404 in the HTTP layer)."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)
    assert svc.status("no-such-job") is None


def test_start_render_filters_unknown_overrides(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """Only whitelisted override keys flow through to the runner."""
    clear_selection_cache()
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    song = fixtures_dir / "tone.wav"

    captured: dict = {}

    def capture_runner(job, clips, s, output, overrides):
        captured["overrides"] = dict(overrides)
        job.state = "done"
        job.finished_at = time.time()

    svc = SelectionService(clips_dir, song_default=song,
                           render_runner=capture_runner)
    job = svc.start_render({
        "theme": "punchy",         # allowed
        "pace": "fast",            # allowed
        "secret_back_door": True,  # NOT allowed
        "song": str(song),         # routed separately, not via overrides
    })
    # Wait briefly for the stub.
    deadline = time.time() + 1.0
    while time.time() < deadline and svc.status(job.job_id)["state"] == "running":
        time.sleep(0.02)

    assert "secret_back_door" not in captured["overrides"]
    assert captured["overrides"]["theme"] == "punchy"
    assert captured["overrides"]["pace"] == "fast"
    # Sanity check: the whitelist is the source of truth.
    assert "secret_back_door" not in ALLOWED_RENDER_OVERRIDES
    assert "theme" in ALLOWED_RENDER_OVERRIDES


# ---- import sources / start_import / import_status -------------------------

class _FakeImportItem:
    """Stand-in for `aftermovie.import_sources.ImportItem` — duck-typed."""

    def __init__(self, src_path: str, kind: str = "video", size_bytes: int = 1024,
                 source_label: str = "fake") -> None:
        self.src_path = src_path
        self.captured_at = 0.0
        self.kind = kind
        self.size_bytes = size_bytes
        self.source_label = source_label
        self.extra = {}


class _FakeCopyResult:
    def __init__(self, copied: int = 0, skipped: int = 0, failed: int = 0,
                 bytes_written: int = 0, dest_folder: str = "") -> None:
        self.copied = copied
        self.skipped = skipped
        self.failed = failed
        self.bytes_written = bytes_written
        self.dest_folder = dest_folder


class _FakeSource:
    """Duck-typed `ImportSource` for service tests; tracks calls."""

    def __init__(self, name: str = "fake", label: str = "Fake",
                 items: list[_FakeImportItem] | None = None,
                 available: bool = True) -> None:
        self.name = name
        self.label = label
        self._items = items or []
        self._available = available
        self.copy_calls: list[tuple] = []

    def available(self) -> bool:
        return self._available

    def list_in_range(self, since, until):
        return list(self._items)

    def copy_into(self, items, dest_folder, progress_cb=None):
        self.copy_calls.append((list(items), Path(dest_folder)))
        # base.copy_files signature: (done, total, src_path). Mirror that
        # contract here so the service's _progress_cb gets the right shape.
        total = len(items)
        for done, item in enumerate(items, start=1):
            if progress_cb is not None:
                progress_cb(done, total, item.src_path)
        return _FakeCopyResult(copied=len(items), dest_folder=str(dest_folder))


def _install_fake_import_module(monkeypatch, sources: list[_FakeSource]) -> None:
    """Inject a fake `aftermovie.import_sources` Module exporting `all_sources`.

    The real Module is being built by a parallel agent; this lets the service
    tests run regardless of whether that work has landed.
    """
    import sys
    import types

    fake_mod = types.ModuleType("aftermovie.import_sources")
    fake_mod.all_sources = lambda: list(sources)  # type: ignore[attr-defined]
    fake_mod.ImportItem = _FakeImportItem  # type: ignore[attr-defined]
    fake_mod.CopyResult = _FakeCopyResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aftermovie.import_sources", fake_mod)


def test_available_import_sources_projects_each_source(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """`available_import_sources` returns `{name, label, available}` per source."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    _install_fake_import_module(monkeypatch, [
        _FakeSource(name="photos_library", label="Photos library", available=True),
        _FakeSource(name="gopro_X", label="GoPro X", available=False),
    ])
    svc = SelectionService(clips_dir)

    rows = svc.available_import_sources()
    assert len(rows) >= 1
    by_name = {r["name"]: r for r in rows}
    assert by_name["photos_library"] == {
        "name": "photos_library", "label": "Photos library", "available": True,
    }
    assert by_name["gopro_X"]["available"] is False


def test_start_import_happy_path_copies_items(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """`start_import` runs the worker; status reaches `done` with copied=2."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    items = [
        _FakeImportItem(src_path=str(fixtures_dir / "clip_a.mp4")),
        _FakeImportItem(src_path=str(fixtures_dir / "clip_b.mp4")),
    ]
    fake = _FakeSource(name="fake", label="Fake", items=items)
    _install_fake_import_module(monkeypatch, [fake])

    dest_parent = tmp_path / "imports"
    svc = SelectionService(clips_dir)
    job = svc.start_import(
        since="2026-05-15",
        until="2026-05-18",
        source_names=["fake"],
        dest_parent=str(dest_parent),
    )
    assert isinstance(job, ImportJob)
    assert job.dest_folder == str(dest_parent / "2026-05-15_to_2026-05-18")

    # Wait for the worker (sub-millisecond stub; cap at 2s).
    deadline = time.time() + 2.0
    while time.time() < deadline:
        snap = svc.import_status(job.job_id)
        assert snap is not None
        if snap["state"] != "running":
            break
        time.sleep(0.02)

    snap = svc.import_status(job.job_id)
    assert snap is not None, "job vanished from the import-job dict"
    assert snap["state"] == "done", snap
    assert snap["copied"] == 2
    assert snap["total"] == 2
    assert snap["failed"] == 0
    assert snap["dest_folder"] == str(dest_parent / "2026-05-15_to_2026-05-18")
    assert Path(snap["dest_folder"]).is_dir()
    assert len(fake.copy_calls) == 1


def test_start_import_dry_run_skips_copy(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """`dry_run=True` counts items via `list_in_range`; no `copy_into` call."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    items = [_FakeImportItem(src_path="a"), _FakeImportItem(src_path="b"),
             _FakeImportItem(src_path="c")]
    fake = _FakeSource(name="fake", label="Fake", items=items)
    _install_fake_import_module(monkeypatch, [fake])

    svc = SelectionService(clips_dir)
    job = svc.start_import(
        since="2026-05-15",
        until="2026-05-18",
        source_names=["fake"],
        dest_parent=str(tmp_path / "imports"),
        dry_run=True,
    )

    deadline = time.time() + 2.0
    while time.time() < deadline:
        snap = svc.import_status(job.job_id)
        assert snap is not None
        if snap["state"] != "running":
            break
        time.sleep(0.02)

    snap = svc.import_status(job.job_id)
    assert snap is not None
    assert snap["state"] == "done"
    assert snap["copied"] == 0
    assert snap["total"] == 3
    assert fake.copy_calls == [], "copy_into should not be called on dry_run"


def test_start_import_bad_date_format_raises(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """Garbage `since` raises ValueError — HTTP layer catches it for 400."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    _install_fake_import_module(monkeypatch, [_FakeSource()])
    svc = SelectionService(clips_dir)

    with pytest.raises(ValueError):
        svc.start_import(
            since="tomorrow",
            until="2026-05-18",
            source_names=["fake"],
        )


def test_import_status_unknown_returns_none(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """Unknown job_id → None (HTTP Adapter renders 404)."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)
    assert svc.import_status("no-such-import-job") is None
