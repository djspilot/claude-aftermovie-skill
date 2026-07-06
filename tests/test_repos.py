"""Tests for the Catalog + Plan repositories.

These are the units the rest of the pipeline talks to for on-disk catalog
and plan state. The repositories own id derivation, the
`_aftermovie.catalog_id` / `_aftermovie.plan_id` stamping, and the
cache-hit lookup that `run_auto` consults before re-running analyze.
"""
from __future__ import annotations

import time
from pathlib import Path

from aftermovie import config
from aftermovie.repos import (
    CatalogRepository,
    PlanIdOpts,
    PlanRepository,
)


def _isolated_data_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect `config.data_dir()` into a tmp dir so the repos write under
    the test's tmp tree instead of the developer's real state dir."""
    fake = tmp_path / "state"
    fake.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake)
    return fake


def _seed_clips(tmp_path: Path) -> Path:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(b"fake video bytes")
    return clips


# ---- CatalogRepository -----------------------------------------------------

def test_catalog_repository_put_stamps_id(tmp_path: Path, monkeypatch) -> None:
    """`CatalogRepository.put(folder, catalog)` must stamp the catalog dict
    with `_aftermovie.catalog_id` matching `id_for(folder)` before writing."""
    _isolated_data_dir(tmp_path, monkeypatch)
    clips = _seed_clips(tmp_path)
    repo = CatalogRepository()

    catalog = {"clips": [{"path": "a.mp4"}]}
    out_path = repo.put(clips, catalog)

    cid = repo.id_for(clips)
    assert out_path.is_file(), f"expected catalog at {out_path}"
    assert catalog["_aftermovie"]["catalog_id"] == cid, (
        "put() must stamp _aftermovie.catalog_id onto the catalog dict in place"
    )
    # And the same id round-trips through the on-disk file.
    import json
    body = json.loads(out_path.read_text())
    assert body["_aftermovie"]["catalog_id"] == cid


def test_catalog_repository_get_returns_cached_catalog(
    tmp_path: Path, monkeypatch,
) -> None:
    """After a put(), get(folder) must return the same catalog dict (cache hit)."""
    _isolated_data_dir(tmp_path, monkeypatch)
    clips = _seed_clips(tmp_path)
    repo = CatalogRepository()

    # First call: miss.
    assert repo.get(clips) is None, "no catalog on disk yet — expected None"

    # Put a catalog, then second call should be a hit.
    catalog = {"clips": [{"path": "a.mp4"}], "folder": str(clips)}
    repo.put(clips, catalog)
    hit = repo.get(clips)
    assert hit is not None, "expected cache hit after put()"
    assert hit["clips"] == [{"path": "a.mp4"}]
    assert hit["_aftermovie"]["catalog_id"] == repo.id_for(clips)


# ---- PlanRepository -------------------------------------------------------

def test_plan_repository_get_latest_for_catalog_returns_none_when_empty(
    tmp_path: Path, monkeypatch,
) -> None:
    """No plan on disk → None (so the GUI can emit a 404)."""
    _isolated_data_dir(tmp_path, monkeypatch)
    repo = PlanRepository()
    assert repo.get_latest_for_catalog("nonexistent-cid") is None


def test_plan_repository_get_latest_for_catalog_returns_most_recent(
    tmp_path: Path, monkeypatch,
) -> None:
    """When multiple plans share a catalog_id, return the newest by mtime."""
    _isolated_data_dir(tmp_path, monkeypatch)
    repo = PlanRepository()
    cid = "fake-cid-1234"
    song = tmp_path / "song.wav"
    song.write_bytes(b"")

    # First plan — older.
    older_opts = PlanIdOpts(theme="cinematic", max_length=30.0, aspect="16:9")
    older_plan = {"entries": [], "marker": "older"}
    older_path = repo.put(cid, song, older_opts, older_plan)
    # Force the mtime so the order is unambiguous even on fast filesystems.
    older_mtime = time.time() - 60
    import os
    os.utime(older_path, (older_mtime, older_mtime))

    # Second plan — newer, different opts so it gets a different id.
    newer_opts = PlanIdOpts(theme="punchy", max_length=60.0, aspect="16:9")
    newer_plan = {"entries": [], "marker": "newer"}
    newer_path = repo.put(cid, song, newer_opts, newer_plan)
    newer_mtime = time.time()
    os.utime(newer_path, (newer_mtime, newer_mtime))

    # And a plan for a *different* catalog — must not be returned.
    other_opts = PlanIdOpts(theme="cinematic", max_length=30.0, aspect="9:16")
    other_plan = {"entries": [], "marker": "other"}
    repo.put("other-cid", song, other_opts, other_plan)

    hit = repo.get_latest_for_catalog(cid)
    assert hit is not None
    assert hit["marker"] == "newer", (
        f"expected the most-recent plan tagged with cid, got marker={hit.get('marker')}"
    )
    # And the stamp matches.
    assert hit["_aftermovie"]["catalog_id"] == cid


def test_plan_repository_put_stamps_both_ids(tmp_path: Path, monkeypatch) -> None:
    """put() must stamp _aftermovie.catalog_id AND _aftermovie.plan_id."""
    _isolated_data_dir(tmp_path, monkeypatch)
    repo = PlanRepository()
    cid = "cid-abc"
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    opts = PlanIdOpts(theme=None, max_length=None, aspect="16:9")
    plan = {"entries": []}

    repo.put(cid, song, opts, plan)

    pid = repo.id_for(cid, song, opts.theme, opts.max_length, opts.aspect, opts.seed)
    assert plan["_aftermovie"]["catalog_id"] == cid
    assert plan["_aftermovie"]["plan_id"] == pid


def test_catalog_id_changes_when_selection_changes(tmp_path: Path, monkeypatch) -> None:
    """Blocking a clip in the GUI writes the selection sidecar; the catalog
    id must change so the cached (unfiltered) catalog misses and analyze
    re-runs with the exclusion applied. Rewriting the SAME selection (the
    GUI saves on every render start) must NOT change the id."""
    _isolated_data_dir(tmp_path, monkeypatch)
    clips = _seed_clips(tmp_path)
    repo = CatalogRepository()

    base = repo.id_for(clips)

    sel = clips / ".aftermovie-selection.json"
    sel.write_text('{"excluded": ["a.mp4"]}')
    blocked = repo.id_for(clips)
    assert blocked != base

    # Same content, fresh mtime → id stays stable (no pointless re-analyze).
    time.sleep(0.01)
    sel.write_text('{"excluded": ["a.mp4"]}')
    assert repo.id_for(clips) == blocked

    # Unblocking changes the content again → new id again.
    sel.write_text('{"excluded": []}')
    assert repo.id_for(clips) != blocked
