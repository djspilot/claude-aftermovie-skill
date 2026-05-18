"""Tests for the catalog/plan on-disk cache wiring in `pipeline_runner`.

`run_auto` should consult `state.catalog_dir() / <cid>.json` before calling
`cmd_analyze`. A hit copies the cached catalog into the run's workdir and
skips the (expensive) analyze stage entirely. A miss runs analyze and then
persists the resulting catalog under the same content-derived id so the
next run is fast. The plan side mirrors this — scoring is cheap so we always
re-score, but the resulting plan gets stamped + saved for the GUI's
`/api/plan` endpoint to find.

These tests stub `cmd_analyze`/`cmd_score`/`cmd_render` so they're fast and
hermetic; they also redirect `state.data_dir()` into `tmp_path` so the cache
they exercise doesn't collide with the developer's real state directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aftermovie import config, pipeline_runner, state
from aftermovie.pipeline_runner import AutoOpts, run_auto


def _patch_pipeline(monkeypatch) -> dict[str, int]:
    """Replace cmd_analyze/score/render with recorders. Returns a counter
    dict (`{"analyze": n, "score": n, "render": n}`) so tests can assert
    how many times each stage actually ran."""
    counts: dict[str, int] = {"analyze": 0, "score": 0, "render": 0}

    def fake_analyze(args: argparse.Namespace) -> None:
        counts["analyze"] += 1
        # Mimic cmd_analyze: write a catalog JSON shaped like the real one.
        Path(args.out).write_text(json.dumps({"clips": [{"path": "x.mp4"}]}))

    def fake_score(args: argparse.Namespace) -> None:
        counts["score"] += 1
        plan = {
            "song": args.song,
            "aspect": args.aspect,
            "entries": [],
        }
        Path(args.out).write_text(json.dumps(plan))

    def fake_render(args: argparse.Namespace) -> None:
        counts["render"] += 1
        Path(args.output).write_bytes(b"")

    monkeypatch.setattr(pipeline_runner, "cmd_analyze", fake_analyze)
    monkeypatch.setattr(pipeline_runner, "cmd_score", fake_score)
    monkeypatch.setattr(pipeline_runner, "cmd_render", fake_render)
    return counts


def _isolated_state_dir(tmp_path: Path, monkeypatch) -> Path:
    """Point state.data_dir() at a tmp dir so cache writes don't escape."""
    fake_data = tmp_path / "state"
    fake_data.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake_data)
    monkeypatch.setattr(state, "data_dir", lambda: fake_data)
    return fake_data


def _seed_clips_and_song(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a clips folder with one fake video, an empty song, an output path."""
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(b"fake video bytes")
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    output = tmp_path / "out.mp4"
    return clips, song, output


def test_first_run_writes_catalog_to_state(tmp_path: Path, monkeypatch) -> None:
    """A cold run should call cmd_analyze and leave a stamped catalog in
    `state.catalog_dir() / <cid>.json` for the next run to pick up."""
    counts = _patch_pipeline(monkeypatch)
    _isolated_state_dir(tmp_path, monkeypatch)
    clips, song, output = _seed_clips_and_song(tmp_path)

    cid = state.catalog_id_for(clips)
    run_auto(clips, song, output, AutoOpts())

    assert counts["analyze"] == 1, "cold cache must invoke cmd_analyze"
    cached = state.catalog_dir() / f"{cid}.json"
    assert cached.is_file(), f"expected cached catalog at {cached}"
    body = json.loads(cached.read_text())
    assert body.get("_aftermovie", {}).get("catalog_id") == cid, (
        "catalog must be stamped with its derived id before caching"
    )


def test_warm_run_skips_analyze(tmp_path: Path, monkeypatch) -> None:
    """Second run on the same source folder must hit the cache and skip analyze."""
    counts = _patch_pipeline(monkeypatch)
    _isolated_state_dir(tmp_path, monkeypatch)
    clips, song, output = _seed_clips_and_song(tmp_path)

    # First run populates the cache.
    run_auto(clips, song, output, AutoOpts())
    assert counts["analyze"] == 1

    # Second run on the same untouched folder must be a cache hit.
    run_auto(clips, song, output, AutoOpts())
    assert counts["analyze"] == 1, (
        f"expected analyze to be skipped on warm run; ran {counts['analyze']} times"
    )
    # Score + render still ran both times.
    assert counts["score"] == 2
    assert counts["render"] == 2


def test_force_reanalyze_bypasses_cache(tmp_path: Path, monkeypatch) -> None:
    """`force_reanalyze=True` must re-run cmd_analyze even when a cache hit exists."""
    counts = _patch_pipeline(monkeypatch)
    _isolated_state_dir(tmp_path, monkeypatch)
    clips, song, output = _seed_clips_and_song(tmp_path)

    run_auto(clips, song, output, AutoOpts())
    assert counts["analyze"] == 1

    run_auto(clips, song, output, AutoOpts(force_reanalyze=True))
    assert counts["analyze"] == 2, (
        "force_reanalyze=True must bypass the cache and re-invoke cmd_analyze"
    )


def test_catalog_id_changes_when_file_added(tmp_path: Path, monkeypatch) -> None:
    """Adding a file to the source folder must change `catalog_id_for`, which
    is what makes the cache automatically invalidate itself when the user
    drops in new footage."""
    _isolated_state_dir(tmp_path, monkeypatch)
    clips, _song, _out = _seed_clips_and_song(tmp_path)

    cid_before = state.catalog_id_for(clips)
    (clips / "b.mp4").write_bytes(b"second clip")
    cid_after = state.catalog_id_for(clips)

    assert cid_before != cid_after, (
        "adding a file to the source folder must produce a different catalog_id"
    )


def test_plan_persisted_to_state(tmp_path: Path, monkeypatch) -> None:
    """Each run should stamp the resulting plan with its catalog_id + plan_id
    and save it under `state.plan_dir() / <pid>.json` so the GUI's /api/plan
    endpoint has something to match against."""
    _patch_pipeline(monkeypatch)
    _isolated_state_dir(tmp_path, monkeypatch)
    clips, song, output = _seed_clips_and_song(tmp_path)

    opts = AutoOpts(max_length=10.0, aspect="16:9")
    run_auto(clips, song, output, opts)

    cid = state.catalog_id_for(clips)
    pid = state.plan_id_for(cid, song.resolve(), opts.theme, opts.max_length,
                            opts.aspect, 0)
    cached_plan = state.plan_dir() / f"{pid}.json"
    assert cached_plan.is_file(), f"expected cached plan at {cached_plan}"
    body = json.loads(cached_plan.read_text())
    tag = body.get("_aftermovie", {})
    assert tag.get("catalog_id") == cid
    assert tag.get("plan_id") == pid
