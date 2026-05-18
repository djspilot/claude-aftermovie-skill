"""Tests for the shared analyze → score → render orchestration.

`run_auto` is the single entry point used by both the CLI (`aftermovie auto`)
and the MCP `auto` tool. The fast tests below stub the per-stage entry points
so we can assert theme-bundle expansion and override semantics without paying
for a real ffmpeg render. The slow test exercises the full pipeline on
fixtures and is gated by ffmpeg availability.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from aftermovie import pipeline_runner
from aftermovie.config import DEFAULT_MUSIC_DB
from aftermovie.pipeline_runner import AutoOpts, run_auto


def _patch_pipeline(monkeypatch) -> dict[str, argparse.Namespace]:
    """Replace cmd_analyze/score/render with recorders. Returns a dict
    populated with the captured Namespaces as each stage is called."""
    captured: dict[str, argparse.Namespace] = {}

    def fake_analyze(args: argparse.Namespace) -> None:
        captured["analyze"] = args
        Path(args.out).write_text(json.dumps({"clips": []}))

    def fake_score(args: argparse.Namespace) -> None:
        captured["score"] = args
        # Persist the score args so cmd_render reads a plausible plan;
        # the contents don't matter, only the keys we assert against.
        plan = {
            "entries": [],
            "lut": args.lut,
            "music_db": args.music_db,
            "audio_mix": getattr(args, "audio_mix", None),
            "transitions": getattr(args, "transitions", None),
            "pace": getattr(args, "pace", None),
        }
        Path(args.out).write_text(json.dumps(plan))

    def fake_render(args: argparse.Namespace, **_: object) -> None:
        captured["render"] = args
        # Touch the output so callers can verify the path was used.
        Path(args.output).write_bytes(b"")

    monkeypatch.setattr(pipeline_runner, "cmd_analyze", fake_analyze)
    monkeypatch.setattr(pipeline_runner, "cmd_score", fake_score)
    monkeypatch.setattr(pipeline_runner, "cmd_render", fake_render)
    return captured


def test_run_auto_applies_theme_bundle(tmp_path: Path, monkeypatch):
    """Theme=cinematic at otherwise-default knobs should thread soft transitions,
    ducked audio, and the cinematic LUT into the score stage."""
    captured = _patch_pipeline(monkeypatch)

    clips = tmp_path / "clips"
    clips.mkdir()
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    output = tmp_path / "out.mp4"

    opts = AutoOpts(theme="cinematic")
    run_auto(clips, song, output, opts)

    s = captured["score"]
    assert s.transitions == "soft", "cinematic theme should set transitions=soft"
    assert s.audio_mix == "ducked", "cinematic theme should set audio_mix=ducked"
    assert s.lut == "cinematic", "cinematic theme should set lut=cinematic"
    # Pace stays at the theme's value ("medium" — same as baseline).
    assert s.pace == "medium"
    # Speed ramp is False in this theme (matches baseline; still False).
    assert s.no_speed_ramp is False
    # Music DB is the theme's preferred -9 (since baseline is the global default).
    assert s.music_db == -9.0


def test_run_auto_user_flag_beats_theme(tmp_path: Path, monkeypatch):
    """Explicit caller-set knobs must NOT be clobbered by the theme bundle."""
    captured = _patch_pipeline(monkeypatch)

    clips = tmp_path / "clips"
    clips.mkdir()
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    output = tmp_path / "out.mp4"

    # User asked for music_only — cinematic's `audio_mix: ducked` must not win.
    opts = AutoOpts(theme="cinematic", audio_mix="music_only")
    run_auto(clips, song, output, opts)

    s = captured["score"]
    assert s.audio_mix == "music_only", "explicit audio_mix must beat theme"
    # Other untouched theme knobs still apply.
    assert s.transitions == "soft"
    assert s.lut == "cinematic"


def test_run_auto_threads_basic_args_through(tmp_path: Path, monkeypatch):
    """Sanity: no theme, default knobs — analyze/score/render all see the
    expected inputs."""
    captured = _patch_pipeline(monkeypatch)

    clips = tmp_path / "clips"
    clips.mkdir()
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    output = tmp_path / "out.mp4"

    run_auto(clips, song, output, AutoOpts())

    assert Path(captured["analyze"].clips) == clips.resolve()
    assert captured["analyze"].still_duration == 2.5
    assert captured["analyze"].no_stills is False

    s = captured["score"]
    assert Path(s.song) == song.resolve()
    assert s.aspect == "16:9"
    assert s.audio_mix == "ducked"  # the new soft default
    assert s.transitions == "cut"
    assert s.music_db == DEFAULT_MUSIC_DB
    assert s.lut is None  # no theme → caller passed nothing → still None

    assert Path(captured["render"].output) == output.resolve()


# ---- slow path: real ffmpeg, no stubs --------------------------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_run_auto_minimal_inputs(tmp_path: Path, fixtures_dir: Path, tone: Path):
    """run_auto with just clips + song + output produces a valid mp4."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("clip_a.mp4", "clip_b.mp4"):
        shutil.copy(fixtures_dir / name, clips_dir / name)

    output = tmp_path / "minimal.mp4"
    opts = AutoOpts(max_length=4.0, res="320x240", fps=24)
    result = run_auto(clips_dir, tone, output, opts)

    assert result == output.resolve()
    assert output.exists() and output.stat().st_size > 0
