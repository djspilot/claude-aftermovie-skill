"""End-to-end smoke for Phase 3: transitions + titles + ducked audio."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from aftermovie import config
from aftermovie.analyze.clip import cmd_analyze
from aftermovie.cli import build_parser
from aftermovie.ffmpeg_cmd import ffprobe_json

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def test_analyze_twice_hits_per_clip_cache(
    tmp_path: Path, fixtures_dir: Path, monkeypatch, capsys
) -> None:
    """Per-clip cache layer: running cmd_analyze twice on the same folder
    should log `analyze: N cached, 0 to compute` the second time.

    This guards the case the folder-level catalog cache can't: when one
    clip is touched (or a new file lands) the folder cache is busted
    but every untouched clip should still skip its analyzers."""
    # Redirect data_dir into tmp so the per-clip cache lives in isolation.
    fake = tmp_path / "skills-data"
    fake.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake)

    # Stage 3 fixtures into a fresh folder so analyze sees them all.
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("clip_a.mp4", "clip_b.mp4", "clip_c.mp4"):
        shutil.copy(fixtures_dir / name, clips_dir / name)

    out_path = tmp_path / "catalog.json"
    args = argparse.Namespace(
        clips=str(clips_dir),
        out=str(out_path),
        still_duration=2.5,
        no_stills=False,
    )

    # First pass: cold cache → every clip computed.
    # Force AFTERMOVIE_ANALYZE_WORKERS=1 so the test stays deterministic
    # and ProcessPoolExecutor spawn overhead doesn't dominate fixture timing.
    monkeypatch.setenv("AFTERMOVIE_ANALYZE_WORKERS", "1")
    cmd_analyze(args)
    cold_out = capsys.readouterr().err
    assert "0 cached, 3 to compute" in cold_out, (
        f"first pass should be a full miss; got:\n{cold_out}"
    )
    body = json.loads(out_path.read_text())
    assert len(body["clips"]) == 3

    # Second pass: warm cache → every clip hits.
    cmd_analyze(args)
    warm_out = capsys.readouterr().err
    assert "3 cached, 0 to compute" in warm_out, (
        f"second pass should be all hits; got:\n{warm_out}"
    )
    body2 = json.loads(out_path.read_text())
    assert len(body2["clips"]) == 3


def test_auto_with_transitions_and_titles_and_ducked_audio(
    tmp_path: Path, fixtures_dir: Path, tone: Path
):
    out_path = tmp_path / "phase3.mp4"
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("clip_a.mp4", "clip_b.mp4", "clip_c.mp4"):
        shutil.copy(fixtures_dir / name, clips_dir / name)

    parser = build_parser()
    args = parser.parse_args([
        "auto",
        "--clips", str(clips_dir),
        "--song", str(tone),
        "--output", str(out_path),
        "--max-length", "4",
        "--res", "320x240",
        "--fps", "24",
        "--transitions", "auto",
        "--titles", "intro,outro",
        "--title-text", "Test",
        "--audio-mix", "ducked",
    ])
    args.func(args)

    assert out_path.exists()
    info = ffprobe_json(out_path)
    duration = float(info["format"]["duration"])
    # Should be close to song length, with small slack for rounding.
    assert 2.5 < duration < 5.0, f"unexpected duration {duration}"
    types = {s["codec_type"] for s in info["streams"]}
    assert "video" in types and "audio" in types
