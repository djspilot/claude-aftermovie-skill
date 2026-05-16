"""End-to-end smoke for Phase 3: transitions + titles + ducked audio."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aftermovie.cli import build_parser
from aftermovie.ffmpeg_cmd import ffprobe_json

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


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
