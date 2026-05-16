"""End-to-end smoke test: `auto` on synthetic clips + tone produces a valid mp4."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aftermovie.cli import build_parser
from aftermovie.ffmpeg_cmd import ffprobe_json


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def test_auto_produces_valid_mp4(tmp_path: Path, fixtures_dir: Path, tone: Path):
    out_path = tmp_path / "smoke.mp4"
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
        "--max-length", "6",
        "--res", "320x240",
        "--fps", "24",
    ])
    args.func(args)

    assert out_path.exists(), "output mp4 was not produced"

    info = ffprobe_json(out_path)
    streams = info.get("streams", [])
    codecs = {s.get("codec_name") for s in streams}
    types = {s.get("codec_type") for s in streams}

    assert "h264" in codecs, f"expected h264 video, got {codecs}"
    assert "aac" in codecs, f"expected aac audio, got {codecs}"
    assert "video" in types and "audio" in types

    duration = float(info["format"]["duration"])
    # tone.wav is 10s, --max-length 6, so duration should be roughly 6s.
    assert 4.0 < duration < 8.0, f"unexpected duration {duration}"


def test_score_then_render_roundtrip(tmp_path: Path, fixtures_dir: Path, tone: Path):
    """analyze → score → render staged workflow produces an mp4 too."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("clip_a.mp4", "clip_b.mp4"):
        shutil.copy(fixtures_dir / name, clips_dir / name)

    catalog = tmp_path / "catalog.json"
    plan = tmp_path / "plan.json"
    output = tmp_path / "staged.mp4"

    parser = build_parser()

    args = parser.parse_args([
        "analyze", "--clips", str(clips_dir), "--out", str(catalog),
    ])
    args.func(args)
    assert catalog.exists()
    cat_data = json.loads(catalog.read_text())
    assert len(cat_data["clips"]) == 2

    args = parser.parse_args([
        "score",
        "--catalog", str(catalog),
        "--song", str(tone),
        "--out", str(plan),
        "--max-length", "5",
    ])
    args.func(args)
    plan_data = json.loads(plan.read_text())
    assert "entries" in plan_data
    assert plan_data["target_length_s"] <= 5.0 + 0.01

    args = parser.parse_args([
        "render", "--plan", str(plan), "--output", str(output),
    ])
    args.func(args)
    assert output.exists()
