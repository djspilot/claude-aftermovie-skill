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


def _ffmpeg_has_encoder(name: str) -> bool:
    """Check if the local ffmpeg ships a given encoder. Used to skip
    VideoToolbox tests on Linux CI."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return any(name in line for line in res.stdout.splitlines())


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
        # Smoke fixture only has 3 clips; allow reuse so the planner can
        # fill ~6s without running out under the default source_cap=1.
        "--source-cap", "3",
    ])
    args.func(args)

    assert out_path.exists(), "output mp4 was not produced"

    info = ffprobe_json(out_path)
    streams = info.get("streams", [])
    codecs = {s.get("codec_name") for s in streams}
    types = {s.get("codec_type") for s in streams}

    # B1: the default encoder is now chip-dependent — h264 (libx264 on Linux
    # CI) or hevc (hevc_videotoolbox on Apple Silicon). Both are valid mp4.
    assert codecs & {"h264", "hevc"}, f"expected h264 or hevc video, got {codecs}"
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


def test_second_render_is_faster_with_prerender_cache(
    tmp_path: Path, fixtures_dir: Path, tone: Path, monkeypatch,
):
    """E1 acceptance: a re-render of the same plan must hit the prerender
    cache on every clip and complete substantially faster than the first
    render. We target ≥3× speedup; in practice the cache-hit path avoids
    ffmpeg entirely so the savings are much larger, but the threshold is
    set conservatively so transient I/O noise on CI doesn't flake."""
    import time as _time

    from aftermovie import config

    # Isolate the prerender-cache root inside the test's tmp tree so this
    # test never depends on (or pollutes) the developer's real cache.
    fake_data = tmp_path / "state"
    fake_data.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake_data)

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("clip_a.mp4", "clip_b.mp4", "clip_c.mp4"):
        shutil.copy(fixtures_dir / name, clips_dir / name)

    parser = build_parser()

    def _render(out_name: str) -> float:
        out = tmp_path / out_name
        args = parser.parse_args([
            "auto",
            "--clips", str(clips_dir),
            "--song", str(tone),
            "--output", str(out),
            "--max-length", "6",
            "--res", "320x240",
            "--fps", "24",
            "--source-cap", "3",
        ])
        t0 = _time.perf_counter()
        args.func(args)
        elapsed = _time.perf_counter() - t0
        assert out.exists(), f"{out_name} not produced"
        return elapsed

    t_cold = _render("cold.mp4")
    t_warm = _render("warm.mp4")

    speedup = t_cold / max(t_warm, 1e-3)
    assert speedup >= 3.0, (
        f"expected ≥3× speedup from prerender cache, got "
        f"{speedup:.2f}× (cold={t_cold:.2f}s warm={t_warm:.2f}s)"
    )


@pytest.mark.skipif(
    not _ffmpeg_has_encoder("h264_videotoolbox"),
    reason="h264_videotoolbox not available (likely non-Apple-Silicon host)",
)
def test_render_with_h264_vt_env_override(
    tmp_path: Path, fixtures_dir: Path, tone: Path, monkeypatch,
):
    """A render with AFTERMOVIE_VIDEO_CODEC=h264_vt produces a valid mp4."""
    monkeypatch.setenv("AFTERMOVIE_VIDEO_CODEC", "h264_vt")

    out_path = tmp_path / "smoke_vt.mp4"
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
        "--source-cap", "3",
    ])
    args.func(args)

    assert out_path.exists(), "VT-encoded mp4 was not produced"
    info = ffprobe_json(out_path)
    codecs = {s.get("codec_name") for s in info.get("streams", [])}
    # h264_videotoolbox produces an h264 stream, just like libx264.
    assert "h264" in codecs, f"expected h264, got {codecs}"
