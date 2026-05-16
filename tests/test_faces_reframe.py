"""Face detection + vertical reframe tests."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aftermovie.analyze.faces import available as faces_available
from aftermovie.cli import build_parser
from aftermovie.ffmpeg_cmd import ffprobe_json
from aftermovie.render.reframe import crop_x_expr_for_entry


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def test_reframe_returns_none_without_faces():
    entry = {"face_bboxes": [None, None, None]}
    assert crop_x_expr_for_entry(entry, 1920, 1080, 1080, 1920) is None


def test_reframe_with_face_data_emits_crop_filter():
    entry = {
        "face_bboxes": [
            {"cx": 0.5, "cy": 0.5},
            {"cx": 0.6, "cy": 0.5},
            {"cx": 0.7, "cy": 0.5},
        ],
    }
    f = crop_x_expr_for_entry(entry, 1920, 1080, 1080, 1920)
    assert f is not None
    assert f.startswith("crop=")
    # The crop width should be source_h * (1080/1920) ≈ 607
    assert "crop=607" in f or "crop=608" in f or "crop=606" in f


def test_reframe_clamps_to_frame_bounds():
    """A face pinned near the left edge should not produce negative x."""
    entry = {"face_bboxes": [{"cx": 0.05, "cy": 0.5}, {"cx": 0.05, "cy": 0.5}]}
    f = crop_x_expr_for_entry(entry, 1920, 1080, 1080, 1920)
    assert f is not None
    # No negative numbers in the expression.
    assert "-" not in f.replace("cx", "").replace("cy", "")


@pytest.mark.skipif(not faces_available(), reason="mediapipe / face model missing")
def test_face_sample_yields_at_least_one_detection(fixtures_dir: Path):
    """The synthetic face fixture should produce at least one detection."""
    from aftermovie.analyze.faces import detect_per_second
    sample = fixtures_dir / "face_sample.mp4"
    if not sample.is_file():
        pytest.skip("face_sample.mp4 fixture not present")
    results = detect_per_second(sample, 5)
    assert any(r is not None for r in results), "no faces detected on synthetic fixture"


@pytest.mark.skipif(not faces_available(), reason="mediapipe / face model missing")
def test_vertical_render_uses_9x16_dimensions(tmp_path: Path, fixtures_dir: Path,
                                              tone: Path):
    sample = fixtures_dir / "face_sample.mp4"
    if not sample.is_file():
        pytest.skip("face_sample.mp4 fixture not present")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    shutil.copy(sample, clips_dir / "face_sample.mp4")

    out = tmp_path / "vertical.mp4"
    parser = build_parser()
    args = parser.parse_args([
        "auto",
        "--clips", str(clips_dir),
        "--song", str(tone),
        "--output", str(out),
        "--max-length", "3",
        "--aspect", "9:16",
        "--fps", "24",
    ])
    args.func(args)
    assert out.exists()
    info = ffprobe_json(out)
    for s in info["streams"]:
        if s["codec_type"] == "video":
            assert s["width"] < s["height"], f"expected vertical, got {s['width']}x{s['height']}"
            break
