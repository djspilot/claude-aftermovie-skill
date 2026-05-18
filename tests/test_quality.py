"""Sharpness + exposure tests.

We sidestep the ffmpeg/cv2 file-IO path by monkeypatching the internal
`_sampled_grayscale_frames` generator. Each test feeds a synthetic numpy
frame and asserts the output curves match the visual intent (edges =>
high sharpness, uniform => low; black/white => extreme exposure,
mid-grey => mid).
"""
from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np  # noqa: E402

from aftermovie.analyze import quality  # noqa: E402


def _checkerboard(size: int = 64, square: int = 4) -> np.ndarray:
    """High-frequency edges → high Laplacian variance."""
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if ((x // square) + (y // square)) % 2 == 0:
                img[y : y + square, x : x + square] = 255
    return img


def _uniform(size: int = 64, value: int = 128) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def _patch_frames(monkeypatch, frames):
    """Replace the cv2 frame generator with a fixed list of (sec, grey) tuples."""
    def fake_gen(path, duration, fps):
        yield from enumerate(frames)

    monkeypatch.setattr(quality, "_sampled_grayscale_frames", fake_gen)


def test_sharpness_high_on_edges_low_on_uniform(monkeypatch):
    sharp = _checkerboard()
    flat = _uniform(value=128)
    _patch_frames(monkeypatch, [sharp, flat])

    out = quality.sharpness_per_second(Path("/dummy"), 2.0, 30.0)
    assert len(out) == 2
    # After min-max within the clip, the textured frame is 1 and the flat 0.
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)


def test_sharpness_flat_clip_returns_mid_value(monkeypatch):
    flat = _uniform(value=128)
    _patch_frames(monkeypatch, [flat, flat, flat])

    out = quality.sharpness_per_second(Path("/dummy"), 3.0, 30.0)
    # Identical Laplacian variance → no range to normalize → 0.5 each.
    assert out == [0.5, 0.5, 0.5]


def test_sharpness_empty_when_no_frames(monkeypatch):
    _patch_frames(monkeypatch, [])
    out = quality.sharpness_per_second(Path("/dummy"), 1.0, 30.0)
    assert out == []


def test_exposure_reports_raw_luminance(monkeypatch):
    dark = _uniform(value=10)     # ~0.04
    mid = _uniform(value=128)     # ~0.50
    bright = _uniform(value=245)  # ~0.96
    _patch_frames(monkeypatch, [dark, mid, bright])

    out = quality.exposure_per_second(Path("/dummy"), 3.0, 30.0)
    assert len(out) == 3
    assert out[0] < 0.1
    assert 0.45 < out[1] < 0.55
    assert out[2] > 0.9


def test_exposure_extremes_would_trip_scorer_thresholds(monkeypatch):
    """Sanity-check that the analyzer's output is in the units the scorer expects."""
    very_dark = _uniform(value=20)    # 20/255 ≈ 0.078 → < 0.25
    very_bright = _uniform(value=240) # 240/255 ≈ 0.941 → > 0.85
    _patch_frames(monkeypatch, [very_dark, very_bright])
    out = quality.exposure_per_second(Path("/dummy"), 2.0, 30.0)
    assert out[0] < 0.25
    assert out[1] > 0.85


def test_cv2_unavailable_returns_empty(monkeypatch):
    """When cv2 import failed we must short-circuit, not crash."""
    # Pretend cv2 didn't import by flipping the shared OptionalImport handle
    # to its missing-dep state.
    monkeypatch.setattr(quality._CV2, "module", None)
    monkeypatch.setattr(quality._CV2, "_warned", False)
    assert quality.sharpness_per_second(Path("/x"), 1.0, 30.0) == []
    assert quality.exposure_per_second(Path("/x"), 1.0, 30.0) == []
    assert quality.sharpness_for_image(Path("/x")) is None
    assert quality.exposure_for_image(Path("/x")) is None


def test_image_helpers_handle_unreadable_file(tmp_path: Path):
    bogus = tmp_path / "not_an_image.heic"
    bogus.write_bytes(b"\x00\x01\x02")
    # cv2.imread returns None for unreadable files; both helpers must
    # surface that as None rather than raising.
    assert quality.sharpness_for_image(bogus) is None
    assert quality.exposure_for_image(bogus) is None
