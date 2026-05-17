"""Tests for still-photo + Live Photo pairing logic."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from aftermovie.analyze.stills import (
    STILL_EXTS,
    find_stills_excluding_live_pairs,
    materialize_still,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def _png(path: Path, color=(200, 50, 50)) -> Path:
    Image.new("RGB", (320, 240), color).save(path)
    return path


def test_standalone_stills_returned(tmp_path: Path):
    _png(tmp_path / "a.jpg")
    (tmp_path / "b.heic").write_bytes(b"")  # extension-only check; PIL can't save .heic
    _png(tmp_path / "c.png")
    stills = find_stills_excluding_live_pairs(tmp_path)
    names = {p.name for p in stills}
    assert names == {"a.jpg", "b.heic", "c.png"}


def test_live_photo_pair_drops_the_still(tmp_path: Path):
    _png(tmp_path / "IMG_0488.jpg")
    (tmp_path / "IMG_0488.mov").write_bytes(b"\x00" * 16)
    _png(tmp_path / "lone.png")
    stills = find_stills_excluding_live_pairs(tmp_path)
    names = {p.name for p in stills}
    assert names == {"lone.png"}, f"expected only lone.png, got {names}"


def test_dotfile_stills_ignored(tmp_path: Path):
    _png(tmp_path / ".DS_Store_thumbnail.jpg")
    _png(tmp_path / "visible.png")
    stills = find_stills_excluding_live_pairs(tmp_path)
    names = {p.name for p in stills}
    assert names == {"visible.png"}


def test_materialize_produces_valid_mp4(tmp_path: Path):
    src = tmp_path / "photo.jpg"
    _png(src)
    out = materialize_still(src, duration_s=1.5, target_res="320x180")
    assert out is not None
    assert out.is_file()
    assert out.suffix == ".mp4"
    from aftermovie.ffmpeg_cmd import ffprobe_json
    info = ffprobe_json(out)
    duration = float(info["format"]["duration"])
    assert 1.3 < duration < 1.7
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v["width"] == 320 and v["height"] == 180


def test_materialize_is_cached(tmp_path: Path):
    src = tmp_path / "photo.jpg"
    _png(src)
    first = materialize_still(src, duration_s=1.0, target_res="320x180")
    second = materialize_still(src, duration_s=1.0, target_res="320x180")
    assert first == second
