"""Portrait-pair stills: two vertical photos share one side-by-side shot."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aftermovie.analyze.stills import (
    materialize_still_duo,
    pair_portrait_stills,
)


def _png(path: Path, w: int, h: int, color=(200, 80, 40)) -> Path:
    from PIL import Image
    Image.new("RGB", (w, h), color=color).save(path)
    return path


def _stamp(path: Path, ts: float) -> None:
    os.utime(path, (ts, ts))


def test_pair_portrait_stills_pairs_adjacent_verticals(tmp_path: Path):
    """Two consecutive portraits within the window pair; a landscape in
    between breaks adjacency; far-apart portraits stay single."""
    a = _png(tmp_path / "a.png", 400, 800)
    b = _png(tmp_path / "b.png", 400, 800)
    wide = _png(tmp_path / "c_wide.png", 800, 400)
    late = _png(tmp_path / "d_late.png", 400, 800)
    _stamp(a, 1000.0)
    _stamp(b, 1030.0)          # 30s after a → pairs with a
    _stamp(wide, 1060.0)       # landscape → single
    _stamp(late, 999_999.0)    # portrait but hours later → single

    pairs, singles = pair_portrait_stills([wide, late, b, a])
    assert pairs == [(a, b)]
    assert set(singles) == {wide, late}


def test_pair_window_blocks_distant_portraits(tmp_path: Path):
    a = _png(tmp_path / "a.png", 400, 800)
    b = _png(tmp_path / "b.png", 400, 800)
    _stamp(a, 1000.0)
    _stamp(b, 1000.0 + 601)  # just past DUO_PAIR_WINDOW_S
    pairs, singles = pair_portrait_stills([a, b])
    assert pairs == []
    assert singles == [a, b]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_materialize_still_duo_renders_landscape_mp4(tmp_path: Path, monkeypatch):
    from aftermovie.analyze import stills as stills_mod
    monkeypatch.setattr(stills_mod, "_stills_cache_dir", lambda: tmp_path)

    a = _png(tmp_path / "left.png", 400, 800, color=(255, 0, 0))
    b = _png(tmp_path / "right.png", 400, 800, color=(0, 0, 255))
    out = materialize_still_duo(a, b, duration_s=1.0, target_res="640x360")
    assert out is not None and out.is_file()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe == "640,360"
