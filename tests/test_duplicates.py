"""Perceptual-hash + grouping tests for analyze/duplicates.py.

These run without ffmpeg by exercising the still-image path (PNG → PIL →
dHash) and feeding `group_duplicates` synthetic (path, phash) tuples, so
they're fast and CI-portable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aftermovie.analyze.duplicates import (
    compute_phash,
    group_duplicates,
    hamming_distance,
)


def _solid_png(path: Path, value: int, size: int = 32) -> None:
    """Write a solid-color grayscale PNG (no PIL feature deps beyond core)."""
    from PIL import Image
    Image.new("L", (size, size), color=value).save(path)


def _gradient_png(path: Path, size: int = 32, flip: bool = False) -> None:
    """Write a horizontal gradient PNG. `flip` mirrors it left↔right so the
    dHash bit pattern is the inverse — perfect for the 'very different'
    case."""
    from PIL import Image
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            v = int(255 * (x / max(1, size - 1)))
            if flip:
                v = 255 - v
            px[x, y] = v
    img.save(path)


def test_identical_images_have_zero_distance(tmp_path: Path):
    """Two byte-identical stills → identical phash → distance 0."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _gradient_png(a)
    _gradient_png(b)  # same recipe, same pixels
    ha = compute_phash(a)
    hb = compute_phash(b)
    assert ha is not None and hb is not None
    assert ha == hb
    assert hamming_distance(ha, hb) == 0


def test_identical_images_group_together(tmp_path: Path):
    """group_duplicates puts byte-identical phashes in one cluster."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _gradient_png(a)
    _gradient_png(b)
    ha = compute_phash(a)
    hb = compute_phash(b)
    groups = group_duplicates([(str(a), ha), (str(b), hb)])
    assert groups[str(a)] is not None
    assert groups[str(a)] == groups[str(b)]


def test_very_different_images_not_grouped(tmp_path: Path):
    """Inverted gradient → dHash bits flip → distance >> threshold."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _gradient_png(a, flip=False)
    _gradient_png(b, flip=True)
    ha = compute_phash(a)
    hb = compute_phash(b)
    assert ha is not None and hb is not None
    # The hashes should differ by many bits — definitely above the default
    # threshold of 8.
    assert hamming_distance(ha, hb) > 8
    groups = group_duplicates([(str(a), ha), (str(b), hb)])
    # Both are singletons → both map to None.
    assert groups[str(a)] is None
    assert groups[str(b)] is None


def test_group_duplicates_chains_transitive(tmp_path: Path):
    """If A≈B and B≈C but A and C are 9 bits apart, all three still cluster.

    Hamming distance isn't transitive, but visual-duplicate grouping needs
    transitive closure — otherwise a slow drift across a burst of frames
    leaves us with stranded singletons that the scorer can't collapse.
    """
    # Construct three phashes by hand so we can control the distances:
    #   A = all zeros (0 ones)
    #   B = 7 ones in the top 7 bits (overlaps zero of A → dist=7)
    #   C = B's 7 ones PLUS 7 more in bits 8..14 (dist to B = 7, dist to A = 14)
    a = "0000000000000000"
    b = "fe00000000000000"
    c = "fefe000000000000"
    da = hamming_distance(a, b)
    dbc = hamming_distance(b, c)
    dac = hamming_distance(a, c)
    # Sanity: chain holds, direct skip doesn't.
    assert da <= 8 and dbc <= 8, (da, dbc)
    assert dac > 8, dac

    groups = group_duplicates(
        [("/a.png", a), ("/b.png", b), ("/c.png", c)],
        threshold=8,
    )
    # All three must end up in the same cluster despite A↔C exceeding the
    # threshold — that's the transitive-closure guarantee.
    assert groups["/a.png"] is not None
    assert groups["/a.png"] == groups["/b.png"] == groups["/c.png"]


def test_group_duplicates_handles_none_phash():
    """Items without a phash always map to None and never join a cluster."""
    groups = group_duplicates(
        [
            ("/has.png", "abcd1234abcd1234"),
            ("/missing.mp4", None),
            ("/twin.png", "abcd1234abcd1234"),
        ]
    )
    assert groups["/missing.mp4"] is None
    assert groups["/has.png"] is not None
    assert groups["/has.png"] == groups["/twin.png"]


def test_group_duplicates_singleton_returns_none():
    """A unique hash with no near-twins is NOT given a group id."""
    groups = group_duplicates(
        [
            ("/alone.png", "0000000000000000"),
            ("/other.png", "ffffffffffffffff"),
        ]
    )
    assert groups["/alone.png"] is None
    assert groups["/other.png"] is None


def test_hamming_distance_handles_length_mismatch():
    """Defensive: a legacy / corrupt phash with the wrong length must NOT
    crash the grouping pass — it just gets the max possible distance."""
    assert hamming_distance("abcd", "abcdef") == 64
    assert hamming_distance("zzzz", "0000") == 64  # invalid hex → 64


def test_compute_phash_returns_none_on_unreadable_file(tmp_path: Path):
    """Missing / corrupt files must return None instead of raising."""
    bad = tmp_path / "nope.png"
    bad.write_bytes(b"not a png")
    assert compute_phash(bad) is None


@pytest.mark.parametrize("value", [0, 64, 200, 255])
def test_compute_phash_returns_16_hex_chars(tmp_path: Path, value: int):
    """The hash format contract: 16 lowercase hex chars (64 bits)."""
    p = tmp_path / f"{value}.png"
    _solid_png(p, value)
    h = compute_phash(p)
    assert h is not None
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
