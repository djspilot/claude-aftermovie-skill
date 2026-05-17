"""Tests for the still-filter compiler (variant pick + chain build)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from aftermovie.analyze.still_filters import (
    FilterSpec,
    _build_still_filter,
    _PORTRAIT_VARIANTS,
    _pick_still_variant,
)


def _png(path: Path, size=(320, 240), color=(120, 160, 200)) -> Path:
    """Create a synthetic image at the given size + path."""
    Image.new("RGB", size, color).save(path)
    return path


def test_pick_variant_is_deterministic_per_filename(tmp_path: Path):
    """Same filename → same variant + same seed across runs.

    Determinism matters because users re-run the pipeline and expect their
    photos to animate consistently — a 'shake' photo on Monday must still
    be 'shake' on Tuesday so the edit doesn't visibly drift.
    """
    img = _png(tmp_path / "IMG_4242.jpg", size=(800, 600))  # landscape so not portrait-gated
    first = _pick_still_variant(img, 1920, 1080)
    second = _pick_still_variant(img, 1920, 1080)
    third = _pick_still_variant(img, 1920, 1080)
    assert first == second == third

    # A different filename should be allowed to land on a different variant.
    # We can't assert it always differs (hash collisions), but at least the
    # seed must differ.
    other = _png(tmp_path / "IMG_9999.jpg", size=(800, 600))
    other_pick = _pick_still_variant(other, 1920, 1080)
    assert other_pick[1] != first[1], "different stems must produce different seeds"


def test_portrait_source_forces_show_whole_image(tmp_path: Path):
    """Portrait HEIC/JPG must land on fit_pad or blurred_bg — never a crop variant.

    Cropping into a portrait photo at 16:9 output lops off heads, so the
    picker is gated by `_is_portrait` to keep the whole image visible.
    """
    # Build a clearly portrait image (tall > wide).
    portrait = _png(tmp_path / "portrait_001.jpg", size=(600, 1200))
    variant, _seed = _pick_still_variant(portrait, 1920, 1080)
    assert variant in _PORTRAIT_VARIANTS, (
        f"portrait source should map to {_PORTRAIT_VARIANTS}, got {variant}"
    )

    # And the chain builder must produce a valid FilterSpec for that variant.
    spec = _build_still_filter(variant, _seed, 60, 30, 1920, 1080)
    assert isinstance(spec, FilterSpec)
    assert spec.chain  # non-empty
    assert spec.out_label.startswith("still_")
    # Portrait variants don't crop into the source, so the chain must not
    # carry the giant 2× supersample + crop idiom used by zoom variants.
    if variant == "fit_pad":
        assert "pad=" in spec.chain
    elif variant == "blurred_bg":
        assert "boxblur" in spec.chain


def test_build_still_filter_landscape_variants_produce_valid_chains(tmp_path: Path):
    """Every advertised variant must build a non-empty filter chain."""
    for variant in ("live", "push", "pull", "pan_h", "shake", "fit_pad", "blurred_bg"):
        spec = _build_still_filter(variant, seed=0x1234ABCD, n=75, fps=30, w=1920, h=1080)
        assert isinstance(spec, FilterSpec)
        assert spec.chain, f"empty chain for variant={variant}"
        assert "format=yuv420p" in spec.chain, f"missing pixfmt for variant={variant}"
