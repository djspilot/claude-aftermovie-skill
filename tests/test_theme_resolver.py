"""Tests for the `ThemeResolver` Module — the single source of truth for
theme bundle expansion shared by `pipeline_runner._apply_theme` and
`effective_config._theme_layer`.
"""
from __future__ import annotations

import pytest

from aftermovie.config import DEFAULT_MUSIC_DB
from aftermovie.themes import (
    BASELINE_DEFAULTS,
    THEMES,
    ThemeResolver,
)


def test_baseline_music_db_matches_config_default():
    """The Module duplicates `DEFAULT_MUSIC_DB` locally to dodge a circular
    import; this guards against the two drifting apart."""
    assert BASELINE_DEFAULTS["music_db"] == DEFAULT_MUSIC_DB


def test_apply_cinematic_over_baseline_yields_lut_and_audio_mix():
    """Theme value wins when every knob is still at the built-in default."""
    current = dict(BASELINE_DEFAULTS)
    out = ThemeResolver.apply("cinematic", current, BASELINE_DEFAULTS)

    assert out["lut"] == "cinematic"
    assert out["audio_mix"] == "ducked"
    assert out["transitions"] == "soft"
    assert out["music_db"] == -9.0
    # Baseline didn't get clobbered.
    assert BASELINE_DEFAULTS["lut"] is None


def test_apply_keeps_user_overridden_values():
    """A field the caller explicitly set must NOT be overwritten by the theme."""
    current = dict(BASELINE_DEFAULTS)
    current["audio_mix"] = "music_only"   # user override
    current["transitions"] = "auto"       # user override
    out = ThemeResolver.apply("cinematic", current, BASELINE_DEFAULTS)

    assert out["audio_mix"] == "music_only", "explicit user value must beat theme"
    assert out["transitions"] == "auto", "explicit user value must beat theme"
    # But fields still at baseline DO get the theme's value.
    assert out["lut"] == "cinematic"
    assert out["music_db"] == -9.0


def test_apply_unknown_theme_is_noop():
    """Unknown theme names are silently ignored — `current_values` come back unchanged.

    (Contract: silent no-op. The user-facing CLI already validates via argparse
    `choices=`, and the env-file / MCP paths prefer 'keep going with defaults'
    over crashing the whole pipeline on a typo.)
    """
    current = dict(BASELINE_DEFAULTS)
    out = ThemeResolver.apply("bogus", current, BASELINE_DEFAULTS)
    assert out == current


def test_apply_empty_theme_name_is_noop():
    current = dict(BASELINE_DEFAULTS)
    assert ThemeResolver.apply(None, current, BASELINE_DEFAULTS) == current
    assert ThemeResolver.apply("", current, BASELINE_DEFAULTS) == current


def test_apply_does_not_leak_description_into_values():
    """`description` is metadata for help text — it must never appear in
    the resolved knob values."""
    current = dict(BASELINE_DEFAULTS)
    out = ThemeResolver.apply("cinematic", current, BASELINE_DEFAULTS)
    assert "description" not in out


def test_apply_skips_keys_not_in_current_values():
    """The caller's value-set is the authority on what's addressable.
    Theme keys absent from `current_values` are dropped — a theme can't
    introduce surprise knobs into the dataclass."""
    current = {"lut": None}   # only one addressable knob
    out = ThemeResolver.apply("cinematic", current, BASELINE_DEFAULTS)
    assert set(out.keys()) == {"lut"}
    assert out["lut"] == "cinematic"


def test_apply_returns_a_fresh_dict():
    """Caller's `current_values` mapping is not mutated."""
    current = dict(BASELINE_DEFAULTS)
    snapshot = dict(current)
    ThemeResolver.apply("cinematic", current, BASELINE_DEFAULTS)
    assert current == snapshot


def test_describe_includes_values_and_description():
    described = ThemeResolver.describe("cinematic")
    assert described["name"] == "cinematic"
    assert "Glide-y" in described["description"]
    # All theme-controlled knobs are present in `values`.
    for k in ("lut", "music_db", "transitions", "audio_mix", "pace", "no_speed_ramp"):
        assert k in described["values"]
    # Description never leaks into values.
    assert "description" not in described["values"]


def test_describe_unknown_theme_returns_empty_dict():
    assert ThemeResolver.describe("bogus") == {}


def test_names_returns_sorted_theme_names():
    assert ThemeResolver.names() == sorted(THEMES.keys())
    assert "cinematic" in ThemeResolver.names()


@pytest.mark.parametrize("theme_name", sorted(THEMES.keys()))
def test_apply_round_trip_for_every_preset(theme_name: str):
    """Every preset, applied over the baseline, must overlay all of its
    own non-meta keys faithfully."""
    current = dict(BASELINE_DEFAULTS)
    out = ThemeResolver.apply(theme_name, current, BASELINE_DEFAULTS)
    preset = THEMES[theme_name]
    for k, v in preset.items():
        if k == "description":
            continue
        if k in BASELINE_DEFAULTS:
            assert out[k] == v, f"{theme_name}.{k} should overlay over baseline"
