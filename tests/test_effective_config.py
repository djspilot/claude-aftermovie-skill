"""Tests for the EffectiveConfig precedence chain
(built-in default -> env file -> theme bundle -> CLI override).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aftermovie.config import THEMES
from aftermovie.effective_config import (
    BUILTIN_DEFAULTS,
    EffectiveConfig,
    resolve,
)


@pytest.fixture(autouse=True)
def _no_env_pollution(monkeypatch):
    """Strip every AFTERMOVIE_* env var so tests start from builtins.

    Without this, a developer's `~/.aftermovie/aftermovie.env` (loaded by
    earlier tests via `load_env_file`) would leak in via `os.environ` and
    make assertions about builtin values flaky.
    """
    for env_key in [
        "AFTERMOVIE_ASPECT", "AFTERMOVIE_RES", "AFTERMOVIE_FPS",
        "AFTERMOVIE_MAX_LENGTH", "AFTERMOVIE_STILL_DURATION",
        "AFTERMOVIE_NO_STILLS", "AFTERMOVIE_AUDIO_MIX", "AFTERMOVIE_MUSIC_DB",
        "AFTERMOVIE_CLIP_DB", "AFTERMOVIE_PACE", "AFTERMOVIE_TRANSITIONS",
        "AFTERMOVIE_NO_SPEED_RAMP", "AFTERMOVIE_NO_REFRAME",
        "AFTERMOVIE_LUT", "AFTERMOVIE_THEME",
        "AFTERMOVIE_AUDIO_INTEREST_THRESHOLD",
    ]:
        monkeypatch.delenv(env_key, raising=False)


def _write_env(tmp_path: Path, body: str) -> Path:
    env_file = tmp_path / "aftermovie.env"
    env_file.write_text(body)
    return env_file


def test_resolve_returns_builtins_when_no_inputs(tmp_path: Path):
    cfg = resolve(env_file=tmp_path / "missing.env")
    assert cfg.aspect == BUILTIN_DEFAULTS["aspect"]
    assert cfg.res == BUILTIN_DEFAULTS["res"]
    assert cfg.fps == BUILTIN_DEFAULTS["fps"]
    assert cfg.audio_mix == BUILTIN_DEFAULTS["audio_mix"]
    assert cfg.pace == BUILTIN_DEFAULTS["pace"]
    assert cfg.transitions == BUILTIN_DEFAULTS["transitions"]
    assert cfg.theme is None


def test_resolve_precedence(tmp_path: Path):
    """CLI override beats theme; theme beats env file; env file beats default."""
    # Env file sets pace=fast (overriding builtin 'medium') and lut='punchy'.
    env_file = _write_env(tmp_path, "AFTERMOVIE_PACE=fast\nAFTERMOVIE_LUT=punchy\n")

    # Sanity: env file alone, no theme, no CLI overrides.
    cfg = resolve(env_file=env_file)
    assert cfg.pace == "fast", "env file should beat builtin default"
    assert cfg.lut == "punchy"

    # Theme 'chill' sets pace=slow and lut=chill — should beat env values.
    cfg = resolve(env_file=env_file, theme="chill")
    assert cfg.pace == "slow", "theme should beat env file"
    assert cfg.lut == "chill", "theme should beat env file"

    # CLI override should beat theme.
    cfg = resolve(
        env_file=env_file,
        theme="chill",
        cli_overrides={"pace": "fast", "lut": "cinematic"},
    )
    assert cfg.pace == "fast", "CLI override should beat theme"
    assert cfg.lut == "cinematic", "CLI override should beat theme"


def test_resolve_theme_does_not_override_explicit_cli(tmp_path: Path):
    cfg = resolve(
        env_file=tmp_path / "missing.env",
        theme="cinematic",  # theme sets audio_mix='ducked'
        cli_overrides={"audio_mix": "music_only"},
    )
    assert cfg.audio_mix == "music_only"
    # Theme name is still preserved.
    assert cfg.theme == "cinematic"


def test_resolve_unknown_theme_is_noop(tmp_path: Path):
    cfg = resolve(env_file=tmp_path / "missing.env", theme="bogus")
    # Doesn't crash and falls back to builtin values for theme-controlled keys.
    assert cfg.audio_mix == BUILTIN_DEFAULTS["audio_mix"]
    assert cfg.pace == BUILTIN_DEFAULTS["pace"]
    assert cfg.transitions == BUILTIN_DEFAULTS["transitions"]
    assert cfg.lut == BUILTIN_DEFAULTS["lut"]
    # But the name is preserved on the resolved config (callers may want to display it).
    assert cfg.theme == "bogus"


def test_resolve_env_file_loaded(tmp_path: Path):
    env_file = _write_env(tmp_path, """
# An env file with a couple of overrides.
AFTERMOVIE_ASPECT=9:16
AFTERMOVIE_FPS=60
AFTERMOVIE_MUSIC_DB=-12.5
AFTERMOVIE_NO_SPEED_RAMP=true
# blank line and inline comment below
AFTERMOVIE_PACE=slow   # comment after value
""")
    cfg = resolve(env_file=env_file)
    assert cfg.aspect == "9:16"
    assert cfg.fps == 60
    assert cfg.music_db == -12.5
    assert cfg.no_speed_ramp is True
    assert cfg.pace == "slow"
    # Unset keys still fall back to builtins.
    assert cfg.res == BUILTIN_DEFAULTS["res"]
    assert cfg.audio_mix == BUILTIN_DEFAULTS["audio_mix"]


def test_resolve_cli_none_means_unset(tmp_path: Path):
    """None values in cli_overrides must NOT clobber lower layers."""
    env_file = _write_env(tmp_path, "AFTERMOVIE_PACE=fast\n")
    cfg = resolve(
        env_file=env_file,
        cli_overrides={"pace": None, "audio_mix": None},
    )
    # pace stays at env-file value, audio_mix stays at builtin.
    assert cfg.pace == "fast"
    assert cfg.audio_mix == BUILTIN_DEFAULTS["audio_mix"]


def test_resolve_theme_from_env_file(tmp_path: Path):
    """A theme picked up from the env file applies its bundle."""
    env_file = _write_env(tmp_path, "AFTERMOVIE_THEME=punchy\n")
    cfg = resolve(env_file=env_file)
    assert cfg.theme == "punchy"
    assert cfg.pace == THEMES["punchy"]["pace"]
    assert cfg.transitions == THEMES["punchy"]["transitions"]


def test_resolve_returns_frozen_dataclass(tmp_path: Path):
    cfg = resolve(env_file=tmp_path / "missing.env")
    assert isinstance(cfg, EffectiveConfig)
    with pytest.raises(Exception):
        cfg.aspect = "1:1"  # type: ignore[misc]


def test_resolve_process_env_beats_env_file(tmp_path: Path, monkeypatch):
    """A one-off `AFTERMOVIE_FOO=x aftermovie ...` should still win."""
    env_file = _write_env(tmp_path, "AFTERMOVIE_PACE=fast\n")
    monkeypatch.setenv("AFTERMOVIE_PACE", "slow")
    cfg = resolve(env_file=env_file)
    assert cfg.pace == "slow"
