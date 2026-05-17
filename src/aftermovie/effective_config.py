"""Single source of truth for the precedence chain
`built-in default -> env file -> theme bundle -> CLI flag`.

Both the CLI (`aftermovie auto/score`) and the MCP server resolve their
runtime config through `EffectiveConfig.resolve(...)` so the two surfaces
cannot silently diverge.

Order of layers (each later layer wins over earlier ones):

1. Built-in defaults baked into `BUILTIN_DEFAULTS`.
2. Env-file values from `~/.aftermovie/aftermovie.env`
   (or `$AFTERMOVIE_CONFIG_FILE`).
3. Theme bundle from `THEMES[theme]` in `config.py`.
4. Explicit CLI overrides (anything not-None in `cli_overrides`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from aftermovie.config import (
    DEFAULT_CLIP_DB,
    DEFAULT_FPS,
    DEFAULT_MUSIC_DB,
    DEFAULT_RES,
    THEMES,
)
from aftermovie.env_config import config_path


# ---- built-in defaults -----------------------------------------------------

# These are the values the program uses when neither env nor theme nor CLI
# overrides specify anything. Kept here (and only here) so the precedence
# chain has a single, declarative bottom layer.
BUILTIN_DEFAULTS: dict[str, Any] = {
    "aspect": "16:9",
    "res": DEFAULT_RES,
    "fps": DEFAULT_FPS,
    "max_length": None,
    "still_duration": 2.5,
    "no_stills": False,
    "audio_mix": "ducked",
    "music_db": DEFAULT_MUSIC_DB,
    "clip_db": DEFAULT_CLIP_DB,
    "pace": "medium",
    "transitions": "cut",
    "no_speed_ramp": False,
    "no_reframe": False,
    "lut": None,
    "theme": None,
    "audio_interest_threshold": 0.35,
}


# Mapping of field name -> env file key + parser. The parser converts the
# raw string from the env file into the right Python type and returns None
# if parsing fails (so a malformed value falls through to the next layer).

def _parse_str(v: str) -> str | None:
    return v if v != "" else None


def _parse_int(v: str) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_bool(v: str) -> bool | None:
    if v == "":
        return None
    return v.strip().lower() in ("1", "true", "yes", "on")


ENV_KEYS: dict[str, tuple[str, Any]] = {
    "aspect":                   ("AFTERMOVIE_ASPECT", _parse_str),
    "res":                      ("AFTERMOVIE_RES", _parse_str),
    "fps":                      ("AFTERMOVIE_FPS", _parse_int),
    "max_length":               ("AFTERMOVIE_MAX_LENGTH", _parse_float),
    "still_duration":           ("AFTERMOVIE_STILL_DURATION", _parse_float),
    "no_stills":                ("AFTERMOVIE_NO_STILLS", _parse_bool),
    "audio_mix":                ("AFTERMOVIE_AUDIO_MIX", _parse_str),
    "music_db":                 ("AFTERMOVIE_MUSIC_DB", _parse_float),
    "clip_db":                  ("AFTERMOVIE_CLIP_DB", _parse_float),
    "pace":                     ("AFTERMOVIE_PACE", _parse_str),
    "transitions":              ("AFTERMOVIE_TRANSITIONS", _parse_str),
    "no_speed_ramp":            ("AFTERMOVIE_NO_SPEED_RAMP", _parse_bool),
    "no_reframe":               ("AFTERMOVIE_NO_REFRAME", _parse_bool),
    "lut":                      ("AFTERMOVIE_LUT", _parse_str),
    "theme":                    ("AFTERMOVIE_THEME", _parse_str),
    "audio_interest_threshold": ("AFTERMOVIE_AUDIO_INTEREST_THRESHOLD",
                                 _parse_float),
}


@dataclass(frozen=True)
class EffectiveConfig:
    """All knobs that can be set via env file / theme / CLI, after resolution."""
    aspect: str
    res: str
    fps: int
    max_length: float | None
    still_duration: float
    no_stills: bool
    audio_mix: str
    music_db: float
    clip_db: float
    pace: str
    transitions: str
    no_speed_ramp: bool
    no_reframe: bool
    lut: str | None
    theme: str | None
    audio_interest_threshold: float


def _read_env_file(path: Path | None) -> dict[str, str]:
    """Parse a KEY=VALUE env file into a dict. Returns {} if path doesn't exist.

    This is a pure parser — it does NOT mutate os.environ. Resolution composes
    layers explicitly so call order can't change the answer.
    """
    p = path if path is not None else config_path()
    out: dict[str, str] = {}
    if not p.is_file():
        return out
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        if "#" in val:
            val = val.split("#", 1)[0].rstrip()
        out[key] = val
    return out


def _env_layer(env_pairs: dict[str, str]) -> dict[str, Any]:
    """Convert a raw env-file dict + os.environ overrides into typed values.

    Process env (`os.environ`) wins over the file so `KEY=... aftermovie ...`
    one-off invocations still take effect.
    """
    typed: dict[str, Any] = {}
    for field_name, (env_key, parser) in ENV_KEYS.items():
        raw = os.environ.get(env_key)
        if raw is None:
            raw = env_pairs.get(env_key)
        if raw is None:
            continue
        parsed = parser(raw)
        if parsed is not None:
            typed[field_name] = parsed
    return typed


def _theme_layer(theme: str | None) -> dict[str, Any]:
    """Pluck the relevant fields from a theme preset; unknown themes are no-ops."""
    if not theme:
        return {}
    preset = THEMES.get(theme)
    if not preset:
        return {}
    out: dict[str, Any] = {}
    for k, v in preset.items():
        if k == "description":
            continue
        # Only project values that EffectiveConfig actually knows about.
        if k in BUILTIN_DEFAULTS:
            out[k] = v
    return out


def _cli_layer(cli_overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Strip None entries — None means 'not set, fall through to lower layer'."""
    if not cli_overrides:
        return {}
    return {k: v for k, v in cli_overrides.items() if v is not None}


def resolve(
    cli_overrides: dict[str, Any] | None = None,
    *,
    theme: str | None = None,
    env_file: Path | None = None,
) -> EffectiveConfig:
    """Compose builtin -> env file -> theme bundle -> CLI overrides.

    Each later source wins. `None` (or absent) in `cli_overrides` means
    'not set', so the next layer down provides the value.

    `theme` is taken from `cli_overrides["theme"]` if not passed explicitly,
    then falls back to the env file. Themes that aren't in `THEMES` are a
    silent no-op (we still keep the theme name on the resolved config so
    callers can display it).
    """
    env_pairs = _read_env_file(env_file)
    env_values = _env_layer(env_pairs)

    # Theme selection: explicit kwarg > CLI override > env file > builtin.
    chosen_theme = theme
    if chosen_theme is None and cli_overrides is not None:
        chosen_theme = cli_overrides.get("theme")
    if chosen_theme is None:
        chosen_theme = env_values.get("theme")

    layers = [
        BUILTIN_DEFAULTS,
        env_values,
        _theme_layer(chosen_theme),
        _cli_layer(cli_overrides),
    ]
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer)

    # Make sure the chosen theme survives even if no layer wrote it.
    if chosen_theme is not None:
        merged["theme"] = chosen_theme

    # Build the dataclass from the merged dict, ignoring keys it doesn't know.
    known = {f.name for f in fields(EffectiveConfig)}
    return EffectiveConfig(**{k: merged[k] for k in known})
