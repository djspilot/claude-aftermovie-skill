"""User-editable defaults via an env file.

Reads `~/.aftermovie/aftermovie.env` (or the path in `$AFTERMOVIE_CONFIG_FILE`)
into the process environment so CLI flags can default to user-chosen values.

Format: `KEY=VALUE`, `#` comments, blank lines ignored. Values are not
shell-expanded. Existing environment variables win over the file so one-off
overrides via `KEY=... aftermovie ...` still work.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".aftermovie" / "aftermovie.env"


def config_path() -> Path:
    override = os.environ.get("AFTERMOVIE_CONFIG_FILE")
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from the env file into os.environ.

    Existing env vars are not overwritten. Returns the dict of parsed pairs
    (after the merge, so callers can introspect what was loaded).
    """
    p = path or config_path()
    loaded: dict[str, str] = {}
    if not p.is_file():
        return loaded
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip a single layer of surrounding quotes if present.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        # Strip trailing inline comment (only when unquoted).
        if "#" in val:
            val = val.split("#", 1)[0].rstrip()
        loaded[key] = val
        os.environ.setdefault(key, val)
    return loaded


def env_str(key: str, fallback: str | None) -> str | None:
    v = os.environ.get(key)
    return fallback if v is None or v == "" else v


def env_float(key: str, fallback: float | None) -> float | None:
    v = os.environ.get(key)
    if v is None or v == "":
        return fallback
    try:
        return float(v)
    except ValueError:
        return fallback


def env_int(key: str, fallback: int | None) -> int | None:
    v = os.environ.get(key)
    if v is None or v == "":
        return fallback
    try:
        return int(v)
    except ValueError:
        return fallback


def env_bool(key: str, fallback: bool) -> bool:
    v = os.environ.get(key)
    if v is None or v == "":
        return fallback
    return v.strip().lower() in ("1", "true", "yes", "on")


DEFAULT_CONFIG_TEMPLATE = """# ~/.aftermovie/aftermovie.env
# Defaults for `aftermovie auto`. Flags on the command line override these.
# Comment out a line (or leave value blank) to fall back to the built-in default.
# After editing, just re-run `aftermovie auto ...` — no restart needed.

# ---- Core look ----
AFTERMOVIE_THEME=cinematic            # cinematic | punchy | chill | nostalgic
AFTERMOVIE_ASPECT=16:9                # 16:9 | 9:16 | 1:1
AFTERMOVIE_RES=1920x1080              # 1920x1080 | 3840x2160 | 1080x1920
AFTERMOVIE_FPS=30

# ---- Length ----
# AFTERMOVIE_MAX_LENGTH=90            # blank = song length, capped 90s
AFTERMOVIE_STILL_DURATION=2.5         # seconds per HEIC/JPG photo (Ken Burns)

# ---- Audio ----
AFTERMOVIE_AUDIO_MIX=ducked           # music_only | ducked | clip_only
AFTERMOVIE_MUSIC_DB=-9                # music level in dB (negative = quieter)
AFTERMOVIE_CLIP_DB=-12                # clip-audio level when ducked/clip_only (closer to 0 = louder voices)
AFTERMOVIE_AUDIO_INTEREST_THRESHOLD=0.35  # mute clip audio below this voice-band energy [0..1]; 0 = keep everything

# ---- Edit feel (the Quik-style knobs) ----
AFTERMOVIE_PACE=auto                  # fast | medium | slow | auto (energy-aware, Quik-style)
AFTERMOVIE_TRANSITIONS=soft           # cut | auto | soft
AFTERMOVIE_NO_SPEED_RAMP=false        # true = disable slow-mo ramps on action peaks
AFTERMOVIE_NO_STILLS=false            # true = ignore photos, video only
AFTERMOVIE_NO_REFRAME=false           # true = center-crop instead of face-track in 9:16
"""
