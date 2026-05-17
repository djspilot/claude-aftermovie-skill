"""Paths, defaults, and asset resolution shared by CLI and MCP server."""
from __future__ import annotations

from pathlib import Path

# ---- Defaults ----------------------------------------------------------------

DEFAULT_TARGET_LEN_S = 90
DEFAULT_FPS = 30
DEFAULT_RES = "1920x1080"
DEFAULT_MUSIC_DB = -8.0
DEFAULT_CLIP_DB = -18.0

# Sub-clip candidate length range (seconds).
MIN_CLIP_S = 0.4
MAX_CLIP_S = 4.0

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".insv", ".lrv", ".MP4", ".MOV", ".M4V"}

# ---- Theme presets -----------------------------------------------------------

THEMES: dict[str, dict] = {
    "cinematic": {
        "lut": "cinematic", "music_db": -9.0, "no_speed_ramp": False,
        "transitions": "soft", "audio_mix": "ducked", "pace": "medium",
        "description": "Glide-y crossfades, ducked audio — looks like a Quik edit.",
    },
    "punchy": {
        "lut": "punchy", "music_db": -6.0, "no_speed_ramp": False,
        "transitions": "auto", "audio_mix": "ducked", "pace": "fast",
        "description": "Fast cuts, whips on peaks, hot music — hype mode.",
    },
    "chill": {
        "lut": "chill", "music_db": -10.0, "no_speed_ramp": True,
        "transitions": "soft", "audio_mix": "ducked", "pace": "slow",
        "description": "Slow downbeat pacing, soft crossfades, no ramps.",
    },
    "nostalgic": {
        "lut": "nostalgic", "music_db": -10.0, "no_speed_ramp": False,
        "transitions": "soft", "audio_mix": "ducked", "pace": "medium",
        "description": "Film-look LUT, warm fades, ducked voice-aware audio.",
    },
}

# ---- Paths -------------------------------------------------------------------

def package_dir() -> Path:
    """Directory of the installed `aftermovie` package."""
    return Path(__file__).resolve().parent


def skill_dir() -> Path:
    """
    Repository root — one level above `src/aftermovie` when running from source,
    or the package dir itself when installed (assets shipped under _assets).
    """
    pkg = package_dir()
    # Source layout: <repo>/src/aftermovie/config.py → repo root is parents[2]
    if (pkg.parents[1] / "assets" / "luts").is_dir():
        return pkg.parents[1]
    # Installed layout: <site-packages>/aftermovie/_assets/luts
    return pkg


def lut_dir() -> Path:
    src = skill_dir() / "assets" / "luts"
    if src.is_dir():
        return src
    return package_dir() / "_assets" / "luts"


def fonts_dir() -> Path:
    src = skill_dir() / "assets" / "fonts"
    if src.is_dir():
        return src
    return package_dir() / "_assets" / "fonts"


def models_dir() -> Path:
    src = skill_dir() / "assets" / "models"
    if src.is_dir():
        return src
    return package_dir() / "_assets" / "models"


def data_dir() -> Path:
    return Path.home() / ".skills-data" / "aftermovie"


def resolve_lut(lut_arg: str | None) -> Path | None:
    """Resolve a LUT argument to a file path. Accepts absolute path or theme name."""
    if not lut_arg:
        return lut_dir() / "cinematic.cube"
    p = Path(lut_arg).expanduser()
    if p.is_file():
        return p
    themed = lut_dir() / f"{lut_arg}.cube"
    if themed.is_file():
        return themed
    return None


def list_luts() -> list[dict]:
    d = lut_dir()
    if not d.is_dir():
        return []
    return [{"name": p.stem, "path": str(p)} for p in sorted(d.glob("*.cube"))]
