"""Read/write the per-folder `.aftermovie-preferences.json` sidecar.

The `aftermovie select` web UI writes a sidecar at the root of the user's
clips folder that captures lightweight per-project preferences:

    {
      "favorited":      ["/abs/path/to/clip.mp4", ...],
      "banned":         ["/abs/path/to/other.mp4", ...],
      "pinned_entries": ["plan-entry-id-1", ...],
      "generated_by":   "aftermovie-select",
      "version":        1
    }

Sibling to `analyze/selection.py` (which owns `.aftermovie-selection.json`):
the two sidecars live side-by-side and are read independently. Selection is
the GUI's "exclude this file from the source pool" toggle; preferences are
the user's longer-lived likes/bans/pins that survive across renders and
influence scoring.

`pinned_entries` is intentionally left unwired for now — pinning requires a
stable plan-entry id model that doesn't exist yet. The field is reserved in
the on-disk schema so we don't break the format when the wire-up lands.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFERENCES_FILENAME = ".aftermovie-preferences.json"
PREFERENCES_VERSION = 1
PREFERENCES_GENERATOR = "aftermovie-select"

# Cache: folder_path_str -> (mtime_ns, dict-with-frozensets).
# Mtime tracking lets the server and tests rewrite the sidecar mid-process
# and have subsequent reads see the new value (same pattern as selection.py).
_CACHE: dict[str, tuple[int, dict[str, frozenset[str]]]] = {}


def preferences_path(folder: Path) -> Path:
    """Where the sidecar lives for a clips folder."""
    return folder / PREFERENCES_FILENAME


def _empty_prefs() -> dict[str, frozenset[str]]:
    return {
        "favorited": frozenset(),
        "banned": frozenset(),
        "pinned_entries": frozenset(),
    }


def _load_cached(folder: Path) -> dict[str, frozenset[str]]:
    """Return the parsed sidecar as frozensets, or empty defaults.

    A missing file, malformed JSON, or unexpected schema all yield empty
    sets — preferences are opt-in and failing closed is the safe default.
    """
    sidecar = preferences_path(folder)
    key = str(folder.resolve())
    try:
        st = sidecar.stat()
    except OSError:
        _CACHE.pop(key, None)
        return _empty_prefs()

    cached = _CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime_ns:
        return cached[1]

    try:
        raw = sidecar.read_text()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        prefs = _empty_prefs()
        _CACHE[key] = (st.st_mtime_ns, prefs)
        return prefs

    if not isinstance(data, dict):
        prefs = _empty_prefs()
        _CACHE[key] = (st.st_mtime_ns, prefs)
        return prefs

    def _field(name: str) -> frozenset[str]:
        items = data.get(name)
        if not isinstance(items, list):
            return frozenset()
        return frozenset(str(p) for p in items if isinstance(p, str))

    prefs = {
        "favorited": _field("favorited"),
        "banned": _field("banned"),
        "pinned_entries": _field("pinned_entries"),
    }
    _CACHE[key] = (st.st_mtime_ns, prefs)
    return prefs


def load_preferences(folder: Path) -> dict[str, list[str]]:
    """Return the preferences for `folder` as a JSON-friendly dict.

    Always returns a dict with the three documented fields. Missing or
    malformed sidecars yield empty lists, never an exception. The lists are
    fresh copies — mutating them does not affect the in-process cache.
    """
    prefs = _load_cached(folder)
    return {
        "favorited": sorted(prefs["favorited"]),
        "banned": sorted(prefs["banned"]),
        "pinned_entries": sorted(prefs["pinned_entries"]),
    }


def save_preferences(folder: Path, prefs: dict[str, Any]) -> Path:
    """Write the sidecar atomically and bust the in-process cache.

    `prefs` is a dict with any subset of the documented fields; missing
    fields are persisted as empty lists. Non-string entries are dropped.
    Duplicates are removed while preserving first-seen order.
    """
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    sidecar = preferences_path(folder)

    def _clean(name: str) -> list[str]:
        raw = prefs.get(name) if isinstance(prefs, dict) else None
        if not isinstance(raw, list):
            return []
        return list(dict.fromkeys(str(p) for p in raw if isinstance(p, str)))

    payload = {
        "favorited": _clean("favorited"),
        "banned": _clean("banned"),
        "pinned_entries": _clean("pinned_entries"),
        "generated_by": PREFERENCES_GENERATOR,
        "version": PREFERENCES_VERSION,
    }
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(sidecar)
    _CACHE.pop(str(folder), None)
    return sidecar


def _abs(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def is_favorited(folder: Path, path: Path | str) -> bool:
    """True if `path` is in the favorited list for `folder`."""
    favorited = _load_cached(folder)["favorited"]
    if not favorited:
        return False
    p = Path(path) if not isinstance(path, Path) else path
    return _abs(p) in favorited


def is_banned(folder: Path, path: Path | str) -> bool:
    """True if `path` is in the banned list for `folder`."""
    banned = _load_cached(folder)["banned"]
    if not banned:
        return False
    p = Path(path) if not isinstance(path, Path) else path
    return _abs(p) in banned


def clear_cache() -> None:
    """Drop the in-process cache. Tests call this between runs."""
    _CACHE.clear()
