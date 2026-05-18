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

This module is a thin Adapter over `SidecarStore` (`analyze/sidecar.py`),
which owns the mtime-aware cache, atomic write, and malformed-JSON
recovery shared with `analyze/selection.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aftermovie.analyze.sidecar import SidecarStore

PREFERENCES_FILENAME = ".aftermovie-preferences.json"
PREFERENCES_VERSION = 1
PREFERENCES_GENERATOR = "aftermovie-select"

_LIST_FIELDS = ("favorited", "banned", "pinned_entries")
_DEFAULTS: dict[str, list[str]] = {name: [] for name in _LIST_FIELDS}


def preferences_path(folder: Path) -> Path:
    """Where the sidecar lives for a clips folder."""
    return folder / PREFERENCES_FILENAME


def _store(folder: Path) -> SidecarStore:
    return SidecarStore(folder, PREFERENCES_FILENAME, _DEFAULTS)


def _read_sets(folder: Path) -> dict[str, frozenset[str]]:
    """Parse the sidecar into frozensets, one per documented list field.

    The store hands back a raw dict (or the defaults on missing/malformed);
    we just project the three list fields into frozensets and drop any
    non-string entries.
    """
    data = _store(folder).read()

    def _field(name: str) -> frozenset[str]:
        items = data.get(name)
        if not isinstance(items, list):
            return frozenset()
        return frozenset(str(p) for p in items if isinstance(p, str))

    return {name: _field(name) for name in _LIST_FIELDS}


def load_preferences(folder: Path) -> dict[str, list[str]]:
    """Return the preferences for `folder` as a JSON-friendly dict.

    Always returns a dict with the three documented fields. Missing or
    malformed sidecars yield empty lists, never an exception. The lists are
    fresh copies — mutating them does not affect the in-process cache.
    """
    prefs = _read_sets(folder)
    return {name: sorted(prefs[name]) for name in _LIST_FIELDS}


def save_preferences(folder: Path, prefs: dict[str, Any]) -> Path:
    """Write the sidecar atomically and bust the in-process cache.

    `prefs` is a dict with any subset of the documented fields; missing
    fields are persisted as empty lists. Non-string entries are dropped.
    Duplicates are removed while preserving first-seen order.
    """
    def _clean(name: str) -> list[str]:
        raw = prefs.get(name) if isinstance(prefs, dict) else None
        if not isinstance(raw, list):
            return []
        return list(dict.fromkeys(str(p) for p in raw if isinstance(p, str)))

    payload: dict[str, Any] = {name: _clean(name) for name in _LIST_FIELDS}
    payload["generated_by"] = PREFERENCES_GENERATOR
    payload["version"] = PREFERENCES_VERSION
    return _store(folder).write(payload)


def _abs(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def is_favorited(folder: Path, path: Path | str) -> bool:
    """True if `path` is in the favorited list for `folder`."""
    favorited = _read_sets(folder)["favorited"]
    if not favorited:
        return False
    p = Path(path) if not isinstance(path, Path) else path
    return _abs(p) in favorited


def is_banned(folder: Path, path: Path | str) -> bool:
    """True if `path` is in the banned list for `folder`."""
    banned = _read_sets(folder)["banned"]
    if not banned:
        return False
    p = Path(path) if not isinstance(path, Path) else path
    return _abs(p) in banned


def clear_cache() -> None:
    """Drop the in-process cache. Tests call this between runs."""
    from aftermovie.analyze.sidecar import clear_all_caches

    clear_all_caches()
