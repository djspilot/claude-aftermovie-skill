"""Read/write the per-folder `.aftermovie-selection.json` exclusion list.

The `aftermovie select` web UI writes a small JSON sidecar at the root of
the user's clips folder that lists which source files the user chose to
exclude before rendering. This module is the single reader so the analyzer,
the server, and the tests all agree on the on-disk format.

On-disk schema:

    {
      "excluded": ["/abs/path/to/IMG_0488.MOV", ...],
      "generated_by": "aftermovie-select",
      "version": 1
    }

`is_excluded(path, folder)` is the hot path the analyzer calls per source.
The mtime-aware caching, atomic write, and malformed-JSON recovery all live
in `SidecarStore` (`analyze/sidecar.py`); this Module is a thin Adapter
that projects the raw sidecar dict into the frozenset shape callers want.
"""
from __future__ import annotations

from pathlib import Path

from aftermovie.analyze.sidecar import SidecarStore

SELECTION_FILENAME = ".aftermovie-selection.json"
SELECTION_VERSION = 1
SELECTION_GENERATOR = "aftermovie-select"

_DEFAULTS: dict[str, list[str]] = {"excluded": []}


def selection_path(folder: Path) -> Path:
    """Where the sidecar lives for a clips folder."""
    return folder / SELECTION_FILENAME


def _store(folder: Path) -> SidecarStore:
    return SidecarStore(folder, SELECTION_FILENAME, _DEFAULTS)


def load_excluded(folder: Path) -> frozenset[str]:
    """Return the set of absolute paths excluded for `folder`, or empty.

    A missing file, malformed JSON, or unexpected schema all yield an
    empty set — selection is opt-in; failing closed (i.e. excluding
    nothing) is the safe default.
    """
    data = _store(folder).read()
    items = data.get("excluded")
    if not isinstance(items, list):
        return frozenset()
    return frozenset(str(p) for p in items if isinstance(p, str))


def save_excluded(folder: Path, excluded: list[str] | tuple[str, ...]) -> Path:
    """Write the sidecar atomically and bust the in-process cache."""
    payload = {
        "excluded": list(dict.fromkeys(str(p) for p in excluded)),  # de-dup, keep order
        "generated_by": SELECTION_GENERATOR,
        "version": SELECTION_VERSION,
    }
    return _store(folder).write(payload)


def is_excluded(path: Path, folder: Path) -> bool:
    """True if `path` is listed in the selection sidecar for `folder`.

    Paths are compared in absolute form so the sidecar (which the GUI
    writes with absolute paths) lines up with whatever absolute path the
    analyzer constructs from `folder.rglob('*')`.
    """
    excluded = load_excluded(folder)
    if not excluded:
        return False
    try:
        abs_path = str(path.resolve())
    except OSError:
        abs_path = str(path)
    return abs_path in excluded


def clear_cache() -> None:
    """Drop the in-process cache. Tests call this between runs."""
    # Selection's cache is keyed per-folder inside SidecarStore. Tests
    # typically call this without a folder reference, so flush everything
    # owned by any SidecarStore — cheap and matches the prior behavior of
    # `_CACHE.clear()` in the old implementation.
    from aftermovie.analyze.sidecar import clear_all_caches

    clear_all_caches()
