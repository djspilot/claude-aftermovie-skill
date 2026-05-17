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

`is_excluded(path, folder)` is the hot path the analyzer calls per source —
it caches the parsed JSON per folder so we read each sidecar at most once
per process (mtime-aware so test runs that rewrite the file see the update).
"""
from __future__ import annotations

import json
from pathlib import Path

SELECTION_FILENAME = ".aftermovie-selection.json"
SELECTION_VERSION = 1
SELECTION_GENERATOR = "aftermovie-select"

# Cache: folder_path_str -> (mtime_ns, frozenset[abs_path_str])
# Mtime tracking lets tests (and the server) rewrite the sidecar mid-process
# and have subsequent `is_excluded` calls see the new value.
_CACHE: dict[str, tuple[int, frozenset[str]]] = {}


def selection_path(folder: Path) -> Path:
    """Where the sidecar lives for a clips folder."""
    return folder / SELECTION_FILENAME


def load_excluded(folder: Path) -> frozenset[str]:
    """Return the set of absolute paths excluded for `folder`, or empty.

    A missing file, malformed JSON, or unexpected schema all yield an
    empty set — selection is opt-in; failing closed (i.e. excluding
    nothing) is the safe default.
    """
    sidecar = selection_path(folder)
    key = str(folder.resolve())
    try:
        st = sidecar.stat()
    except OSError:
        # No sidecar — clear any stale cache entry for this folder.
        _CACHE.pop(key, None)
        return frozenset()

    cached = _CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime_ns:
        return cached[1]

    try:
        raw = sidecar.read_text()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        excluded: frozenset[str] = frozenset()
        _CACHE[key] = (st.st_mtime_ns, excluded)
        return excluded

    items = data.get("excluded") if isinstance(data, dict) else None
    if not isinstance(items, list):
        excluded = frozenset()
    else:
        excluded = frozenset(str(p) for p in items if isinstance(p, str))
    _CACHE[key] = (st.st_mtime_ns, excluded)
    return excluded


def save_excluded(folder: Path, excluded: list[str] | tuple[str, ...]) -> Path:
    """Write the sidecar atomically and bust the in-process cache."""
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    sidecar = selection_path(folder)
    payload = {
        "excluded": list(dict.fromkeys(str(p) for p in excluded)),  # de-dup, keep order
        "generated_by": SELECTION_GENERATOR,
        "version": SELECTION_VERSION,
    }
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(sidecar)
    # Invalidate cache; load_excluded will pick up the new mtime next call.
    _CACHE.pop(str(folder), None)
    return sidecar


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
    _CACHE.clear()
