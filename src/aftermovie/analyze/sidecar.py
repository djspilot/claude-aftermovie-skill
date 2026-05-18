"""Per-folder JSON sidecar repository — the shared seam under preferences + selection.

The `aftermovie select` web UI writes two sidecars at the root of the user's
clips folder: `.aftermovie-selection.json` (the exclusion list) and
`.aftermovie-preferences.json` (favorites / bans / reserved pins). Both
share an identical on-disk shape:

    {
      "<some-list-field>": [...],
      "generated_by":      "aftermovie-select",
      "version":           1
    }

…and both Modules used to implement the same plumbing independently:

  * an mtime-aware in-process cache so the analyzer can read once per run,
  * atomic write-then-rename so a crashed writer can't truncate the file,
  * malformed-JSON recovery so a hand-edited or partial sidecar doesn't
    explode the pipeline (caller sees `schema_defaults` instead).

`SidecarStore` owns that plumbing once. `analyze/preferences.py` and
`analyze/selection.py` are thin adapters that translate the raw dict the
store returns into the frozenset / list projections their callers want.

Interface (the test surface):

    store = SidecarStore(folder, filename, schema_defaults)
    store.path()             -> Path          # where the sidecar lives
    store.read()             -> dict          # cached, mtime-aware
    store.write(payload)     -> Path          # atomic, busts cache, returns sidecar path
    store.clear_cache()      -> None          # drop this store's cache entry

Invariants:

  * `read()` never raises for missing files, bad JSON, or non-dict roots —
    those all yield a fresh copy of `schema_defaults`.
  * A `read()` after `write(p)` returns the just-written payload.
  * The same on-disk mtime is parsed at most once per process per store.
  * `write()` is atomic from the reader's point of view: a partially written
    sidecar never replaces a good one (write-to-`.tmp`-then-`os.replace`).
  * Returned dicts are fresh copies; mutating them does not corrupt the cache.
  * Malformed JSON is logged to stderr exactly once per (folder, filename)
    per process — subsequent malformed reads stay silent.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

# Module-level cache keyed by (resolved_folder, filename):
#   value is (mtime_ns, parsed_dict). The parsed_dict is a deep-copyable
#   snapshot of the on-disk JSON; we hand callers fresh copies on each
#   `read()` so mutation can't leak back into the cache.
_CACHE: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}

# Track which (folder, filename) pairs have already complained about
# malformed JSON. "Log once" keeps the analyzer's stderr quiet when the
# same broken sidecar is read repeatedly during a single run.
_WARNED_MALFORMED: set[tuple[str, str]] = set()


class SidecarStore:
    """Read/write a single JSON sidecar at `folder / filename`.

    `schema_defaults` is the dict the store hands back when there's nothing
    valid to return (no file, bad JSON, non-dict root). Pass a fresh dict —
    the store deep-copies it before returning so callers can't mutate the
    template by accident.
    """

    def __init__(
        self,
        folder: Path,
        filename: str,
        schema_defaults: dict[str, Any],
    ) -> None:
        self._folder = folder
        self._filename = filename
        self._defaults = schema_defaults

    # ---- identity ---------------------------------------------------------

    def path(self) -> Path:
        """Absolute path to the sidecar (whether or not it exists)."""
        return self._folder / self._filename

    def _key(self) -> tuple[str, str]:
        # Resolve lazily — the folder may not exist on the first call, and
        # `Path.resolve()` still returns a usable absolute key in that case.
        try:
            resolved = str(self._folder.resolve())
        except OSError:
            resolved = str(self._folder)
        return (resolved, self._filename)

    # ---- read -------------------------------------------------------------

    def read(self) -> dict[str, Any]:
        """Return the parsed sidecar dict, or a fresh copy of `schema_defaults`.

        Mtime-aware: if the file hasn't changed since the last parse, the
        cached dict is returned (deep-copied so callers can't mutate it).
        Missing files, malformed JSON, and non-dict roots all yield the
        defaults — preferences and selection are both opt-in, so failing
        closed is the safe behavior. Malformed JSON also emits one stderr
        line per (folder, filename) per process.
        """
        sidecar = self.path()
        key = self._key()
        try:
            st = sidecar.stat()
        except OSError:
            # No file — drop any stale cache entry and return defaults.
            _CACHE.pop(key, None)
            return copy.deepcopy(self._defaults)

        cached = _CACHE.get(key)
        if cached is not None and cached[0] == st.st_mtime_ns:
            return copy.deepcopy(cached[1])

        try:
            raw = sidecar.read_text()
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            data = None

        if not isinstance(data, dict):
            if data is not None or sidecar.is_file():
                # We touched the file but couldn't parse a dict out of it.
                self._warn_malformed_once(key)
            data = copy.deepcopy(self._defaults)

        _CACHE[key] = (st.st_mtime_ns, data)
        return copy.deepcopy(data)

    # ---- write ------------------------------------------------------------

    def write(self, payload: dict[str, Any]) -> Path:
        """Atomically write `payload` as JSON, bust the cache, return the path.

        Uses write-to-`.tmp`-then-`os.replace` so a partial write never
        replaces a good sidecar on disk. The folder is created if missing.
        """
        # Resolve here so the cache key after the write matches future reads.
        folder = self._folder.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        sidecar = folder / self._filename

        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(sidecar)

        # Invalidate both the pre-resolve and post-resolve cache keys; the
        # next `read()` will repopulate from the new mtime.
        _CACHE.pop((str(folder), self._filename), None)
        _CACHE.pop(self._key(), None)
        # A successful write means the file is well-formed again; reset the
        # "we already complained" flag so a future corruption gets one line.
        _WARNED_MALFORMED.discard((str(folder), self._filename))
        _WARNED_MALFORMED.discard(self._key())
        return sidecar

    # ---- cache control ----------------------------------------------------

    def clear_cache(self) -> None:
        """Drop this store's cache entry. Tests call this between runs."""
        _CACHE.pop(self._key(), None)
        # Also clear the unresolved-folder variant in case the folder didn't
        # exist when the store was constructed.
        _CACHE.pop((str(self._folder), self._filename), None)

    # ---- helpers ----------------------------------------------------------

    def _warn_malformed_once(self, key: tuple[str, str]) -> None:
        if key in _WARNED_MALFORMED:
            return
        _WARNED_MALFORMED.add(key)
        try:
            print(
                f"aftermovie: ignoring malformed sidecar {self.path()} "
                "(falling back to defaults)",
                file=sys.stderr,
                flush=True,
            )
        except OSError:
            # Stderr can be closed under test capture; swallow rather than
            # let a logging hiccup propagate into a caller's pipeline.
            pass


def clear_all_caches() -> None:
    """Drop every store's cache entry. Module-wide reset for tests."""
    _CACHE.clear()
    _WARNED_MALFORMED.clear()
