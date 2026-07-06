"""Per-clip analyze cache Module.
2
The folder-level `CatalogRepository` cache (Phase 1, `repos.py`) handles
*"this whole folder was analyzed before"*. It's content-hashed on the
sorted file list + mtimes, so ANY change to the folder (one new file, one
touched mtime) busts the entire cache — all 61 clips re-analyze.

`AnalyzeCache` is the layer below: a per-clip cache keyed by
SHA1(`abs_path | mtime_ns | size`) so the common case (60 unchanged clips
+ 1 new file) only pays the analyze cost for the new file.

Interface
=========

    cache = AnalyzeCache()
    key = cache.key_for(path)              # stable until file changes
    hit = cache.get(key)                   # dict | None
    cache.put(key, info)                   # ClipInfo → on-disk JSON

Storage layout
==============

    ~/.skills-data/aftermovie/analyze-cache/<hash[:2]>/<hash>.json

The 2-char shard keeps any single directory from accumulating thousands
of entries (some macOS filesystems get slow when a dir has > ~10k
children). The cache is bounded by `AFTERMOVIE_ANALYZE_CACHE_MAX_GB`
(default 1 GB); when an insertion would push the cache past the cap, the
oldest-mtime entries are evicted until we're back under.

Invariants
==========

  * `key_for` is pure: same `(abs_path, mtime_ns, size)` → same key.
    In-place edits change mtime_ns and produce a fresh key — the stale
    entry is never read again and gets evicted by LRU eventually.
  * `get` returns `None` (not raises) on missing entries OR corrupt
    JSON. Corrupt files are silently dropped; the caller re-analyzes.
  * `put` only stores successful results. Failed analyses (`None` from
    `analyze_clip`) MUST NOT be cached — a transient probe failure would
    otherwise poison the cache forever.
  * LRU eviction is best-effort and runs on `put`. It uses `st_mtime`
    rather than `st_atime` because most macOS filesystems mount with
    `noatime` and `st_atime` is no longer a reliable touch signal.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from aftermovie import config


_DEFAULT_MAX_GB = 1.0
_ENV_MAX_GB = "AFTERMOVIE_ANALYZE_CACHE_MAX_GB"


def _cache_root() -> Path:
    """Where the on-disk cache lives.

    Routed through `config.data_dir()` so tests can monkeypatch the data
    root and not pollute the user's real `~/.skills-data/aftermovie/`.
    """
    d = config.data_dir() / "analyze-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _max_bytes() -> int:
    """Cache cap in bytes, honouring the env override."""
    raw = os.environ.get(_ENV_MAX_GB, "").strip()
    try:
        gb = float(raw) if raw else _DEFAULT_MAX_GB
    except ValueError:
        gb = _DEFAULT_MAX_GB
    return max(0, int(gb * 1024 * 1024 * 1024))


class AnalyzeCache:
    """Per-clip on-disk cache for `analyze_clip` outputs.

    Stateless — every method recomputes paths from the file system. Safe
    to instantiate per-call; the module-level `analyze_cache` singleton at
    the bottom of the file is the conventional handle.
    """

    # Bump when the ANALYZER changes what it extracts (new ClipInfo fields,
    # different captured_at strategy, ...) so every stale entry misses and
    # re-analyzes — nobody remembers to --force-reanalyze.
    SCHEMA_VERSION = 3  # v3: origin_still on materialized stills

    def key_for(self, path: Path) -> str:
        """SHA1 over `schema | abs_path | mtime_ns | size`.

        Returns a 40-char hex digest. Raises OSError if the file can't be
        stat'd — callers should let that bubble (a vanished file is also
        a vanished cache entry, and `analyze_clip` will fail on the same
        path moments later).
        """
        p = Path(path).resolve()
        st = p.stat()
        seed = f"v{self.SCHEMA_VERSION}|{p}|{st.st_mtime_ns}|{st.st_size}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()

    def path_for_key(self, key: str) -> Path:
        """Where the entry with this key lives (or would live)."""
        return _cache_root() / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached ClipInfo-as-dict, or None on miss / corruption."""
        p = self.path_for_key(key)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            # Corrupt or unreadable cache entry — treat as miss. The next
            # `put` will overwrite it with a fresh, valid payload.
            return None

    def put(self, key: str, info: Any) -> Path:
        """Persist `info` (ClipInfo dataclass OR plain dict) under `key`.

        Touches `st_mtime` on insertion so the LRU eviction sweep can use
        it as a "last used" signal. Returns the on-disk path.
        """
        p = self.path_for_key(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(info) if is_dataclass(info) and not isinstance(info, type) else info
        p.write_text(json.dumps(payload))
        # Bump mtime AFTER write so the eviction sweep sees this entry as
        # the most-recently-used. Without it, write order alone would
        # determine eviction order on a busy run.
        os.utime(p, None)
        self._evict_if_over_cap()
        return p

    # ---- introspection (used by `doctor` + tests) --------------------------

    def entries(self) -> list[Path]:
        """All cache files, regardless of shard."""
        root = _cache_root()
        if not root.is_dir():
            return []
        return [p for p in root.rglob("*.json") if p.is_file()]

    def total_bytes(self) -> int:
        """Sum of `st_size` across every cache entry on disk."""
        total = 0
        for p in self.entries():
            try:
                total += p.stat().st_size
            except OSError:
                continue
        return total

    def root(self) -> Path:
        """Public handle for the doctor command."""
        return _cache_root()

    # ---- LRU eviction (private) -------------------------------------------

    def _evict_if_over_cap(self) -> None:
        """Drop oldest-mtime entries until total_bytes <= cap.

        Best-effort: failures to delete (race with another writer, perms)
        are swallowed; the cap is a soft target. We re-stat each pass so
        a concurrent writer adding entries between sweeps doesn't blow
        our budget.

        The newest entry is protected — even if the cap is set absurdly
        low (smaller than a single entry's payload), we never evict the
        most recently written file. Otherwise a fresh `put` would
        immediately delete its own result.
        """
        cap = _max_bytes()
        if cap <= 0:
            return
        entries = self.entries()
        # Snapshot size+mtime per file so a concurrent write doesn't
        # change ordering mid-sort.
        sized: list[tuple[float, int, Path]] = []
        total = 0
        for p in entries:
            try:
                st = p.stat()
            except OSError:
                continue
            sized.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        if total <= cap:
            return
        # Sort oldest-first so we evict stale entries before fresh ones.
        # Drop the newest entry from the eviction pool — we never delete
        # the file we just wrote (or whichever is currently freshest).
        sized.sort(key=lambda t: t[0])
        protected = sized[-1][2] if sized else None
        for _mtime, size, p in sized:
            if total <= cap:
                break
            if p == protected:
                continue
            try:
                p.unlink()
                total -= size
            except OSError:
                continue


# Module-level singleton — stateless, safe to share.
analyze_cache = AnalyzeCache()


__all__ = ["AnalyzeCache", "analyze_cache"]
