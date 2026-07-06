"""Per-clip prerender cache.

Renders are dominated by the `_prerender_clip` stage — one ffmpeg invocation
per Entry that decodes, scales, color-grades, speed-ramps and re-encodes the
source. Tweaking a downstream knob (LUT, music duck, mux) re-runs the whole
prerender stage even though the per-clip outputs are byte-for-byte identical
to the previous run. E1 makes that "tweak knob → re-render" loop ~instant.

The Module exposes two Interfaces:

    PrerenderOpts       Value object — the (encoder, geometry, audio) knobs
                        that participate in the cache key, alongside the
                        per-entry source/start/end/speed fields.
    PrerenderCache      Storage Module — `key_for`, `get`, `put`,
                        `lookup_or_compute`, plus LRU eviction.

The cache directory lives at `~/.skills-data/aftermovie/prerender-cache/` and
is sharded `<hash[:2]>/<hash>.mp4` so we never put 10k+ files into a single
directory (slow `readdir`, FS-specific limits).

Eviction is pure LRU keyed by file `atime`, capped at
`AFTERMOVIE_PRERENDER_CACHE_MAX_GB` (default 10 GB). There is no time-based
TTL — a cached clip is valid forever as long as the source file hasn't
changed (mtime + size are part of the key).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aftermovie import config
from aftermovie.render.encoder import EncoderProfile


__all__ = [
    "PrerenderOpts",
    "PrerenderCache",
]


# ---- Tunables ---------------------------------------------------------------

_DEFAULT_MAX_GB = 10.0
_STATS_FILENAME = "stats.json"


def _max_bytes() -> int:
    """Cache size cap in bytes from `AFTERMOVIE_PRERENDER_CACHE_MAX_GB`."""
    raw = os.environ.get("AFTERMOVIE_PRERENDER_CACHE_MAX_GB", "")
    try:
        gb = float(raw) if raw else _DEFAULT_MAX_GB
    except ValueError:
        gb = _DEFAULT_MAX_GB
    return int(max(0.0, gb) * (1024 ** 3))


# ---- PrerenderOpts ----------------------------------------------------------

@dataclass(frozen=True)
class PrerenderOpts:
    """The render-wide knobs that participate in the prerender cache key.

    Per-entry knobs (source path, start/end, speed ramp, audio_interest) come
    from the Entry dict in `PrerenderCache.key_for`. These are the values that
    are constant across all clips of a single render but change between
    renders (e.g. changing the LUT invalidates every clip; flipping aspect
    ratio invalidates every clip; bumping the encoder bitrate ditto).
    """

    aspect: str
    target_res: str
    fps: int
    lut: Path | None
    encoder: EncoderProfile
    keep_audio: bool
    audio_interest_threshold: float


# ---- PrerenderCache ---------------------------------------------------------

class PrerenderCache:
    """Content-addressed prerender cache backed by `data_dir/prerender-cache/`.

    Storage is "move into place" semantics: `put(key, src)` hard-links (or
    copies, when the workdir is on a different volume) the freshly-rendered
    clip into the cache. The renderer never writes directly into the cache —
    that way a failed ffmpeg run (which would leave a half-written file at
    the workdir path) cannot poison the cache.

    Stats (`hits`, `misses`) are tracked in a sidecar `stats.json` for the
    `aftermovie cache stats` CLI. Writes are best-effort: a corrupted
    stats.json silently resets to zeros rather than crashing a render.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root_override = Path(root) if root is not None else None

    # ---- paths ------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Cache root — `data_dir/prerender-cache/` unless overridden."""
        if self._root_override is not None:
            return self._root_override
        return config.data_dir() / "prerender-cache"

    def _shard_dir(self, key: str) -> Path:
        return self.root / key[:2]

    def path_for_key(self, key: str) -> Path:
        """Where the clip with this key lives (or would live) on disk."""
        return self._shard_dir(key) / f"{key}.mp4"

    def _stats_path(self) -> Path:
        return self.root / _STATS_FILENAME

    # ---- key derivation ---------------------------------------------------

    def key_for(self, entry: dict, opts: PrerenderOpts) -> str:
        """SHA1 over every input that affects the per-clip ffmpeg output.

        The mtime + size guard against silent source-file edits — if the
        source was re-encoded out from under us, the cached clip is no
        longer valid and we get a fresh key (miss → recompute).

        Recipe (in order):
            source_path | source_mtime | source_size
            | start_s | end_s | speed_start | speed_end
            | aspect | target_res | fps
            | lut_name_or_none
            | encoder.name | encoder.pix_fmt
            | audio_interest_gate_threshold | keep_audio
            | luma_offset
        """
        src = Path(entry["source"])
        try:
            st = src.stat()
            src_mtime = int(st.st_mtime_ns)
            src_size = int(st.st_size)
        except OSError:
            # Source file vanished — punt to a key that will miss; the
            # caller will then try to ffmpeg the source, also fail, and
            # surface a clean error rather than silently returning stale
            # cache.
            src_mtime = -1
            src_size = -1

        speed = float(entry.get("speed", 1.0))
        s_start = float(entry.get("speed_start", speed))
        s_end = float(entry.get("speed_end", speed))
        # `out_duration_s` belongs to the slot, not the source span — it's
        # what `_compensated_render_entry` mutates to compensate for
        # transition overlap. Two entries with the same source span but
        # different transition-in durations must produce different cached
        # clips, otherwise the assemble stage gets a clip that's the wrong
        # length. Include it in the key.
        out_duration = float(entry.get(
            "out_duration_s",
            (float(entry["end_s"]) - float(entry["start_s"])) / max(speed, 0.0001),
        ))

        lut_name = opts.lut.name if opts.lut is not None else ""

        parts = [
            str(src.resolve()) if src.exists() else str(src),
            str(src_mtime),
            str(src_size),
            f"{float(entry['start_s']):.6f}",
            f"{float(entry['end_s']):.6f}",
            f"{s_start:.6f}",
            f"{s_end:.6f}",
            f"{out_duration:.6f}",
            opts.aspect,
            opts.target_res,
            str(opts.fps),
            lut_name,
            opts.encoder.name,
            opts.encoder.pix_fmt,
            f"{float(opts.audio_interest_threshold):.6f}",
            "1" if opts.keep_audio else "0",
            # Color-consistency nudge — same span, different brightness
            # correction must be a different cached clip.
            f"{float(entry.get('luma_offset') or 0.0):.4f}",
            # Outro fade — the same span with/without a tail fade differs.
            f"{float(entry.get('fade_out_s') or 0.0):.3f}",
            "stab" if entry.get("stabilize") else "",
        ]
        seed = "|".join(parts)
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()

    # ---- get / put / lookup_or_compute -----------------------------------

    def get(self, key: str) -> Path | None:
        """Return the cached clip's path, or None for a miss.

        Touches `atime` on hit so the LRU evictor sees recent use.
        """
        p = self.path_for_key(key)
        if not p.is_file():
            self._bump_stat("misses")
            return None
        # Refresh atime/mtime so eviction treats this as a recent use. We
        # bump both because some filesystems mount with `noatime`, but
        # `os.utime` always updates mtime so we have a fallback ordering
        # signal regardless of mount flags.
        try:
            now = _now()
            os.utime(p, (now, now))
        except OSError:
            pass
        self._bump_stat("hits")
        return p

    def put(self, key: str, src_path: Path) -> Path:
        """Move `src_path` into the cache. Returns the cached location.

        Hard-links when source and dest share a device, otherwise copies.
        After the put, LRU-evicts oldest entries until total disk usage is
        under the cap.
        """
        dest = self.path_for_key(key)
        self._shard_dir(key).mkdir(parents=True, exist_ok=True)
        # If a partial / stale file exists at dest, scrap it.
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        try:
            os.link(src_path, dest)
        except OSError:
            # Cross-device or filesystem-without-hardlinks (e.g. SMB) →
            # fall back to a copy. Slower but correct.
            shutil.copy2(src_path, dest)
        # Mark fresh so LRU keeps this entry around.
        try:
            now = _now()
            os.utime(dest, (now, now))
        except OSError:
            pass
        self._evict_to_cap()
        return dest

    def lookup_or_compute(
        self,
        entry: dict,
        opts: PrerenderOpts,
        compute_fn: Callable[[Path], bool],
    ) -> Path | None:
        """Hit → return cached path. Miss → compute_fn(tmp) → put → return.

        `compute_fn(out_path)` must:
          - write a finished mp4 at `out_path`
          - return True on success, False on failure
        On failure (False) the cache is NOT updated, the temp file (if any)
        is removed by the caller, and this returns None so the renderer can
        skip the entry.
        """
        key = self.key_for(entry, opts)
        hit = self.get(key)
        if hit is not None:
            return hit
        # The caller hands us a workdir path it owns — compute into it and
        # then move it into the cache. We don't manage the workdir lifecycle
        # because each call site (currently just `_prerender_clip`) already
        # owns one via tempfile.TemporaryDirectory.
        return None  # pragma: no cover - convenience API not used in MVP

    # ---- stats ------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of cache state for `aftermovie cache stats`.

        Includes:
            hits / misses                — counts since the last clear
            entry_count                  — files currently in the cache
            bytes_used                   — total file size
            oldest_atime / newest_atime  — for "is this cache stale?"
            cap_bytes                    — current max from env / default
        """
        counters = self._load_stats()
        entries = list(self._iter_entries())
        bytes_used = sum(e[1] for e in entries)
        atimes = [e[2] for e in entries]
        return {
            "hits": int(counters.get("hits", 0)),
            "misses": int(counters.get("misses", 0)),
            "entry_count": len(entries),
            "bytes_used": bytes_used,
            "oldest_atime": min(atimes) if atimes else None,
            "newest_atime": max(atimes) if atimes else None,
            "cap_bytes": _max_bytes(),
            "root": str(self.root),
        }

    def clear(self) -> None:
        """Remove the entire cache directory (entries + stats)."""
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    # ---- internals --------------------------------------------------------

    def _iter_entries(self) -> list[tuple[Path, int, float]]:
        """Yield (path, size, atime) for every cached clip in the tree."""
        out: list[tuple[Path, int, float]] = []
        if not self.root.is_dir():
            return out
        for shard in self.root.iterdir():
            if not shard.is_dir():
                continue
            for f in shard.iterdir():
                if not f.is_file() or f.suffix != ".mp4":
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                # Some filesystems mount with noatime — fall back to mtime
                # so the LRU ordering signal is monotonic regardless.
                atime = max(st.st_atime, st.st_mtime)
                out.append((f, int(st.st_size), float(atime)))
        return out

    def _evict_to_cap(self) -> None:
        """LRU eviction loop: drop oldest entries until under `_max_bytes()`."""
        cap = _max_bytes()
        if cap <= 0:
            # Disabled — never evict (also never populate sensibly, but the
            # caller is responsible for that).
            return
        entries = self._iter_entries()
        total = sum(e[1] for e in entries)
        if total <= cap:
            return
        # Oldest first — pop until under cap.
        entries.sort(key=lambda e: e[2])
        for path, size, _atime in entries:
            if total <= cap:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue

    def _load_stats(self) -> dict:
        p = self._stats_path()
        if not p.is_file():
            return {"hits": 0, "misses": 0}
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {"hits": 0, "misses": 0}

    def _save_stats(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self._stats_path().write_text(json.dumps(data))
        except OSError:
            pass

    def _bump_stat(self, key: str) -> None:
        data = self._load_stats()
        data[key] = int(data.get(key, 0)) + 1
        self._save_stats(data)


def _now() -> float:
    import time
    return time.time()
