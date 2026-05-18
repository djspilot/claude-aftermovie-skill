"""Tests for the per-clip prerender cache.

These cover the four Seams the renderer relies on:
  1. `PrerenderCache.key_for` — stable hash recipe (entry + opts).
  2. `put` + `get` — round-trip into / out of the cache directory.
  3. `lookup_or_compute` shape — hit returns cached path; miss returns None
     so the caller's ffmpeg path keeps owning the workdir.
  4. LRU eviction — oldest entry is dropped when the cap is exceeded.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from aftermovie import config
from aftermovie.render.encoder import X264
from aftermovie.render.prerender_cache import PrerenderCache, PrerenderOpts


def _isolated_cache(tmp_path: Path, monkeypatch) -> PrerenderCache:
    """Redirect `config.data_dir()` into the test's tmp tree."""
    fake = tmp_path / "state"
    fake.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake)
    return PrerenderCache()


def _entry_for(src: Path) -> dict:
    """Minimal Entry shape that `key_for` reads from."""
    return {
        "source": str(src),
        "start_s": 0.0,
        "end_s": 1.0,
        "speed": 1.0,
        "out_duration_s": 1.0,
    }


def _opts(target_res: str = "1920x1080") -> PrerenderOpts:
    return PrerenderOpts(
        aspect="16:9",
        target_res=target_res,
        fps=30,
        lut=None,
        encoder=X264,
        keep_audio=False,
        audio_interest_threshold=0.35,
    )


# ---- key_for --------------------------------------------------------------

def test_key_for_stable_for_identical_inputs(tmp_path: Path, monkeypatch) -> None:
    """Same Entry + same Opts → same key on repeated calls."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video bytes")

    k1 = cache.key_for(_entry_for(src), _opts())
    k2 = cache.key_for(_entry_for(src), _opts())
    assert k1 == k2
    # 40-char hex SHA1 by construction.
    assert len(k1) == 40
    assert all(c in "0123456789abcdef" for c in k1)


def test_key_for_differs_when_target_res_differs(
    tmp_path: Path, monkeypatch,
) -> None:
    """Changing `target_res` must produce a different cache key."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video bytes")
    entry = _entry_for(src)

    k_hd = cache.key_for(entry, _opts(target_res="1920x1080"))
    k_4k = cache.key_for(entry, _opts(target_res="3840x2160"))
    assert k_hd != k_4k


def test_key_for_changes_when_source_mtime_changes(
    tmp_path: Path, monkeypatch,
) -> None:
    """Touching the source file invalidates the cache key — guards against
    silent re-encodes of the source under the cache."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video bytes")
    entry = _entry_for(src)

    k_before = cache.key_for(entry, _opts())
    # Bump mtime far enough that integer-ns coercion sees the change even
    # on coarse-grained filesystems.
    new_mtime = time.time() + 60
    os.utime(src, (new_mtime, new_mtime))
    k_after = cache.key_for(entry, _opts())
    assert k_before != k_after


# ---- put / get ------------------------------------------------------------

def test_put_then_get_returns_cached_path(tmp_path: Path, monkeypatch) -> None:
    """put() writes the file under root; get() returns its path."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video bytes")

    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"rendered bytes")

    key = cache.key_for(_entry_for(src), _opts())
    stored = cache.put(key, rendered)
    assert stored.is_file()
    # Sharded path: <root>/<key[:2]>/<key>.mp4
    assert stored.parent.name == key[:2]
    assert stored.name == f"{key}.mp4"

    hit = cache.get(key)
    assert hit == stored


def test_get_returns_none_on_miss(tmp_path: Path, monkeypatch) -> None:
    """No cache file → None (so caller knows to ffmpeg the entry)."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    assert cache.get("never-stored-key") is None


def test_second_get_after_put_still_returns_cached_path(
    tmp_path: Path, monkeypatch,
) -> None:
    """Two get()s after one put() both hit (no implicit eviction)."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"")
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"x")
    key = cache.key_for(_entry_for(src), _opts())
    cache.put(key, rendered)

    first = cache.get(key)
    second = cache.get(key)
    assert first is not None and second is not None
    assert first == second


# ---- lookup_or_compute shape ---------------------------------------------

def test_lookup_or_compute_returns_cached_path_on_hit(
    tmp_path: Path, monkeypatch,
) -> None:
    """When the entry is already in cache, compute_fn must not run."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"")
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"y")
    entry = _entry_for(src)
    opts = _opts()
    key = cache.key_for(entry, opts)
    cache.put(key, rendered)

    calls: list[Path] = []

    def _compute(out: Path) -> bool:
        calls.append(out)
        return True

    hit = cache.lookup_or_compute(entry, opts, _compute)
    assert hit is not None
    assert hit == cache.path_for_key(key)
    assert calls == [], "compute_fn must NOT run on a hit"


def test_lookup_or_compute_returns_none_on_miss(
    tmp_path: Path, monkeypatch,
) -> None:
    """Miss → return None so caller's ffmpeg path keeps owning the workdir.

    The MVP doesn't have `lookup_or_compute` orchestrate the workdir itself;
    `_prerender_clip` does the cache.put() after a successful ffmpeg exit.
    """
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"")
    entry = _entry_for(src)

    def _compute(out: Path) -> bool:
        return True

    assert cache.lookup_or_compute(entry, _opts(), _compute) is None


# ---- LRU eviction --------------------------------------------------------

def test_lru_eviction_drops_oldest_entry_when_over_cap(
    tmp_path: Path, monkeypatch,
) -> None:
    """Writing entries that together exceed the cap evicts the oldest."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    # Cap = 3 KB. Each entry = 2 KB → first two fit, third evicts oldest.
    monkeypatch.setenv("AFTERMOVIE_PRERENDER_CACHE_MAX_GB",
                       f"{(3 * 1024) / (1024 ** 3)}")
    body = b"x" * 2048

    def _write_and_put(idx: int, atime_offset: float) -> tuple[str, Path]:
        rendered = tmp_path / f"r{idx}.mp4"
        rendered.write_bytes(body)
        # Use a unique synthetic key — we don't care about the real entry
        # for eviction, only the size + atime ordering.
        key = f"{'a' * 38}{idx:02d}"  # 40 hex chars
        stored = cache.put(key, rendered)
        os.utime(stored, (atime_offset, atime_offset))
        return key, stored

    # Sleep-free atime ordering: pick explicit timestamps.
    now = time.time()
    k1, p1 = _write_and_put(1, now - 300)  # oldest
    k2, p2 = _write_and_put(2, now - 200)
    # After this put the cache holds 4 KB (>3 KB cap) → eviction runs and
    # drops k1 (oldest atime).
    # But cache.put() runs eviction internally; let's directly add the
    # third entry and check.
    k3, p3 = _write_and_put(3, now - 100)
    # Eviction should have dropped at least k1, may have dropped k2 too
    # depending on whether 2*2KB fits the 3KB cap (it doesn't — only 1
    # entry fits). So we expect only k3 remaining.
    assert p3.is_file(), "newest entry must survive eviction"
    assert not p1.is_file(), "oldest entry must be evicted"


def test_eviction_skips_when_under_cap(tmp_path: Path, monkeypatch) -> None:
    """When total usage stays under the cap, every entry survives."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    # Cap = 1 GB; 3 tiny entries should never touch eviction.
    monkeypatch.setenv("AFTERMOVIE_PRERENDER_CACHE_MAX_GB", "1.0")
    paths: list[Path] = []
    for i in range(3):
        rendered = tmp_path / f"r{i}.mp4"
        rendered.write_bytes(b"x" * 64)
        key = f"{'b' * 38}{i:02d}"
        paths.append(cache.put(key, rendered))
    for p in paths:
        assert p.is_file(), "small entries should never evict"


# ---- stats ----------------------------------------------------------------

def test_stats_tracks_hits_and_misses(tmp_path: Path, monkeypatch) -> None:
    """get() bumps hits on hit, misses on miss; stats() reports both."""
    cache = _isolated_cache(tmp_path, monkeypatch)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"")
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"x")
    entry = _entry_for(src)
    key = cache.key_for(entry, _opts())

    # Miss.
    assert cache.get(key) is None
    # Hit.
    cache.put(key, rendered)
    assert cache.get(key) is not None
    assert cache.get(key) is not None

    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["entry_count"] == 1
    assert s["bytes_used"] >= 1
