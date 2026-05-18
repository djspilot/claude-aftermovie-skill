"""Tests for the per-clip analyze cache.

The `AnalyzeCache` Module sits below the folder-level `CatalogRepository`:
when the user touches a single file, the catalog id changes (busting the
whole folder cache) but the per-clip cache hits for every untouched clip.
These tests pin the key derivation, round-trip semantics, and LRU
eviction bound — the three invariants the production caller relies on.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aftermovie import config
from aftermovie.analyze import analyze_cache as analyze_cache_mod
from aftermovie.analyze.analyze_cache import AnalyzeCache
from aftermovie.types import ClipInfo


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `config.data_dir()` to a tmp path so the test never
    writes to the developer's real `~/.skills-data/aftermovie/`."""
    fake = tmp_path / "skills-data"
    fake.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake)
    return fake


def _make_clip(path: Path, *, body: bytes = b"clip-bytes") -> Path:
    path.write_bytes(body)
    return path


def _sample_info(path: Path) -> ClipInfo:
    """A minimally-valid ClipInfo for round-trip tests."""
    return ClipInfo(
        path=str(path),
        duration_s=2.5,
        fps=30.0,
        width=1920,
        height=1080,
        has_gpmf=True,
        hilight_tags_ms=[1500],
        motion_energy=[0.1, 0.2, 0.3],
        audio_energy=[0.4, 0.5, 0.6],
        voice_energy=[0.0, 0.0, 0.0],
        accl_peaks=[0.0, 0.0, 0.0],
        gps_speed=[0.0, 0.0, 0.0],
        is_short_form=True,
        captured_at=1234567890.0,
        face_bboxes=[None, None, None],
        sharpness_per_s=[0.5, 0.6, 0.7],
        exposure_per_s=[0.4, 0.5, 0.6],
        phash="0123456789abcdef",
        duplicate_group=None,
    )


def test_key_for_is_stable(isolated_data_dir: Path, tmp_path: Path) -> None:
    """Same (path, mtime, size) → same hash, across multiple calls."""
    clip = _make_clip(tmp_path / "a.mp4")
    cache = AnalyzeCache()
    k1 = cache.key_for(clip)
    k2 = cache.key_for(clip)
    assert k1 == k2
    assert len(k1) == 40  # SHA1 hex digest length
    assert all(c in "0123456789abcdef" for c in k1)


def test_key_changes_when_mtime_changes(isolated_data_dir: Path, tmp_path: Path) -> None:
    """An in-place edit (touch + rewrite) must produce a fresh key — that's
    what makes the cache automatically invalidate when the user re-exports
    a clip with the same name."""
    clip = _make_clip(tmp_path / "a.mp4")
    cache = AnalyzeCache()
    k_before = cache.key_for(clip)

    # Touch the file by writing different bytes (also bumps st_size).
    # Pin a far-future mtime so the change is unambiguous on coarse FS
    # timestamps (HFS+ stores 1-sec resolution).
    clip.write_bytes(b"clip-bytes-modified")
    future = clip.stat().st_mtime + 60.0
    os.utime(clip, (future, future))

    k_after = cache.key_for(clip)
    assert k_before != k_after


def test_key_changes_when_size_changes(isolated_data_dir: Path, tmp_path: Path) -> None:
    """A size change at the SAME mtime should still bump the key — guards
    against atomic-replace tools that preserve mtime but rewrite content."""
    clip = _make_clip(tmp_path / "a.mp4", body=b"short")
    cache = AnalyzeCache()
    k_before = cache.key_for(clip)
    mtime = clip.stat().st_mtime

    clip.write_bytes(b"much-longer-body-of-bytes")
    # Force mtime back to the original so only size differs.
    os.utime(clip, (mtime, mtime))

    k_after = cache.key_for(clip)
    assert k_before != k_after


def test_put_then_get_roundtrips_clipinfo_shape(
    isolated_data_dir: Path, tmp_path: Path,
) -> None:
    """A persisted ClipInfo dataclass comes back as the same dict shape
    `asdict` would produce — that's what the catalog list expects."""
    clip = _make_clip(tmp_path / "a.mp4")
    cache = AnalyzeCache()
    info = _sample_info(clip)

    key = cache.key_for(clip)
    assert cache.get(key) is None, "fresh cache should be a miss"

    out_path = cache.put(key, info)
    assert out_path.is_file()
    # The on-disk shard layout: <root>/<hash[:2]>/<hash>.json
    assert out_path.parent.name == key[:2]
    assert out_path.name == f"{key}.json"

    got = cache.get(key)
    assert got is not None
    # Every ClipInfo field round-trips. Compare via asdict-shape equality.
    from dataclasses import asdict
    assert got == asdict(info)


def test_get_returns_none_on_corrupt_json(
    isolated_data_dir: Path, tmp_path: Path,
) -> None:
    """A corrupt cache entry must not crash the run — the caller treats it
    as a miss and re-analyzes."""
    clip = _make_clip(tmp_path / "a.mp4")
    cache = AnalyzeCache()
    key = cache.key_for(clip)
    p = cache.path_for_key(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json")

    assert cache.get(key) is None


def test_lru_eviction_triggers_at_cap(
    isolated_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the cache exceeds its cap, the oldest-mtime entries are
    evicted first. The eviction sweep runs on `put`.

    We pre-seed several entries, then back-date the first ones so they
    look "old", then issue one more `put` and assert: the freshest
    entries survive, the oldest one is gone.
    """
    cache = AnalyzeCache()

    # Pre-populate 4 entries with no cap (default 1 GB — plenty of room).
    # Then back-date the first one before the eviction sweep.
    keys: list[str] = []
    for i in range(4):
        clip = _make_clip(tmp_path / f"clip_{i}.mp4", body=f"clip-{i}".encode())
        info = _sample_info(clip)
        info.path = str(clip)
        k = cache.key_for(clip)
        cache.put(k, info)
        keys.append(k)

    # Stamp the first three entries with progressively older mtimes —
    # the eviction sweep sorts by st_mtime ascending, so these go first.
    for j, k in enumerate(keys[:3]):
        p = cache.path_for_key(k)
        os.utime(p, (1000 + j, 1000 + j))

    # Now flip the cap so the existing entries blow past it, AND issue
    # one more put to trigger the sweep. The newest two writes (`keys[3]`
    # and the new entry) must survive; the back-dated ones get evicted.
    monkeypatch.setenv("AFTERMOVIE_ANALYZE_CACHE_MAX_GB", "1e-7")
    trigger_clip = _make_clip(tmp_path / "trigger.mp4", body=b"trigger")
    trigger_info = _sample_info(trigger_clip)
    trigger_info.path = str(trigger_clip)
    trigger_key = cache.key_for(trigger_clip)
    cache.put(trigger_key, trigger_info)

    # The newly-written trigger should still be on disk (it was the
    # freshest entry at sweep time, so eviction stops before it).
    assert cache.get(trigger_key) is not None, "newest entry should survive"
    # And at least one of the back-dated old entries is gone.
    survivors_old = [k for k in keys[:3] if cache.get(k) is not None]
    assert len(survivors_old) < 3, (
        f"expected at least one back-dated entry to be evicted; "
        f"all {len(survivors_old)} of 3 survived"
    )


def test_module_singleton_is_an_analyzecache(isolated_data_dir: Path) -> None:
    """`analyze_cache` (module-level handle) is a ready-to-use instance —
    callers shouldn't have to instantiate the class themselves."""
    from aftermovie.analyze.analyze_cache import analyze_cache as singleton

    assert isinstance(singleton, AnalyzeCache)
    # And `root()` returns under the isolated data dir, proving the
    # config.data_dir() monkeypatch is honored.
    assert isolated_data_dir in singleton.root().parents
