"""Tests for `parallel_analyze` + `choose_max_workers` (analyze pool).

Module-level worker stubs so ProcessPoolExecutor can actually pickle the
callable — closures defined inside a test body would fail to pickle and
silently fall back to ThreadPoolExecutor, which would mask production
bugs in the spawn path.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from aftermovie.analyze.parallel import (
    MAX_WORKER_CAP,
    choose_max_workers,
    parallel_analyze,
)
from aftermovie.render.chip import ChipInfo


# ---- module-level worker stubs (picklable for ProcessPoolExecutor) --------


def _sleep_100ms(item: Path) -> dict:
    """Pretend to analyze: sleep 100 ms, return a marker dict."""
    time.sleep(0.1)
    return {"path": str(item)}


def _echo_item(item: Path) -> dict:
    """Return the item unchanged so the caller can assert order-preservation."""
    return {"path": str(item)}


def _raises(item: Path) -> dict:
    """Always raise — exercises the worker-failure → None path."""
    raise RuntimeError(f"intentional failure for {item.name}")


def _none_for_slot_2(item: Path) -> dict | None:
    """Returns None for path containing '_002' — exercises the None
    propagation through the result list."""
    if "_002" in str(item):
        return None
    return {"path": str(item)}


# ---- choose_max_workers ----------------------------------------------------


def test_choose_max_workers_perf_minus_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """An M5 Pro (10 P-cores) should get 8 workers under the heuristic."""
    monkeypatch.delenv("AFTERMOVIE_ANALYZE_WORKERS", raising=False)
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=1)
    assert choose_max_workers(chip, env_override="") == 8


def test_choose_max_workers_floor_is_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2-core box still gets a worker pool of size 2 (the floor)."""
    monkeypatch.delenv("AFTERMOVIE_ANALYZE_WORKERS", raising=False)
    chip = ChipInfo(brand="Tiny", arch="arm64",
                    perf_cores=2, eff_cores=0, media_engines=0)
    # max(2, 2-2) = max(2, 0) = 2
    assert choose_max_workers(chip, env_override="") == 2


def test_choose_max_workers_caps_at_twelve(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 24-core Ultra chip is clamped to MAX_WORKER_CAP (12)."""
    monkeypatch.delenv("AFTERMOVIE_ANALYZE_WORKERS", raising=False)
    chip = ChipInfo(brand="Apple M5 Ultra", arch="arm64",
                    perf_cores=24, eff_cores=8, media_engines=4)
    # max(2, 24-2) = 22, clamped to 12.
    assert choose_max_workers(chip, env_override="") == MAX_WORKER_CAP
    assert MAX_WORKER_CAP == 12


def test_choose_max_workers_env_override_one_forces_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AFTERMOVIE_ANALYZE_WORKERS=1` returns 1 regardless of chip."""
    monkeypatch.setenv("AFTERMOVIE_ANALYZE_WORKERS", "1")
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=1)
    # env_override=None means "read AFTERMOVIE_ANALYZE_WORKERS from os.environ"
    assert choose_max_workers(chip, env_override=None) == 1


def test_choose_max_workers_env_override_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even a wild env override clamps to MAX_WORKER_CAP."""
    monkeypatch.delenv("AFTERMOVIE_ANALYZE_WORKERS", raising=False)
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=1)
    assert choose_max_workers(chip, env_override="999") == MAX_WORKER_CAP


def test_choose_max_workers_env_garbage_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-numeric env values shouldn't raise; the heuristic wins."""
    monkeypatch.delenv("AFTERMOVIE_ANALYZE_WORKERS", raising=False)
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=1)
    assert choose_max_workers(chip, env_override="banana") == 8


# ---- parallel_analyze ------------------------------------------------------


def _paths(n: int, root: Path) -> list[Path]:
    """Build n distinct Path objects under `root` (files don't need to exist
    — the stub workers never touch the filesystem)."""
    return [root / f"clip_{i:03d}.mp4" for i in range(n)]


def test_parallel_analyze_beats_sequential_wall_clock(tmp_path: Path) -> None:
    """16 clips × 100 ms with workers=4 should land well under 40% of the
    sequential wall-clock — the spec's bar. We use 16 clips (not 8) so
    that ProcessPool spawn / pickle overhead amortizes out and the
    parallel-speedup signal is dominated by actual compute, not pool
    startup."""
    n = 16

    seq_paths = _paths(n, tmp_path)
    t0 = time.time()
    seq_results = parallel_analyze(
        seq_paths, _sleep_100ms, max_workers=1, progress_cb=None,
    )
    seq_elapsed = time.time() - t0

    par_paths = _paths(n, tmp_path)
    t0 = time.time()
    par_results = parallel_analyze(
        par_paths, _sleep_100ms, max_workers=4, progress_cb=None,
    )
    par_elapsed = time.time() - t0

    assert all(r is not None for r in seq_results)
    assert all(r is not None for r in par_results)
    # Spec: parallel wall-clock < 40% of sequential. 16 × 0.1s sequential
    # ≈ 1.6s; 4 workers should land near 0.4s + pool spawn (~0.2s) ≈ 0.6s,
    # comfortably under the 0.64s bar.
    assert par_elapsed < seq_elapsed * 0.4, (
        f"parallel wall-clock {par_elapsed:.2f}s was not <40% of "
        f"sequential {seq_elapsed:.2f}s"
    )


def test_parallel_analyze_preserves_input_order(tmp_path: Path) -> None:
    """Output list slots correspond to input order even when workers complete
    out of order."""
    n = 6
    paths = _paths(n, tmp_path)
    results = parallel_analyze(
        paths, _echo_item, max_workers=4, progress_cb=None,
    )
    assert len(results) == n
    for i, r in enumerate(results):
        assert r is not None
        assert r["path"] == str(paths[i])


def test_parallel_analyze_worker_failures_become_none(tmp_path: Path) -> None:
    """A worker raising must NOT propagate — the slot fills with None and
    the caller drops it from the catalog downstream."""
    paths = _paths(3, tmp_path)
    results = parallel_analyze(
        paths, _raises, max_workers=2, progress_cb=None,
    )
    assert results == [None, None, None]


def test_parallel_analyze_none_returns_pass_through(tmp_path: Path) -> None:
    """A worker returning None (e.g. probe-failed clip) lands as None in the
    aligned result list."""
    paths = _paths(5, tmp_path)
    results = parallel_analyze(
        paths, _none_for_slot_2, max_workers=2, progress_cb=None,
    )
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None  # clip_002
    assert results[3] is not None
    assert results[4] is not None


def test_parallel_analyze_empty_input_returns_empty(tmp_path: Path) -> None:
    """Zero items → zero results, no pool spin-up."""
    assert parallel_analyze([], _echo_item, max_workers=4) == []


def test_parallel_analyze_max_workers_one_runs_inline(tmp_path: Path) -> None:
    """`max_workers=1` short-circuits the pool — useful for deterministic CI runs."""
    n = 3
    paths = _paths(n, tmp_path)
    t0 = time.time()
    results = parallel_analyze(
        paths, _sleep_100ms, max_workers=1, progress_cb=None,
    )
    elapsed = time.time() - t0
    # 3 × 100ms sequential ≈ 0.3s. Anything below ~0.25s implies parallelism.
    assert elapsed >= 0.25, (
        f"max_workers=1 ran in {elapsed:.2f}s — appears to have parallelized"
    )
    assert all(r is not None for r in results)


def test_parallel_analyze_progress_cb_fires_per_completion(tmp_path: Path) -> None:
    """`progress_cb` receives one event per finished clip with idx 1..N."""
    events: list[tuple[int, int, Path, float]] = []

    def cb(idx: int, total: int, item, elapsed: float) -> None:
        events.append((idx, total, item, elapsed))

    paths = _paths(4, tmp_path)
    parallel_analyze(paths, _echo_item, max_workers=2, progress_cb=cb)

    assert len(events) == 4
    indices = sorted(e[0] for e in events)
    assert indices == [1, 2, 3, 4]
    for _idx, total, _item, elapsed in events:
        assert total == 4
        assert elapsed >= 0.0


def test_env_var_analyze_workers_one_forces_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end env contract: `AFTERMOVIE_ANALYZE_WORKERS=1` → sequential."""
    monkeypatch.setenv("AFTERMOVIE_ANALYZE_WORKERS", "1")
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=1)
    workers = choose_max_workers(chip, env_override=None)
    assert workers == 1

    # And feeding that 1 into `parallel_analyze` runs sequentially.
    n = 3
    paths = _paths(n, tmp_path)
    t0 = time.time()
    results = parallel_analyze(
        paths, _sleep_100ms, max_workers=workers, progress_cb=None,
    )
    elapsed = time.time() - t0
    assert elapsed >= 0.25
    assert len(results) == n
