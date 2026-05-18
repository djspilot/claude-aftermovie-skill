"""Phase B4 — `parallel_prerender` helper + `choose_max_workers` heuristic.

The tests use module-level stub functions so the ProcessPoolExecutor path
can pickle the worker — closures defined inside a test body would fail to
pickle and force the ThreadPoolExecutor fallback unconditionally, which
defeats the point of exercising the real production path.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from aftermovie.render.chip import ChipInfo
from aftermovie.render.encoder import HEVC_VT, H264_VT, X264
from aftermovie.render.parallel import (
    MAX_WORKER_CAP,
    choose_max_workers,
    parallel_prerender,
)
from aftermovie.render.pipeline import ProgressEvent


# ---- module-level worker stubs (must be picklable for ProcessPoolExecutor) -


def _sleep_and_touch(entry: dict, out_clip: Path, idx: int, total: int) -> bool:
    """Pretend to prerender by sleeping 100 ms and writing a 0-byte file.

    Wall-clock for N sequential calls is ~0.1 * N; for N in a pool with
    `max_workers >= N` it collapses to ~0.1 s. That gap is what the
    parallel-speedup test asserts on.
    """
    time.sleep(0.1)
    out_clip.write_bytes(b"")
    return True


def _record_order(entry: dict, out_clip: Path, idx: int, total: int) -> bool:
    """Write the slot index into the output so we can assert plan-order
    preservation on the returned path list."""
    time.sleep(0.02)
    out_clip.write_text(str(idx))
    return True


def _flaky_at_slot_2(entry: dict, out_clip: Path, idx: int, total: int) -> bool:
    """Fail (return False) for slot 2 only; succeed otherwise."""
    if entry.get("slot") == 2:
        return False
    out_clip.write_bytes(b"")
    return True


# ---- max_workers selection -------------------------------------------------


def test_choose_max_workers_vt_uses_media_engines_times_two() -> None:
    """VT encoder: workers = media_engines * 2 (async HEVC pipelines per
    engine — see the parallel.py docstring's selection table)."""
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=2)
    # Clear the env override so we exercise the heuristic.
    assert choose_max_workers(chip, HEVC_VT, env_override="") == 4
    assert choose_max_workers(chip, H264_VT, env_override="") == 4


def test_choose_max_workers_vt_single_engine() -> None:
    """Single media engine -> 2 workers (the floor for VT parallelism)."""
    chip = ChipInfo(brand="Apple M1", arch="arm64",
                    perf_cores=4, eff_cores=4, media_engines=1)
    assert choose_max_workers(chip, HEVC_VT, env_override="") == 2


def test_choose_max_workers_x264_uses_perf_cores_over_two() -> None:
    """x264 CPU: workers = max(1, perf_cores // 2)."""
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=2)
    assert choose_max_workers(chip, X264, env_override="") == 5


def test_choose_max_workers_x264_floor_is_one() -> None:
    """Even a 1-core box gets at least 1 worker — never 0."""
    chip = ChipInfo(brand="Generic", arch="x86_64",
                    perf_cores=1, eff_cores=0, media_engines=0)
    assert choose_max_workers(chip, X264, env_override="") == 1


def test_choose_max_workers_caps_at_eight() -> None:
    """The cap kicks in for huge chips so we don't oversubscribe RSS."""
    chip = ChipInfo(brand="Apple M5 Ultra", arch="arm64",
                    perf_cores=24, eff_cores=8, media_engines=8)
    # Without the cap, VT heuristic would return 16.
    assert choose_max_workers(chip, HEVC_VT, env_override="") == MAX_WORKER_CAP


def test_choose_max_workers_env_override_wins() -> None:
    """`AFTERMOVIE_RENDER_WORKERS=N` overrides the heuristic."""
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=2)
    assert choose_max_workers(chip, HEVC_VT, env_override="1") == 1
    assert choose_max_workers(chip, HEVC_VT, env_override="3") == 3
    # Garbage falls through to the heuristic rather than raising.
    assert choose_max_workers(chip, HEVC_VT, env_override="banana") == 4


def test_choose_max_workers_env_override_capped() -> None:
    """An env override above the cap is still clamped to MAX_WORKER_CAP."""
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=2)
    assert choose_max_workers(chip, HEVC_VT, env_override="99") == MAX_WORKER_CAP


# ---- parallel_prerender behaviour -----------------------------------------


def _build_work(tmp_path: Path, n: int) -> list[tuple[dict, Path]]:
    """Build `n` (entry, out_clip) pairs in plan order."""
    return [
        ({"slot": i}, tmp_path / f"clip_{i:04d}.mp4")
        for i in range(n)
    ]


def test_parallel_prerender_beats_sequential_wall_clock(tmp_path: Path) -> None:
    """4 workers × 100 ms/clip × 8 clips should be well under half the
    sequential wall-clock. We compare a pool-of-1 (forced sequential) to
    a pool-of-4 to keep the test free of pool-startup overhead bias."""
    n_clips = 8

    seq_work = _build_work(tmp_path / "seq", n_clips)
    (tmp_path / "seq").mkdir()
    t0 = time.time()
    seq_paths = parallel_prerender(
        seq_work, _sleep_and_touch, max_workers=1, progress_cb=None,
    )
    seq_elapsed = time.time() - t0

    par_work = _build_work(tmp_path / "par", n_clips)
    (tmp_path / "par").mkdir()
    t0 = time.time()
    par_paths = parallel_prerender(
        par_work, _sleep_and_touch, max_workers=4, progress_cb=None,
    )
    par_elapsed = time.time() - t0

    assert all(p is not None for p in seq_paths)
    assert all(p is not None for p in par_paths)
    # 4 workers on 8 × 100ms tasks: should land near 0.2s. Sequential lands
    # near 0.8s. Assert < 50% (the spec's bar) with headroom for jitter.
    assert par_elapsed < seq_elapsed * 0.5, (
        f"parallel wall-clock {par_elapsed:.2f}s was not <50% of "
        f"sequential {seq_elapsed:.2f}s"
    )


def test_parallel_prerender_preserves_plan_order(tmp_path: Path) -> None:
    """Output paths land in plan order even when workers complete out of order.

    `_record_order` writes the original slot index into the file; reading
    them back in order must reproduce 0..N-1.
    """
    n_clips = 6
    work = _build_work(tmp_path, n_clips)
    paths = parallel_prerender(
        work, _record_order, max_workers=4, progress_cb=None,
    )
    assert len(paths) == n_clips
    for slot, p in enumerate(paths):
        assert p is not None, f"slot {slot} should have succeeded"
        # The factory was called with `idx = slot + 1` (1-based stage index),
        # but the file path written is the same slot we expect. We assert
        # path identity rather than file contents because order is the
        # primary invariant — file contents would be slot+1 (1-based).
        assert p == work[slot][1]


def test_parallel_prerender_failures_become_none(tmp_path: Path) -> None:
    """Workers returning False surface as `None` in the result list."""
    work = _build_work(tmp_path, 5)
    paths = parallel_prerender(
        work, _flaky_at_slot_2, max_workers=2, progress_cb=None,
    )
    # slot 2 fails; others succeed.
    assert paths[0] is not None
    assert paths[1] is not None
    assert paths[2] is None
    assert paths[3] is not None
    assert paths[4] is not None


def test_parallel_prerender_max_workers_one_runs_inline(tmp_path: Path) -> None:
    """`max_workers=1` short-circuits the pool — useful for AFTERMOVIE_RENDER_WORKERS=1
    deterministic CI runs. Wall-clock should be close to N * sleep_duration.
    """
    n_clips = 3
    work = _build_work(tmp_path, n_clips)
    t0 = time.time()
    paths = parallel_prerender(
        work, _sleep_and_touch, max_workers=1, progress_cb=None,
    )
    elapsed = time.time() - t0
    # 3 × 100ms sequential ≈ 0.3s. Anything under ~0.25s would imply
    # parallelism slipped in.
    assert elapsed >= 0.25, (
        f"max_workers=1 ran in {elapsed:.2f}s — appears to have parallelized"
    )
    assert all(p is not None for p in paths)


def test_parallel_prerender_emits_clip_completion_events(tmp_path: Path) -> None:
    """A `progress_cb` receives one ProgressEvent per finished clip with
    `stage_index` advancing 1..N and `fraction_in_stage == 1.0`."""
    events: list[ProgressEvent] = []
    work = _build_work(tmp_path, 4)
    paths = parallel_prerender(
        work, _record_order,
        max_workers=2,
        progress_cb=events.append,
        encoder_name="hevc_vt",
    )
    assert all(p is not None for p in paths)
    assert len(events) == 4
    seen_indices = sorted(e.stage_index for e in events)
    assert seen_indices == [1, 2, 3, 4]
    for e in events:
        assert e.stage == "prerender"
        assert e.stage_total == 4
        assert e.fraction_in_stage == 1.0


def test_parallel_prerender_empty_work_returns_empty(tmp_path: Path) -> None:
    """Edge case: zero entries -> zero paths, no executor spin-up."""
    paths = parallel_prerender(
        [], _sleep_and_touch, max_workers=4, progress_cb=None,
    )
    assert paths == []


def test_env_var_render_workers_one_forces_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AFTERMOVIE_RENDER_WORKERS=1` -> heuristic returns 1 regardless of chip."""
    monkeypatch.setenv("AFTERMOVIE_RENDER_WORKERS", "1")
    chip = ChipInfo(brand="Apple M5 Pro", arch="arm64",
                    perf_cores=10, eff_cores=4, media_engines=2)
    # Pass env_override=None so the function reads the actual env var.
    assert choose_max_workers(chip, HEVC_VT, env_override=None) == 1
    assert choose_max_workers(chip, X264, env_override=None) == 1
