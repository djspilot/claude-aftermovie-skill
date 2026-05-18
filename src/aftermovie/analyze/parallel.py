"""Parallel analyze Module.

The analyze stage used to be a sequential `for f in files: analyze_clip(f)`
loop. On an M5 Pro (10 P-cores), one core saturated while the other nine
sat idle through 61 clips of motion + audio + face + GPMF + sharpness +
exposure + phash. The wall-clock scaled linearly with clip count and the
fans climbed for minutes before render even began.

`parallel_analyze` is the Adapter that swaps the for-loop for a worker
pool while keeping the per-clip `analyze_clip` Interface unchanged. Each
clip is an independent CPU-bound task; the pool picks up the next file
the moment a worker frees up.

This mirrors the precedent set by `render.parallel.parallel_prerender`
(Phase B4) but lives in `analyze/` to keep the analyze Module's
dependencies independent of `render/`.

Interface
=========

    infos = parallel_analyze(
        items,                  # list of opaque items (typically Path or
                                #   (path, origin_still) tuples)
        analyze_fn,             # Callable[item, ClipInfo | None]
        max_workers=N,          # pool size (>= 1)
        progress_cb=cb,         # optional callable(idx, total, item, elapsed_s)
    )

Returns a list of `ClipInfo | None` aligned with `items`. Failures land
as `None` in their slot — the caller filters them out, same as the
sequential loop did with `if info is None: continue`.

`analyze_fn` MUST be picklable. The production caller passes
`aftermovie.analyze.clip._analyze_clip_for_pool` which is a module-level
function — not a closure — so it pickles cleanly to ProcessPoolExecutor
workers. Items are passed as `(path_str, origin_still_str_or_None)`
tuples so the worker process (which doesn't see `_STILL_ORIGIN`) has
enough context to analyze stills correctly.

Invariants
==========

  * `len(results) == len(paths)`; order matches input.
  * `max_workers=1` short-circuits the pool and runs inline. Useful for
    `AFTERMOVIE_ANALYZE_WORKERS=1` deterministic test runs.
  * `ProcessPoolExecutor` is preferred (avoids GIL contention with cv2
    and ffmpeg subprocesses). Falls back to `ThreadPoolExecutor` if the
    process pool can't start (some macOS sandboxes block `fork`/`spawn`).

Workers selection (`choose_max_workers`)
========================================

      Heuristic:  max(2, perf_cores - 2)        clamped to [1, MAX_WORKER_CAP]
      Override:   AFTERMOVIE_ANALYZE_WORKERS=N  (int >= 1)

The `-2` reserve keeps the foreground UI / browser responsive while the
pool is hot. The `max(2, ...)` floor means even a 2-core box gets a
small amount of parallelism.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from aftermovie.ffmpeg_cmd import log

if TYPE_CHECKING:  # pragma: no cover
    from aftermovie.render.chip import ChipInfo
    from aftermovie.types import ClipInfo


# Hard cap regardless of chip/env. Above ~12 workers the orchestrator
# (and RSS pressure from concurrent cv2 + ffmpeg subprocesses) starts
# eating the gains; 12 is the empirical knee on M-series chips.
MAX_WORKER_CAP = 12


def choose_max_workers(
    chip: "ChipInfo",
    *,
    env_override: str | None = None,
) -> int:
    """Pick a worker count for the analyze pool.

    Heuristic: `max(2, perf_cores - 2)`, clamped to [1, MAX_WORKER_CAP].

    `env_override` (or `AFTERMOVIE_ANALYZE_WORKERS` when None is passed)
    takes precedence. A non-numeric override falls through to the
    heuristic — we never raise mid-pipeline.
    """
    raw = (env_override if env_override is not None
           else os.environ.get("AFTERMOVIE_ANALYZE_WORKERS", ""))
    if raw:
        try:
            n = int(str(raw).strip())
            if n >= 1:
                return min(n, MAX_WORKER_CAP)
        except ValueError:
            pass

    cores = int(getattr(chip, "perf_cores", 0) or 0)
    n = max(2, cores - 2) if cores > 0 else 2
    return max(1, min(n, MAX_WORKER_CAP))


def _build_executor(max_workers: int):
    """Prefer ProcessPoolExecutor, fall back to ThreadPoolExecutor.

    The fallback keeps the analyze stage functional in sandboxed
    environments where `fork`/`spawn` is restricted; thread-based
    parallelism still helps because every analyzer spends most of its
    time in ffmpeg / cv2 / mediapipe C code that releases the GIL.
    """
    try:
        return ProcessPoolExecutor(max_workers=max_workers)
    except (OSError, RuntimeError, ValueError):
        return ThreadPoolExecutor(max_workers=max_workers)


ProgressCb = Callable[[int, int, Any, float], None]


def _item_label(item: Any) -> str:
    """Render an item for log messages — Paths show their name, tuples
    pick the first Path-shaped element, anything else stringifies."""
    if isinstance(item, Path):
        return item.name
    if isinstance(item, tuple) and item:
        head = item[0]
        if isinstance(head, Path):
            return head.name
        if isinstance(head, str):
            return Path(head).name
    return str(item)


def parallel_analyze(
    items: list[Any],
    analyze_fn: Callable[[Any], "ClipInfo | None"],
    *,
    max_workers: int,
    progress_cb: ProgressCb | None = None,
) -> list["ClipInfo | None"]:
    """Run `analyze_fn` over `items` in parallel; preserve input order.

    `progress_cb(idx, total, item, elapsed_s)` fires per-completion on
    the main thread. Use it for live logging — the production caller
    emits `analyzed GH012799.MP4 (1.3s, GPMF)` style lines from this
    hook.

    Failures (worker raised, or `analyze_fn` returned `None`) land as
    `None` in the output list. The caller filters them downstream.
    """
    n = len(items)
    if n == 0:
        return []

    if max_workers <= 1:
        return _run_sequential(items, analyze_fn, progress_cb=progress_cb)

    results: list["ClipInfo | None"] = [None] * n
    submit_times: dict[int, float] = {}

    with _build_executor(max_workers) as ex:
        future_to_slot: dict[Future, tuple[int, Any]] = {}
        for idx, it in enumerate(items):
            submit_times[idx] = time.time()
            fut = ex.submit(analyze_fn, it)
            future_to_slot[fut] = (idx, it)

        for fut in as_completed(future_to_slot):
            slot_idx, item = future_to_slot[fut]
            try:
                info = fut.result()
            except Exception as exc:  # noqa: BLE001 — any worker error is a clip-skip
                log(f"  ! analyze worker failed for {_item_label(item)}: {exc}")
                info = None
            results[slot_idx] = info
            if progress_cb is not None:
                elapsed = time.time() - submit_times.get(slot_idx, time.time())
                progress_cb(slot_idx + 1, n, item, elapsed)

    return results


def _run_sequential(
    items: list[Any],
    analyze_fn: Callable[[Any], "ClipInfo | None"],
    *,
    progress_cb: ProgressCb | None,
) -> list["ClipInfo | None"]:
    """Sequential fallback. Same contract as `parallel_analyze`.

    Triggered when `max_workers <= 1` (env override or single-clip
    batches). Runs inline so per-clip logs stay in submit order.
    """
    n = len(items)
    results: list["ClipInfo | None"] = [None] * n
    for idx, it in enumerate(items):
        t0 = time.time()
        info: "ClipInfo | None" = None
        try:
            info = analyze_fn(it)
        except Exception as exc:  # noqa: BLE001
            log(f"  ! analyze failed for {_item_label(it)}: {exc}")
            info = None
        results[idx] = info
        if progress_cb is not None:
            progress_cb(idx + 1, n, it, time.time() - t0)
    return results


__all__ = [
    "MAX_WORKER_CAP",
    "choose_max_workers",
    "parallel_analyze",
]
