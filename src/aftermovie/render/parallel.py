"""Parallel prerender Module.

Phase B4: the renderer's prerender phase used to be a sequential
`for entry in entries: _prerender_clip(entry, ...)` loop. On Apple Silicon
the per-clip ffmpeg subprocess only saturates one media engine (HEVC_VT)
or a couple of CPU cores (x264), so the rest of the chip sat idle while
the loop chewed through 20–30 clips.

`parallel_prerender` is the Adapter that swaps the for-loop for a worker
pool while keeping the existing `_prerender_clip` Interface unchanged.
Each prerender becomes an independent task; the pool picks up the next
entry the moment a worker frees up.

Interface
=========

    paths = parallel_prerender(
        work,                   # list[(entry_dict, out_clip_path)] in plan order
        render_factory,         # Callable[(entry, out_clip, idx, total), bool]
        max_workers=N,          # workers in the pool (>=1)
        progress_cb=cb,         # optional ProgressCallback, fires per completion
    )

`render_factory` is the per-task closure that wraps `_prerender_clip` with
the encoder/aspect/etc. bound. The factory is the Seam that keeps this
Module ignorant of the renderer's flag soup — it sees only "function from
(entry, path, idx, total) to bool".

Invariants
==========

  * `len(paths) == len(work)` — the returned list mirrors plan order
    exactly. Failed prerenders surface as `None` in their slot; the
    caller filters them out (matches the legacy `if not ok: continue`).
  * Per-completion progress events fire on the **main thread / caller's
    thread**, never inside a worker process — `progress_cb` only needs
    to be thread-safe (not multiprocessing-safe). Aggregation is by
    completed-clip count (option (a) in the design doc): each completion
    emits `fraction_in_stage = (k_done) / N`. Per-frame ffmpeg progress
    inside workers is **not** plumbed back across the process boundary;
    that's option (b) and tracked as future work — the per-frame ticks
    are too noisy to justify the multiprocessing.Queue plumbing when 30
    clip-grained ticks cover the bar adequately.
  * Plan order is preserved by indexing futures back to their slot via
    a dict mapping `Future -> slot_idx`, not by `executor.map` (which
    forces strict iteration order and would serialize completion
    notifications behind the slowest clip).
  * `ProcessPoolExecutor` is preferred so each ffmpeg subprocess runs
    under its own Python parent — keeps OS scheduling clean and avoids
    GIL contention in the orchestrator. We fall back to
    `ThreadPoolExecutor` if the process pool can't start (Windows
    without freeze_support, restricted sandboxes); ffmpeg is the heavy
    worker either way so the difference is small.

Workers selection (`choose_max_workers`)
========================================

      Encoder         Heuristic                                Cap
      --------------------------------------------------------------
      hevc_vt/h264_vt media_engines * 2 (async HEVC pipelines) 8
      x264 (CPU)      max(1, perf_cores // 2)                  8

`AFTERMOVIE_RENDER_WORKERS` overrides the heuristic; integer ≥ 1.
"""
from __future__ import annotations

import os
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from aftermovie.ffmpeg_cmd import log

if TYPE_CHECKING:  # pragma: no cover - import-cycle guard
    from aftermovie.render.chip import ChipInfo
    from aftermovie.render.encoder import EncoderProfile
    from aftermovie.render.pipeline import ProgressCallback, ProgressEvent


# Hard upper bound on the pool size regardless of chip/env. Above 8, the
# orchestrator itself becomes a bottleneck (queue thrash, RSS pressure
# from 8 concurrent ffmpeg subprocesses each holding ~150-200 MB) and
# returns diminish.
MAX_WORKER_CAP = 8


def choose_max_workers(
    chip: "ChipInfo",
    encoder: "EncoderProfile",
    *,
    env_override: str | None = None,
) -> int:
    """Pick a worker count for the prerender pool.

    See the module docstring's Workers selection table. The env override
    takes precedence — but a bogus value (negative, non-numeric) falls
    through to the heuristic rather than raising mid-render.
    """
    raw = (env_override if env_override is not None
           else os.environ.get("AFTERMOVIE_RENDER_WORKERS", ""))
    if raw:
        try:
            n = int(str(raw).strip())
            if n >= 1:
                return min(n, MAX_WORKER_CAP)
        except ValueError:
            pass

    if encoder.is_hardware:
        engines = max(1, int(getattr(chip, "media_engines", 0) or 0))
        n = engines * 2
    else:
        cores = max(1, int(getattr(chip, "perf_cores", 0) or 0))
        n = max(1, cores // 2)
    return max(1, min(n, MAX_WORKER_CAP))


# The future-result-to-slot mapping lives in a small dataclass-less tuple
# so the orchestrator stays close to the bare `as_completed` Interface.
_WorkItem = tuple[int, dict, Path]


def _build_executor(max_workers: int):
    """Prefer ProcessPoolExecutor; fall back to ThreadPoolExecutor.

    Some environments (Windows without `if __name__ == '__main__'`,
    sandbox runners that block `fork`/`spawn`) can't start a process
    pool. The fallback keeps the renderer functional, trading per-worker
    process isolation for GIL contention — fine when ffmpeg is the real
    workhorse.
    """
    try:
        return ProcessPoolExecutor(max_workers=max_workers)
    except (OSError, RuntimeError, ValueError):
        return ThreadPoolExecutor(max_workers=max_workers)


def parallel_prerender(
    work: list[tuple[dict, Path]],
    render_factory: Callable[[dict, Path, int, int], bool],
    *,
    max_workers: int,
    progress_cb: "ProgressCallback | None" = None,
    encoder_name: str = "",
) -> list[Path | None]:
    """Run `render_factory` over `work` in parallel, return paths in plan order.

    `work` is a list of `(entry_dict, out_clip_path)` pairs. The factory is
    invoked as `factory(entry, out_clip, stage_index, stage_total)` and
    must return True on success, False on failure. The factory must be
    picklable when ProcessPoolExecutor is used — `functools.partial`
    around a module-level function is fine.

    Returns a list of `Path | None` aligned with `work`: success → the
    `out_clip_path` the factory was given, failure → `None`. The caller
    filters Nones and keeps the surviving paths in plan order.
    """
    # Deferred import: pipeline.py imports parallel.py, so we keep
    # ProgressEvent import lazy to avoid circulars during module load.
    from aftermovie.render.pipeline import ProgressEvent

    n = len(work)
    if n == 0:
        return []

    # `max_workers=1` short-circuits the pool overhead entirely. Useful for
    # AFTERMOVIE_RENDER_WORKERS=1 (deterministic CI runs) and as a fast
    # path for plans with a single clip.
    if max_workers <= 1:
        return _run_sequential(work, render_factory, progress_cb=progress_cb,
                               encoder_name=encoder_name)

    results: list[Path | None] = [None] * n
    completed = 0
    label = f"prerender (parallel, {max_workers} workers" + (
        f", encoder={encoder_name})" if encoder_name else ")"
    )

    import time

    with _build_executor(max_workers) as ex:
        future_to_slot: dict[Future, _WorkItem] = {}
        submit_times: dict[int, float] = {}
        for idx, (entry, out_clip) in enumerate(work):
            submit_times[idx] = time.time()
            fut = ex.submit(render_factory, entry, out_clip, idx + 1, n)
            future_to_slot[fut] = (idx, entry, out_clip)

        for fut in as_completed(future_to_slot):
            slot_idx, _entry, out_clip = future_to_slot[fut]
            try:
                ok = fut.result()
            except Exception as exc:  # noqa: BLE001 — any worker error is a clip-skip
                log(f"  ! prerender worker failed slot {slot_idx + 1}/{n}: {exc}")
                ok = False
            if ok:
                results[slot_idx] = out_clip
            completed += 1
            elapsed = time.time() - submit_times.get(slot_idx, time.time())
            log(f"{label} — clip {completed}/{n} done in {elapsed:.2f}s")
            if progress_cb is not None:
                # Coalesced clip-granularity event. The consumer maps
                # (stage_index, stage_total, fraction) onto an overall %;
                # `fraction_in_stage=1.0` with `stage_index=completed`
                # advances the bar exactly one Nth per finished clip.
                progress_cb(ProgressEvent(
                    stage="prerender",
                    stage_index=completed,
                    stage_total=n,
                    fraction_in_stage=1.0,
                    current_pid=None,
                ))

    return results


def _run_sequential(
    work: list[tuple[dict, Path]],
    render_factory: Callable[[dict, Path, int, int], bool],
    *,
    progress_cb: "ProgressCallback | None",
    encoder_name: str,
) -> list[Path | None]:
    """Sequential fallback. Same contract as `parallel_prerender`.

    Used for `max_workers=1` (env override or single-clip plans) — runs
    inline on the caller's thread so the legacy progress per-clip path
    in `_prerender_clip` keeps emitting frame-granular events.
    """
    from aftermovie.render.pipeline import ProgressEvent

    n = len(work)
    results: list[Path | None] = [None] * n
    for idx, (entry, out_clip) in enumerate(work):
        ok = False
        try:
            ok = render_factory(entry, out_clip, idx + 1, n)
        except Exception as exc:  # noqa: BLE001
            log(f"  ! prerender failed slot {idx + 1}/{n}: {exc}")
            ok = False
        if ok:
            results[idx] = out_clip
        if progress_cb is not None:
            progress_cb(ProgressEvent(
                stage="prerender",
                stage_index=idx + 1,
                stage_total=n,
                fraction_in_stage=1.0,
                current_pid=None,
            ))
    if encoder_name:
        log(f"prerender (sequential, encoder={encoder_name}) — {n} clips done")
    return results


__all__ = [
    "MAX_WORKER_CAP",
    "choose_max_workers",
    "parallel_prerender",
]
