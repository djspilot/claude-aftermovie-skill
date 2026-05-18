"""SelectionService — the GUI's domain Interface, sans HTTP.

Before this Module existed, `select.server._Handler` was a god-Module: each
route handler reached straight into disk state (selection sidecar, prefs
sidecar, thumb cache, Plan Repository, render-job dict) and the HTTP layer
was tangled up with the domain ops it dispatched. Tests had to spin a real
HTTP server to assert anything; future surfaces (a CLI, a Textual TUI,
inline scripting) would have re-implemented the same operations.

`SelectionService` lifts those domain operations out behind a small
Interface the GUI consumes. `_Handler` shrinks to a parse-call-serialize
Adapter; `SelectServer` boot wires a service instance into the handler
factory and is otherwise untouched.

Interface:

    svc = SelectionService(clips_root, song_default=None)
    svc.list_sources()                       -> list[SourceRow]
    svc.get_selection()                      -> list[str]
    svc.save_selection(excluded)             -> SaveResult
    svc.get_preferences()                    -> dict
    svc.save_preferences(prefs)              -> SaveResult
    svc.latest_plan()                        -> dict | None
    svc.available_options()                  -> dict
    svc.thumb_for_key(key)                   -> bytes | None
    svc.start_render(body)                   -> RenderJob
    svc.status(job_id)                       -> dict | None

Invariants:

  * The service holds no HTTP state — no request/response, no headers, no
    status codes. Errors surface as exceptions or `None` returns.
  * `start_render` is non-blocking: a worker thread is spawned and the
    `RenderJob` returned immediately. `status(job_id)` is the only way to
    learn the outcome.
  * The render-job dict + its lock are instance attributes (not class
    state) so two SelectionServices in the same process are independent.
  * `list_sources` walks the folder fresh each call; the sidecar reads
    are mtime-cached by `SidecarStore`.
  * `latest_plan` returns `None` (not an exception) when no plan tagged
    with this folder's catalog_id is on disk yet.
"""
from __future__ import annotations

import io
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aftermovie.analyze.capture_time import captured_at_for
from aftermovie.analyze.preferences import (
    load_preferences,
    save_preferences,
)
from aftermovie.analyze.selection import (
    SELECTION_FILENAME,
    load_excluded,
    save_excluded,
)
from aftermovie.analyze.stills import (
    LIVE_PHOTO_VIDEO_EXTS,
    STILL_EXTS,
    _is_excluded_output,
    _under_skipped_dir,
)
from aftermovie.config import THEMES, VIDEO_EXTS, list_luts
from aftermovie.select.thumbnails import _cache_key as thumb_cache_key
from aftermovie.select.thumbnails import _thumbs_cache_dir, thumb_path_for


# ---- value objects ---------------------------------------------------------

@dataclass
class SourceRow:
    """One row the GUI's source grid renders."""

    path: str
    name: str
    kind: str  # "video" | "still" | "live_photo"
    thumb_url: str
    selected: bool
    captured_at: float | None
    size_bytes: int | None


@dataclass
class SaveResult:
    """Return type for `save_selection` / `save_preferences` — what the GUI
    needs to know about a successful write."""

    sidecar: Path
    filename: str
    n_items: int


@dataclass
class RenderJob:
    """Mutable state of one in-flight (or finished) render.

    `stage` / `stage_index` / `stage_total` / `progress_percent` / `eta_s`
    are written from the worker thread under `_progress_lock`; readers
    (HTTP poll) grab a consistent snapshot via `status()`. `cpu_seconds_used`
    is the sum of `time.process_time()` deltas across stage transitions —
    when VideoToolbox lands (Phase B) this number drops dramatically while
    wall-clock holds steady, which is the whole point of A4.
    """

    job_id: str
    state: str = "running"  # "running" | "done" | "error"
    output_path: str | None = None
    error: str | None = None
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    # ---- progress fields (Phase A2) ----
    # `stage` is one of: "", "analyze", "score", "prerender", "assemble",
    # "mux", "done", "error". Empty string before the worker has set anything.
    stage: str = ""
    stage_index: int = 0
    stage_total: int = 0
    progress_percent: float = 0.0
    eta_s: float | None = None
    current_ffmpeg_pid: int | None = None
    cpu_seconds_used: float = 0.0


@dataclass
class ImportJob:
    """Mutable state of one in-flight (or finished) import.

    Mirrors `RenderJob` shape — same lifecycle (`running` → `done`/`error`),
    same `log_tail` deque so `_LogCapture` can be reused unchanged. Adds
    progress counters (`copied`, `skipped`, `failed`, `total`) and the
    `dest_folder` the worker is writing into.
    """

    job_id: str
    state: str = "running"  # "running" | "done" | "error"
    copied: int = 0
    skipped: int = 0
    failed: int = 0
    total: int = 0
    dest_folder: str | None = None
    error: str | None = None
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


# Whitelist of render overrides — only known knobs flow through. Kept at
# module level so it's discoverable and the test suite can assert against it.
ALLOWED_RENDER_OVERRIDES = frozenset({
    "max_length", "pace", "aspect", "res", "fps", "music_db", "clip_db",
    "audio_mix", "transitions", "no_speed_ramp", "no_reframe", "lut",
    "theme", "source_cap", "chronological", "preview", "no_stills",
    "still_duration", "titles", "title_text", "burst_window_s",
})


# ---- internal helpers ------------------------------------------------------

def _kind_for(p: Path, live_movs: set[str]) -> str:
    if str(p) in live_movs:
        return "live_photo"
    if p.suffix.lower() in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
        return "still"
    return "video"


class _LogCapture(io.TextIOBase):
    """A tee-style stderr replacement that pushes each line into a job's log_tail.

    Works for any object with a `log_tail: deque[str]` attribute — currently
    `RenderJob` and `ImportJob` both qualify.
    """

    def __init__(self, job: "RenderJob | ImportJob", mirror: io.TextIOBase | None) -> None:
        super().__init__()
        self._job = job
        self._mirror = mirror
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        if self._mirror is not None:
            try:
                self._mirror.write(s)
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._job.log_tail.append(line)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass


# Per-stage weight table used to map per-stage fractions onto the overall
# `progress_percent`. The numbers come from real-world session profiling on
# an M5 Pro: prerender dominates wall-clock, assemble is a single x264 pass
# (or a near-instant concat-copy), mux is a video-copy + AAC encode, and
# analyze/score are tiny compared to the encode work. They sum to 100.
RENDER_STAGE_WEIGHTS: dict[str, float] = {
    "analyze": 1.0,
    "score": 1.0,
    "prerender": 75.0,
    "assemble": 18.0,
    "mux": 5.0,
}
# Cumulative % budget before each stage starts — gives the closure a fast
# "convert (stage, fraction_in_stage) -> overall %" without re-summing.
_STAGE_CUMULATIVE: dict[str, float] = {}
_acc = 0.0
for _stage in ("analyze", "score", "prerender", "assemble", "mux"):
    _STAGE_CUMULATIVE[_stage] = _acc
    _acc += RENDER_STAGE_WEIGHTS[_stage]
del _acc, _stage


def _run_render_job(
    job: RenderJob,
    clips: Path,
    song: Path | None,
    output: Path | None,
    overrides: dict[str, Any],
) -> None:
    """Worker thread: drive `pipeline_runner.run_auto` and capture state on the job."""
    # Imports are deferred so just importing `select.service` doesn't drag in
    # librosa/numpy/ffmpeg-heavy modules in test contexts that don't render.
    from aftermovie.effective_config import resolve
    from aftermovie.pipeline_runner import opts_from_namespace, run_auto

    saved_stderr = sys.stderr
    sys.stderr = _LogCapture(job, saved_stderr if sys.stderr is not None else None)
    # Thread-safe progress writes — the HTTP poll thread reads these fields
    # via `status()` so all mutations go through the lock.
    progress_lock = threading.Lock()
    # CPU-seconds bookkeeping. `time.process_time()` measures user+system CPU
    # consumed by *this* process (the worker thread + any main-thread work it
    # triggers — librosa/numpy stuff). It doesn't include child processes
    # like ffmpeg; we add those via `os.times().children_*` below at each
    # stage boundary so the GUI sees the full picture.
    import os
    cpu_start = time.process_time()
    children_start = os.times()
    # When did the current stage begin? Used for ETA math.
    stage_started_wall: dict[str, float] = {}
    last_stage_seen: list[str] = [""]

    def _set_stage(stage: str, *, index: int = 0, total: int = 0) -> None:
        """Mark `stage` started; record wall-clock + accumulate CPU seconds."""
        nonlocal cpu_start, children_start
        now = time.time()
        cpu_now = time.process_time()
        children_now = os.times()
        delta = (cpu_now - cpu_start) + max(
            0.0,
            (children_now.children_user + children_now.children_system)
            - (children_start.children_user + children_start.children_system),
        )
        with progress_lock:
            job.cpu_seconds_used += delta
            job.stage = stage
            job.stage_index = index
            job.stage_total = total
            # Anchor `progress_percent` at the stage's cumulative budget so the
            # bar jumps forward at each boundary even before the first
            # frame=N tick of the new stage arrives.
            job.progress_percent = _STAGE_CUMULATIVE.get(stage, job.progress_percent)
            if stage in ("done", "error"):
                job.eta_s = 0.0
        cpu_start = cpu_now
        children_start = children_now
        stage_started_wall[stage] = now
        last_stage_seen[0] = stage

    def _on_progress(event: "Any") -> None:
        """Translate a `ProgressEvent` into RenderJob field updates.

        Stage boundaries inside the prerender loop are detected here too:
        when `event.stage_index` jumps to a new value we treat that as the
        next clip's slice of the prerender budget starting. The per-clip
        slice is `prerender_weight / N`, so overall % advances smoothly
        across the whole pool.
        """
        stage = event.stage
        idx = event.stage_index
        total = max(1, event.stage_total or 1)
        frac = max(0.0, min(1.0, event.fraction_in_stage))
        # Map (stage, idx, frac) onto overall percent. For multi-step stages
        # like prerender, each clip owns 1/N of the stage budget; the current
        # clip is at fraction `(idx-1+frac) / N`.
        per_step_frac = ((max(idx, 1) - 1) + frac) / total
        stage_budget = RENDER_STAGE_WEIGHTS.get(stage, 0.0)
        base = _STAGE_CUMULATIVE.get(stage, 0.0)
        overall = base + stage_budget * per_step_frac
        # ETA from the in-stage progress so far. Cheap, slightly biased on the
        # first ~5 % of the stage; the GUI surfaces "—" until the bias bleeds
        # out (>5 % done, see `pollStatus` in app.js).
        eta: float | None = None
        started = stage_started_wall.get(stage)
        if started is not None and overall > 1.0:
            elapsed = max(0.001, time.time() - job.started_at)
            remaining_frac = max(0.0, 1.0 - overall / 100.0)
            eta = (elapsed / max(0.0001, overall / 100.0)) * remaining_frac

        with progress_lock:
            if stage != job.stage:
                job.stage = stage
                stage_started_wall.setdefault(stage, time.time())
            job.stage_index = idx
            job.stage_total = total
            # Monotonic: never let a delayed callback walk the bar backwards.
            if overall > job.progress_percent:
                job.progress_percent = min(100.0, overall)
            if eta is not None:
                job.eta_s = max(0.0, eta)
            if event.current_pid is not None:
                job.current_ffmpeg_pid = event.current_pid

    try:
        if song is None:
            raise ValueError("`song` is required to render")
        cfg = resolve(cli_overrides=overrides, theme=overrides.get("theme"))
        # Build AutoOpts from the resolved config, then layer the few extras
        # the CLI Namespace would normally carry.
        import argparse
        from dataclasses import asdict as _asdict
        ns = argparse.Namespace(**_asdict(cfg))
        for k, v in overrides.items():
            setattr(ns, k, v)
        opts = opts_from_namespace(ns)
        opts.reveal = False  # never trigger Finder reveal from the web flow
        out_path = output
        if out_path is None:
            out_dir = Path(cfg.output_dir).expanduser()
            out_path = out_dir / f"aftermovie-{clips.resolve().name or 'edit'}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Coarse-grained stage signposts — run_auto goes analyze → score →
        # render (with prerender/assemble/mux inside render). We mark
        # analyze/score here; the ProgressEvent callback drives the rest.
        _set_stage("analyze")
        # `run_auto` runs analyze internally, but exposing per-Entry analyze
        # progress is Phase E territory; for now we let analyze + score
        # consume their cumulative 2 % budget atomically.
        result_path = run_auto(clips, song, out_path, opts, progress_cb=_on_progress)
        job.output_path = str(result_path)
        _set_stage("done")
        with progress_lock:
            job.progress_percent = 100.0
            job.eta_s = 0.0
            job.current_ffmpeg_pid = None
        job.state = "done"
    except Exception as e:
        tb = traceback.format_exc()
        job.error = f"{type(e).__name__}: {e}"
        for line in tb.splitlines():
            job.log_tail.append(line)
        with progress_lock:
            job.stage = "error"
            job.eta_s = None
        job.state = "error"
    finally:
        sys.stderr = saved_stderr
        job.finished_at = time.time()


def _run_import_job_subprocess(
    job: ImportJob,
    source_names: list[str],
    since_str: str,
    until_str: str,
    dest_parent: Path,
    dry_run: bool,
) -> None:
    """Run the import by spawning `aftermovie import` as a subprocess.

    Some Adapters (notably `GoProICCAdapter`, which talks to MTP cameras
    via ImageCaptureCore) require the *process main thread* — Cocoa's
    NSRunLoop delivers ICC delegate callbacks only there. SelectionService
    runs in HTTP worker threads, so we shell out to the CLI, which gets
    its own fresh process with its own main thread.

    Parses the CLI's stderr lines into `job.copied/skipped/failed`.
    """
    import re
    import subprocess
    import sys as _sys

    cmd = [
        _sys.executable, "-m", "aftermovie", "import",
        "--since", since_str, "--until", until_str,
        "--to", str(dest_parent),
        "--sources", ",".join(source_names),
    ]
    if dry_run:
        cmd.append("--dry-run")

    per_source_re = re.compile(
        r"copied=(\d+)\s+skipped=(\d+)\s+failed=(\d+)"
    )
    in_range_re = re.compile(r":\s+(\d+)\s+item\(s\)\s+in\s+range")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            job.log_tail.append(line)
            m = per_source_re.search(line)
            if m:
                job.copied += int(m.group(1))
                job.skipped += int(m.group(2))
                job.failed += int(m.group(3))
                continue
            m = in_range_re.search(line)
            if m:
                job.total += int(m.group(1))
        rc = proc.wait()
        if rc != 0:
            job.state = "error"
            job.error = f"aftermovie import exited with code {rc}"
            return
        job.state = "done"
    except Exception as e:
        tb = traceback.format_exc()
        job.error = f"{type(e).__name__}: {e}"
        for line in tb.splitlines():
            job.log_tail.append(line)
        job.state = "error"
    finally:
        job.finished_at = time.time()


def _run_import_job(
    job: ImportJob,
    sources: list[Any],
    since: "Any",
    until: "Any",
    dest_folder: Path,
    dry_run: bool,
) -> None:
    """Worker thread: walk each `ImportSource`, copy items, update `job`.

    `sources` is a list of `ImportSource` instances already filtered to the
    user's selection. `since` / `until` are `datetime` objects passed straight
    through to each source's `list_in_range`. On `dry_run=True` we only count
    items via `list_in_range` and never call `copy_into`.
    """
    saved_stderr = sys.stderr
    sys.stderr = _LogCapture(job, saved_stderr if sys.stderr is not None else None)
    try:
        # First pass: discover items per source so we know the grand total.
        items_per_source: list[tuple[Any, list[Any]]] = []
        for src in sources:
            items = list(src.list_in_range(since, until))
            items_per_source.append((src, items))
            job.total += len(items)

        if dry_run:
            job.state = "done"
            return

        dest_folder.mkdir(parents=True, exist_ok=True)

        # base.copy_files signature: progress_cb(done, total, src_path) — fires
        # once per file processed. We use `done` for live progress feedback,
        # then fold the accurate copied/skipped/failed breakdown from
        # CopyResult after each source finishes.
        def _progress_cb(done: int, _total: int, _src: str | None = None) -> None:
            # Provisional running count so the GUI's progress bar moves; the
            # CopyResult fold below replaces it with the true breakdown.
            job.copied = done

        for src, items in items_per_source:
            if not items:
                continue
            # Reset the provisional counter so we can fold this source cleanly.
            before_done = job.copied
            result = src.copy_into(items, dest_folder, progress_cb=_progress_cb)
            # Roll back the provisional in-flight count and apply the real
            # breakdown from this source's CopyResult.
            job.copied = before_done
            if result is not None:
                job.copied += getattr(result, "copied", 0)
                job.skipped += getattr(result, "skipped", 0)
                job.failed += getattr(result, "failed", 0)
        job.state = "done"
    except Exception as e:
        tb = traceback.format_exc()
        job.error = f"{type(e).__name__}: {e}"
        for line in tb.splitlines():
            job.log_tail.append(line)
        job.state = "error"
    finally:
        sys.stderr = saved_stderr
        job.finished_at = time.time()


# ---- the service -----------------------------------------------------------

class SelectionService:
    """GUI-facing domain Module. One instance per `SelectServer`.

    The Module owns the per-folder operations the GUI invokes — discovery,
    selection, preferences, plan lookup, thumb fetch, render dispatch.
    HTTP-only concerns (status codes, content-types, JSON serialization)
    stay in `_Handler`; the service deals in domain values (`SourceRow`,
    `SaveResult`, `RenderJob`, raw bytes for thumbs).
    """

    def __init__(
        self,
        clips_root: Path,
        song_default: Path | None = None,
        *,
        render_runner: "Any" = None,
        import_runner: "Any" = None,
    ) -> None:
        self.clips_root = Path(clips_root).expanduser().resolve()
        self.song_default = song_default
        # Render-job tracking lives on the instance so two SelectionServices
        # in the same process don't share state.
        self._jobs: dict[str, RenderJob] = {}
        self._jobs_lock = threading.Lock()
        # Import-job tracking — separate dict so render and import job IDs
        # never collide and either Module can evolve independently.
        self._import_jobs: dict[str, ImportJob] = {}
        self._import_jobs_lock = threading.Lock()
        # Render dispatch is parameterized so tests can stub `run_auto`
        # without monkey-patching the module-level import.
        self._render_runner = render_runner or _run_render_job
        # Same pattern for the import worker so tests can stub it out.
        self._import_runner = import_runner or _run_import_job

    # ---- source discovery ----

    def list_sources(self) -> list[SourceRow]:
        """Walk `clips_root` the same way the analyzer does and return GUI rows.

        Live Photo pairs (`IMG_0488.HEIC` + `IMG_0488.MOV`) collapse to a
        single `live_photo` entry pointing at the MOV (the HEIC is implied
        by the shared stem). Standalone stills are `still`. Everything else
        with a VIDEO_EXTS suffix is `video`. Hidden files, prior aftermovie
        outputs, and SKIP_DIR_NAMES subtrees are dropped exactly like
        `discover_sources`.

        Items are sorted by their capture timestamp (EXIF / ffprobe / mtime).
        """
        folder = self.clips_root
        if not folder.is_dir():
            return []

        excluded = load_excluded(folder)

        # First pass: group files by stem so we can detect paired Live Photos.
        by_stem: dict[str, list[Path]] = {}
        for p in folder.rglob("*"):
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            if p.name.startswith("."):
                continue
            if _under_skipped_dir(p, folder):
                continue
            by_stem.setdefault(p.stem, []).append(p)

        # Sets so we can identify Live Photo MOVs and stills that have a video pair.
        live_movs: set[str] = set()
        paired_still_paths: set[str] = set()
        for files in by_stem.values():
            movs = [f for f in files if f.suffix in LIVE_PHOTO_VIDEO_EXTS]
            stills = [f for f in files if f.suffix in STILL_EXTS]
            if movs and stills:
                # Treat the MOV as a Live Photo entry; hide the still companions.
                for m in movs:
                    live_movs.add(str(m))
                for s in stills:
                    paired_still_paths.add(str(s))

        # Allow both the case-sensitive VIDEO_EXTS set the analyzer uses AND a
        # case-insensitive fallback so .lrv / .insv variants captured on other
        # filesystems still show up.
        video_exts_ci = {e.lower() for e in VIDEO_EXTS}
        still_exts_ci = {".heic", ".heif", ".jpg", ".jpeg", ".png"}

        rows: list[SourceRow] = []
        seen_paths: set[str] = set()
        for files in by_stem.values():
            for p in files:
                abs_p = str(p)
                if abs_p in seen_paths:
                    continue
                if abs_p in paired_still_paths:
                    continue  # hidden — its MOV represents the Live Photo
                ext_l = p.suffix.lower()
                if ext_l in video_exts_ci:
                    if _is_excluded_output(p):
                        continue
                elif ext_l in still_exts_ci:
                    pass
                else:
                    continue
                kind = _kind_for(p, live_movs)
                try:
                    size_b = p.stat().st_size
                except OSError:
                    size_b = None
                try:
                    captured = captured_at_for(p)
                except Exception:
                    captured = None
                thumb_url = f"/thumbs/{thumb_cache_key(p)}.jpg"
                rows.append(SourceRow(
                    path=abs_p,
                    name=p.name,
                    kind=kind,
                    thumb_url=thumb_url,
                    selected=(abs_p not in excluded),
                    captured_at=captured,
                    size_bytes=size_b,
                ))
                seen_paths.add(abs_p)

        rows.sort(key=lambda r: (r.captured_at if r.captured_at is not None else float("inf"), r.name))
        return rows

    # ---- selection sidecar ----

    def get_selection(self) -> list[str]:
        """Return the current excluded-paths list (sorted, deduped)."""
        return sorted(load_excluded(self.clips_root))

    def save_selection(self, excluded: list[str]) -> SaveResult:
        """Persist `excluded` to the selection sidecar and return the result.

        Non-string entries are dropped silently; duplicates collapse with
        first-seen order preserved by the sidecar Adapter.
        """
        cleaned = [str(p) for p in excluded if isinstance(p, str)]
        sidecar = save_excluded(self.clips_root, cleaned)
        return SaveResult(sidecar=sidecar, filename=SELECTION_FILENAME,
                          n_items=len(cleaned))

    # ---- preferences sidecar ----

    def get_preferences(self) -> dict[str, list[str]]:
        """Return the persisted favorited/banned/pinned dict (always 3 fields)."""
        return load_preferences(self.clips_root)

    def save_preferences(self, prefs: dict[str, Any]) -> SaveResult:
        """Persist favorited/banned (+ reserved pinned_entries) to the sidecar.

        Missing fields are written as empty lists — the GUI always POSTs the
        full intended state. Non-string entries are dropped by the Adapter.
        """
        payload = {
            "favorited": prefs.get("favorited", []),
            "banned": prefs.get("banned", []),
            "pinned_entries": prefs.get("pinned_entries", []),
        }
        sidecar = save_preferences(self.clips_root, payload)
        return SaveResult(sidecar=sidecar, filename=sidecar.name,
                          n_items=sum(len(payload[k]) for k in payload if isinstance(payload[k], list)))

    # ---- plan lookup ----

    def latest_plan(self) -> dict[str, Any] | None:
        """Return the most-recent Plan tagged with this folder's catalog_id.

        Delegates to `PlanRepository.get_latest_for_catalog`; returns None
        (not an exception) when no matching Plan exists yet.
        """
        from aftermovie.repos import catalog_repo, plan_repo

        catalog_id = catalog_repo.id_for(self.clips_root)
        return plan_repo.get_latest_for_catalog(catalog_id)

    # ---- options menu ----

    def available_options(self) -> dict[str, Any]:
        """The GUI's options dropdowns: themes, luts, audio modes, etc."""
        return {
            "luts": list_luts(),
            "themes": [
                {"name": name, **meta}
                for name, meta in sorted(THEMES.items())
            ],
            "audio_mix": ["ducked", "music_only", "clip_only"],
            "pace": ["auto", "slow", "medium", "fast"],
            "transitions": ["soft", "auto", "cut"],
            "aspect": ["16:9", "9:16", "1:1"],
            "resolution": ["1920x1080", "1080x1920", "1080x1080", "1280x720"],
        }

    # ---- thumbnails ----

    def thumb_for_key(self, key: str) -> bytes | None:
        """Return raw JPG bytes for the cached thumb keyed by `key`, or None.

        If the cache file is missing we re-walk `clips_root` to find the
        SourceRow whose thumb_url matches, then synthesize the thumb via
        `thumb_path_for`. None means "no such thumb / read failed."
        """
        cache_dir = _thumbs_cache_dir()
        cached = cache_dir / f"{key}.jpg"
        if not cached.is_file():
            for r in self.list_sources():
                if r.thumb_url == f"/thumbs/{key}.jpg":
                    p = thumb_path_for(Path(r.path))
                    if p is not None:
                        cached = p
                    break
        if not cached.is_file():
            return None
        try:
            return cached.read_bytes()
        except OSError:
            return None

    # ---- render jobs ----

    def start_render(self, body: dict[str, Any]) -> RenderJob:
        """Kick off a render in a worker thread; return the new `RenderJob`.

        Returns immediately — call `status(job.job_id)` to poll for the
        result. Selection edits embedded in `body["excluded"]` are persisted
        before the worker starts, so the render sees the new state.
        """
        raw_excluded = body.get("excluded")
        if isinstance(raw_excluded, list):
            self.save_selection([p for p in raw_excluded if isinstance(p, str)])

        # `song` may come from the request, otherwise the CLI-supplied default.
        song = self.song_default
        if isinstance(body.get("song"), str):
            song = Path(body["song"]).expanduser().resolve()

        output: Path | None = None
        if isinstance(body.get("output"), str) and body["output"]:
            output = Path(body["output"]).expanduser().resolve()

        overrides: dict[str, Any] = {}
        for k in ALLOWED_RENDER_OVERRIDES:
            if k in body and body[k] is not None:
                overrides[k] = body[k]

        job = RenderJob(job_id=str(uuid.uuid4()))
        with self._jobs_lock:
            self._jobs[job.job_id] = job
        worker = threading.Thread(
            target=self._render_runner,
            args=(job, self.clips_root, song, output, overrides),
            name=f"render-{job.job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return job

    def status(self, job_id: str) -> dict[str, Any] | None:
        """Snapshot of `job_id`'s state as a JSON-friendly dict, or None.

        None means "no such job in this service's tracking dict" — the GUI
        renders that as a 404. The `log_tail` is joined with newlines so
        the response shape is the same one the HTTP handler produced before
        the refactor.
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "state": job.state,
            "output_path": job.output_path,
            "error": job.error,
            "log_tail": "\n".join(job.log_tail),
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            # Phase A2/A3/A4 progress fields. Defaults are sensible for a
            # job that hasn't yet emitted a progress event ("" stage,
            # 0 percent, no ETA, no pid).
            "stage": job.stage,
            "stage_index": job.stage_index,
            "stage_total": job.stage_total,
            "progress_percent": job.progress_percent,
            "eta_s": job.eta_s,
            "current_ffmpeg_pid": job.current_ffmpeg_pid,
            "cpu_seconds_used": job.cpu_seconds_used,
        }

    # ---- import jobs ----

    def available_import_sources(self) -> list[dict[str, Any]]:
        """Return `[{name, label, available}]` for each known ImportSource.

        Lazy import of `aftermovie.import_sources` so the (parallel) backend
        Module landing late doesn't break the rest of the service. The HTTP
        Adapter catches `ModuleNotFoundError` and returns 503 to the GUI.
        """
        from aftermovie.import_sources import all_sources  # local: lazy
        rows: list[dict[str, Any]] = []
        for src in all_sources():
            try:
                avail = bool(src.available())
            except Exception:
                avail = False
            rows.append({
                "name": src.name,
                "label": src.label,
                "available": avail,
            })
        return rows

    def start_import(
        self,
        since: str,
        until: str,
        source_names: list[str],
        dest_parent: str | Path | None = None,
        dry_run: bool = False,
    ) -> ImportJob:
        """Kick off an import in a worker thread; return the new `ImportJob`.

        `since` / `until` are ISO `YYYY-MM-DD` strings; they're parsed here
        (a `ValueError` bubbles up so the HTTP Adapter can map it to 400).
        `source_names` selects which `ImportSource`s to drive; unknown names
        are silently dropped so a stale GUI selection doesn't error out.
        """
        from datetime import datetime as _dt

        from aftermovie.import_sources import all_sources  # local: lazy

        # Date parsing — fromisoformat raises ValueError on garbage like "tomorrow".
        since_dt = _dt.fromisoformat(since)
        until_dt = _dt.fromisoformat(until)

        if not source_names:
            raise ValueError("at least one source name is required")

        # Filter `all_sources()` down to the user's selection, preserving the
        # order they requested so the worker's traversal is deterministic.
        by_name = {s.name: s for s in all_sources()}
        selected = [by_name[n] for n in source_names if n in by_name]
        if not selected:
            raise ValueError("none of the requested sources are known")

        parent = Path(dest_parent).expanduser() if dest_parent else (
            Path.home() / "Movies" / "aftermovie-imports"
        )
        dest_folder = parent / f"{since}_to_{until}"

        job = ImportJob(job_id=str(uuid.uuid4()), dest_folder=str(dest_folder))
        with self._import_jobs_lock:
            self._import_jobs[job.job_id] = job

        # Sources whose Adapter requires the process main thread
        # (ImageCaptureCore-based GoPro MTP) must run in a subprocess so
        # the CLI's own main thread can pump NSRunLoop. If any selected
        # source needs that, route the whole job through the subprocess
        # runner. Tests inject a custom `import_runner` to bypass this.
        needs_subprocess = (
            self._import_runner is _run_import_job
            and any(n.startswith("gopro_icc_") for n in source_names)
        )
        if needs_subprocess:
            worker = threading.Thread(
                target=_run_import_job_subprocess,
                args=(job, source_names, since, until, parent, dry_run),
                name=f"import-sub-{job.job_id[:8]}",
                daemon=True,
            )
        else:
            worker = threading.Thread(
                target=self._import_runner,
                args=(job, selected, since_dt, until_dt, dest_folder, dry_run),
                name=f"import-{job.job_id[:8]}",
                daemon=True,
            )
        worker.start()
        return job

    def import_status(self, job_id: str) -> dict[str, Any] | None:
        """Snapshot of import `job_id` as a JSON-friendly dict, or None.

        Shape matches the HTTP contract the GUI is hard-coded against:
        `{job_id, state, copied, skipped, failed, total, dest_folder,
        error, log_tail}`. None → the HTTP Adapter renders 404.
        """
        with self._import_jobs_lock:
            job = self._import_jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "state": job.state,
            "copied": job.copied,
            "skipped": job.skipped,
            "failed": job.failed,
            "total": job.total,
            "dest_folder": job.dest_folder,
            "error": job.error,
            "log_tail": "\n".join(job.log_tail),
        }
