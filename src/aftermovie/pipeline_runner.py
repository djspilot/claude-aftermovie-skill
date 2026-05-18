"""Shared analyze → score → render orchestration used by both CLI and MCP.

Today both the `aftermovie auto` CLI subcommand and the MCP `auto` tool need
to drive the same three-stage pipeline. This module is the single source of
truth — `run_auto(...)` is the only entry point. Theme bundles are applied
here too, so the CLI's `--theme` flag and the MCP `auto(theme=...)` parameter
share one implementation.

The orchestration intentionally goes through the existing argparse-style
entry points (`cmd_analyze`, `cmd_score`, `cmd_render`) so the analyze /
score / render modules themselves stay untouched.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

from aftermovie.analyze.clip import cmd_analyze
from aftermovie.config import (
    DEFAULT_CLIP_DB,
    DEFAULT_FPS,
    DEFAULT_MUSIC_DB,
    DEFAULT_RES,
)
from aftermovie.ffmpeg_cmd import log
from aftermovie.render.pipeline import ProgressEvent, cmd_render
from aftermovie.repos import PlanIdOpts, catalog_repo, plan_repo
from aftermovie.score.scorer import cmd_score
from aftermovie.themes import BASELINE_DEFAULTS as _THEME_DEFAULTS
from aftermovie.themes import ThemeResolver

# Re-export so callers can `from aftermovie.pipeline_runner import ProgressEvent`
# without reaching into render/pipeline.py — keeps the orchestration Module the
# one Seam progress flows through.
__all__ = [
    "AutoOpts", "ProgressEvent", "opts_from_namespace",
    "run_auto", "run_render_only",
]

ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class AutoOpts:
    """Knobs for `run_auto`. Each default matches the CLI built-in default —
    theme bundles only override fields still at these defaults."""

    aspect: str = "16:9"
    res: str = DEFAULT_RES
    fps: int = DEFAULT_FPS
    max_length: float | None = None
    still_duration: float = 2.5
    no_stills: bool = False
    audio_mix: str = "ducked"
    music_db: float = DEFAULT_MUSIC_DB
    clip_db: float = DEFAULT_CLIP_DB
    pace: str = "medium"
    transitions: str = "cut"
    no_speed_ramp: bool = False
    no_reframe: bool = False
    lut: str | None = None
    theme: str | None = None
    titles: str | None = None
    title_text: str | None = None
    source_cap: int = 1
    chronological: bool = True
    burst_window_s: float = 3.0
    preview: bool = False
    reveal: bool = True
    force_reanalyze: bool = False
    moments_per_source: int | None = None  # F3 ceiling; None = use scorer default (1)


# Preview-mode overrides — applied after theme expansion so --preview wins.
_PREVIEW_RES_BY_ASPECT = {
    "16:9": "854x480",
    "9:16": "480x854",
    "1:1":  "480x480",
}
PREVIEW_FPS = 24
PREVIEW_MARKER = "[PREVIEW MODE — quarter-res, no LUT]"


def _apply_preview_overrides(opts: "AutoOpts") -> "AutoOpts":
    """Knock down resolution / fps / LUT / reframe for a fast iteration render."""
    if not opts.preview:
        return opts
    opts.res = _PREVIEW_RES_BY_ASPECT.get(opts.aspect, "854x480")
    opts.fps = PREVIEW_FPS
    opts.lut = None
    opts.no_reframe = True
    return opts


# The baseline-defaults snapshot used to detect "is this field still at its
# built-in default?" lives in `aftermovie.themes` (imported above as
# `_THEME_DEFAULTS`). The precedence logic lives there too — this Module
# only owns the AutoOpts dataclass <-> dict adapter glue.


def _apply_theme(opts: AutoOpts) -> AutoOpts:
    """If `opts.theme` is set, fill in unset (still-at-default) theme fields."""
    current = {k: getattr(opts, k) for k in _THEME_DEFAULTS if hasattr(opts, k)}
    resolved = ThemeResolver.apply(opts.theme, current, _THEME_DEFAULTS)
    for k, v in resolved.items():
        setattr(opts, k, v)
    return opts


def opts_from_namespace(args: argparse.Namespace) -> AutoOpts:
    """Build an `AutoOpts` from an argparse Namespace, honouring whatever
    fields are present and falling back to dataclass defaults for the rest."""
    kwargs: dict = {}
    for f in fields(AutoOpts):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    return AutoOpts(**kwargs)


def run_render_only(plan: Path, output: Path, opts: AutoOpts | None = None,
                    *, progress_cb: ProgressCallback | None = None) -> Path:
    """Skip analyze + score; just dispatch `cmd_render` on an existing plan.

    Used by `aftermovie auto --from-plan` and `aftermovie render-from-plan`
    so a saved plan can be re-rendered without re-walking the source folder
    or re-scoring against the song. Honours `opts.reveal` for the macOS
    Finder reveal at the end (default True for parity with `run_auto`).
    """
    plan_path = Path(plan).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = argparse.Namespace(plan=str(plan_path), output=str(output_path))
    cmd_render(r, progress_cb=progress_cb)

    if opts is None or opts.reveal:
        _notify_and_reveal(output_path)
    return output_path


def run_auto(clips: Path, song: Path, output: Path, opts: AutoOpts,
             *, progress_cb: ProgressCallback | None = None) -> Path:
    """Full analyze → score → render. Returns the output path on success.

    Applies the theme bundle (if `opts.theme` is set) before dispatching, so
    callers don't need to worry about expansion order. Intermediate catalog
    and plan JSON files are written under a temp dir that survives the call
    (logged on completion).
    """
    opts = _apply_theme(opts)
    opts = _apply_preview_overrides(opts)
    if opts.preview:
        log(PREVIEW_MARKER)

    clips_path = Path(clips).expanduser().resolve()
    song_path = Path(song).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workdir = Path(tempfile.mkdtemp(prefix="aftermovie_auto_"))
    log(f"Working dir: {workdir}")
    catalog_path = workdir / "catalog.json"
    plan_path = workdir / "plan.json"

    # 1) analyze — consult the Catalog Repository's on-disk cache first.
    # IDs are content-derived (folder path + file list + sizes + mtimes),
    # so a hit means the source tree is byte-identical to the prior run.
    # `--force-analyze` / `opts.force_reanalyze=True` bypass the cache.
    cid = catalog_repo.id_for(clips_path)
    cached_path = catalog_repo.path_for_id(cid)
    if cached_path.is_file() and not opts.force_reanalyze:
        log(f"Using cached catalog {cid}")
        catalog_repo.copy_into(cid, catalog_path)
    else:
        a = argparse.Namespace(
            clips=str(clips_path),
            out=str(catalog_path),
            still_duration=opts.still_duration,
            no_stills=opts.no_stills,
        )
        cmd_analyze(a)
        # Hand the freshly-written catalog to the repository — `put` stamps
        # `_aftermovie.catalog_id` and persists into the cache. Best-effort:
        # a malformed catalog falls through to the score stage unchanged.
        try:
            catalog = json.loads(catalog_path.read_text())
            if isinstance(catalog, dict):
                catalog_repo.put(clips_path, catalog)
                catalog_path.write_text(json.dumps(catalog, indent=2))
                log(f"Cached catalog {cid}")
        except (OSError, ValueError) as e:
            log(f"  ! could not cache catalog {cid}: {type(e).__name__}: {e}")

    # 2) score
    s = argparse.Namespace(
        catalog=str(catalog_path),
        song=str(song_path),
        out=str(plan_path),
        max_length=opts.max_length,
        aspect=opts.aspect,
        res=opts.res,
        fps=opts.fps,
        lut=opts.lut,
        music_db=opts.music_db,
        clip_db=opts.clip_db,
        no_speed_ramp=opts.no_speed_ramp,
        audio_mix=opts.audio_mix,
        pace=opts.pace,
        transitions=opts.transitions,
        titles=opts.titles,
        title_text=opts.title_text,
        no_reframe=opts.no_reframe,
        source_cap=opts.source_cap,
        chronological=opts.chronological,
        burst_window_s=opts.burst_window_s,
        moments_per_source=opts.moments_per_source,
    )
    cmd_score(s)

    # Hand the freshly-scored plan to the Plan Repository. Scoring is cheap
    # so we always re-score and never skip, but `put` stamps the plan with
    # both ids and persists it under the canonical plan-dir name — that's
    # what the GUI's /api/plan endpoint reads back.
    id_opts = PlanIdOpts(opts.theme, opts.max_length, opts.aspect, seed=0)
    try:
        plan = json.loads(plan_path.read_text())
        if isinstance(plan, dict):
            plan_repo.put(cid, song_path, id_opts, plan)
            plan_path.write_text(json.dumps(plan, indent=2))
    except (OSError, ValueError) as e:
        pid = plan_repo.id_for(cid, song_path, opts.theme, opts.max_length,
                               opts.aspect, 0)
        log(f"  ! could not cache plan {pid}: {type(e).__name__}: {e}")

    # 3) render
    r = argparse.Namespace(plan=str(plan_path), output=str(output_path))
    cmd_render(r, progress_cb=progress_cb)

    log(f"Intermediate files preserved in: {workdir}")
    if opts.reveal:
        _notify_and_reveal(output_path)
    return output_path


def _notify_and_reveal(output_path: Path) -> None:
    """macOS-only: post a notification and reveal the file in Finder.

    No-op on non-tty / non-macOS / when osascript isn't available, so the MCP
    server and CI runs stay quiet.
    """
    import shutil
    import subprocess
    import sys

    if not sys.stdout.isatty():
        return
    if sys.platform != "darwin":
        return
    osa = shutil.which("osascript")
    if osa:
        try:
            subprocess.run(
                [osa, "-e",
                 f'display notification "{output_path.name} ready" '
                 f'with title "aftermovie"'],
                check=False, capture_output=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    op = shutil.which("open")
    if op:
        try:
            subprocess.run([op, "-R", str(output_path)],
                           check=False, capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
