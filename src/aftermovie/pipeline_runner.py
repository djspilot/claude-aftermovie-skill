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
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

from aftermovie.analyze.clip import cmd_analyze
from aftermovie.config import (
    DEFAULT_CLIP_DB,
    DEFAULT_FPS,
    DEFAULT_MUSIC_DB,
    DEFAULT_RES,
    THEMES,
)
from aftermovie.ffmpeg_cmd import log
from aftermovie.render.pipeline import cmd_render
from aftermovie.score.scorer import cmd_score


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


# Per-field "is this still the built-in default?" snapshot. The theme bundle
# only overrides a field when its current value equals the baseline — anything
# the caller (CLI flag, env file, MCP arg) explicitly set is preserved.
_THEME_DEFAULTS: dict[str, object] = {
    "lut": None,
    "music_db": DEFAULT_MUSIC_DB,
    "no_speed_ramp": False,
    "transitions": "cut",
    "audio_mix": "ducked",
    "pace": "medium",
}


def _apply_theme(opts: AutoOpts) -> AutoOpts:
    """If `opts.theme` is set, fill in unset (still-at-default) theme fields."""
    if not opts.theme:
        return opts
    preset = THEMES.get(opts.theme)
    if not preset:
        return opts
    for k, v in preset.items():
        if k == "description":
            continue
        if not hasattr(opts, k):
            continue
        cur = getattr(opts, k)
        baseline = _THEME_DEFAULTS.get(k)
        # Only override when the field is still at the built-in default.
        if cur == baseline:
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


def run_render_only(plan: Path, output: Path, opts: AutoOpts | None = None) -> Path:
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
    cmd_render(r)

    if opts is None or opts.reveal:
        _notify_and_reveal(output_path)
    return output_path


def run_auto(clips: Path, song: Path, output: Path, opts: AutoOpts) -> Path:
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

    # 1) analyze
    a = argparse.Namespace(
        clips=str(clips_path),
        out=str(catalog_path),
        still_duration=opts.still_duration,
        no_stills=opts.no_stills,
    )
    cmd_analyze(a)

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
    )
    cmd_score(s)

    # 3) render
    r = argparse.Namespace(plan=str(plan_path), output=str(output_path))
    cmd_render(r)

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
