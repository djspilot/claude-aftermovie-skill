"""Render pipeline.

Two paths:
    Fast (concat-demuxer)   — all transitions are hard cuts AND no titles
                              AND audio_mix == "music_only".
    Slow (filter_complex)   — anything else: crossfades, whips, titles, or
                              clip-audio mixing.
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aftermovie.config import DEFAULT_FPS, DEFAULT_RES, resolve_lut
from aftermovie.ffmpeg_cmd import log, run, run_with_progress
from aftermovie.render.audio_mix import filtergraph as audio_filtergraph
from aftermovie.render.audio_mix import needs_clip_audio
from aftermovie.render.chip import detect_chip
from aftermovie.render.encoder import (
    EncoderProfile,
    hwaccel_input_flags,
    select_from_env,
    vfilter_input_shim,
    vfilter_output_guard,
)
from aftermovie.render.filters import aspect_filter
from aftermovie.render.parallel import choose_max_workers, parallel_prerender
from aftermovie.render.reframe import crop_x_expr_for_entry
from aftermovie.render.titles import (
    build_overlay_chain,
    render_title_png,
    resolve_title_times,
)
from aftermovie.render.transitions import build_xfade_graph, has_non_cut


# ---- progress callback Interface ------------------------------------------

@dataclass(frozen=True)
class ProgressEvent:
    """One observation of the render's progress, surfaced to the caller's UI.

    `stage` is the coarse phase (`prerender` / `assemble` / `mux`). `stage_index`
    / `stage_total` count items within a multi-step stage — for `prerender`
    that's clip 1..N out of len(entries); for `assemble`/`mux` both equal 1.
    `fraction_in_stage` is 0..1 (frames done out of frames-in-this-step), so
    the upstream weighting code can map it onto the overall % budget.

    `current_pid` is the ffmpeg subprocess pid driving the current step (or
    None at stage boundaries between subprocess spawns). The select GUI
    surfaces it so `kill -INT <pid>` from the operator's terminal aborts the
    correct process.
    """

    stage: str
    stage_index: int
    stage_total: int
    fraction_in_stage: float
    current_pid: int | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def _audio_interest_threshold() -> float:
    """Threshold [0..1] below which a cut's clip audio is muted at render time."""
    import os
    raw = os.environ.get("AFTERMOVIE_AUDIO_INTEREST_THRESHOLD", "")
    try:
        return max(0.0, min(1.0, float(raw))) if raw else 0.35
    except ValueError:
        return 0.35


def _source_has_audio(src: Path) -> bool:
    """ffprobe whether the source has at least one audio stream."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False


def _aspect_dims(aspect: str, target_res: str) -> tuple[int, int]:
    w, h = (int(x) for x in target_res.split("x"))
    if aspect == "9:16":
        w, h = min(w, h), max(w, h)
        if w == h:
            w, h = 1080, 1920
    elif aspect == "1:1":
        w = h = min(w, h)
    return w, h


def _ramp_speeds(entry: dict) -> tuple[float, float]:
    """Return (speed_start, speed_end) for the entry.

    When only `speed` is set (no ramp data) both ends equal `speed`.
    Ramps with start/end within 0.05 of each other collapse to constant
    speed at the average — too subtle to be worth a setpts expression.
    """
    speed = float(entry.get("speed", 1.0))
    s_start = float(entry.get("speed_start", speed))
    s_end = float(entry.get("speed_end", speed))
    if abs(s_start - s_end) <= 0.05:
        avg = (s_start + s_end) / 2.0
        return (avg, avg)
    return (s_start, s_end)


def _prerender_clip(entry: dict, out_clip: Path, *,
                    aspect: str, target_res: str, target_fps: int,
                    lut: Path | None, keep_audio: bool,
                    encoder: EncoderProfile,
                    enable_reframe: bool = True,
                    progress_cb: ProgressCallback | None = None,
                    stage_index: int = 0,
                    stage_total: int = 0) -> bool:
    src = Path(entry["source"])
    duration = entry["end_s"] - entry["start_s"]
    s_start, s_end = _ramp_speeds(entry)
    is_ramp = s_start != s_end
    # When ramping, use the arithmetic mean as the "effective" speed for
    # downstream math (slot-fill, audio atempo). The harmonic mean would
    # be marginally more accurate for output-length prediction but the
    # difference is well under our beat-sync tolerance.
    speed = (s_start + s_end) / 2.0 if is_ramp else s_start
    # The slot the planner wanted to fill. If the source ran short we still
    # want the prerendered clip to be `out_duration_s` long so beat-sync holds
    # — we hold the last frame (tpad) and silence-pad audio.
    slot_dur = float(entry.get("out_duration_s", duration / max(speed, 0.0001)))

    target_w, target_h = _aspect_dims(aspect, target_res)
    vfilter: list[str] = []

    # B2: when -hwaccel videotoolbox is in effect, decoded frames arrive as
    # VT hardware surfaces. The first filter must hwdownload them to NV12 so
    # the rest of the chain (scale/crop/setpts/lut3d/tpad) can run on the
    # CPU. SW profiles return None and we skip the shim entirely.
    input_shim = vfilter_input_shim(encoder)
    if input_shim:
        vfilter.append(input_shim)

    reframe_filter = None
    if (enable_reframe and aspect == "9:16"
            and entry.get("face_bboxes")
            and entry.get("source_width") and entry.get("source_height")):
        reframe_filter = crop_x_expr_for_entry(
            entry,
            int(entry["source_width"]),
            int(entry["source_height"]),
            target_w, target_h,
        )

    if reframe_filter:
        # crop → scale to target res (the crop already enforces the aspect ratio).
        vfilter.append(reframe_filter)
        vfilter.append(f"scale={target_w}:{target_h}")
    else:
        # Letterbox when the source is meaningfully narrower than the target
        # frame (vertical / near-vertical videos on 16:9 output). Same rule
        # as analyze/still_filters.py:_should_letterbox so video + photo
        # behaviour matches.
        src_w = int(entry.get("source_width", 0) or 0)
        src_h = int(entry.get("source_height", 0) or 0)
        if (src_w > 0 and src_h > 0
                and (src_w / src_h) < (target_w / target_h) * 0.85):
            # Pillarbox/letterbox with literal black bars.
            vfilter.append(
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black"
            )
        else:
            vfilter.append(aspect_filter(aspect, target_res))

    if is_ramp:
        # Two-segment speed ramp via time-varying setpts. The expression
        # linearly interpolates the PTS scale factor (1/speed) from
        # f0=1/s_start at input time T=0 to f1=1/s_end at T=duration.
        # ffmpeg evaluates per-frame: new_pts = (f0 + (f1-f0)*T/D) * old_pts.
        # The resulting visible motion ramp matches the (s_start, s_end)
        # endpoints; the actual output length is approximately
        # `duration * (f0+f1)/2`, with `-t slot_dur` truncating any overshoot
        # and tpad below padding any undershoot to keep beat-sync exact.
        f0 = 1.0 / max(s_start, 0.0001)
        f1 = 1.0 / max(s_end, 0.0001)
        d_safe = max(duration, 0.0001)
        vfilter.append(
            f"setpts='({f0:.4f} + ({f1 - f0:+.4f})*(T/{d_safe:.4f}))*PTS'"
        )
    elif speed != 1.0:
        vfilter.append(f"setpts={1.0/speed:.4f}*PTS")
    if lut:
        vfilter.append(f"lut3d={lut.as_posix()}")
    vfilter.append(f"fps={target_fps}")
    # Hold the last frame if we couldn't read enough source to fill the slot.
    if is_ramp:
        # Average PTS factor over the ramp ≈ arithmetic mean of f0,f1.
        native_out_dur = duration * ((1.0 / max(s_start, 0.0001)) +
                                     (1.0 / max(s_end, 0.0001))) / 2.0
    else:
        native_out_dur = duration / max(speed, 0.0001)
    pad_dur = max(0.0, slot_dur - native_out_dur)
    if pad_dur > 0.05:
        vfilter.append(f"tpad=stop_mode=clone:stop_duration={pad_dur:.3f}")
    # B5: VT encoders are picky about input pixel formats. Re-assert the
    # encoder's target pixfmt at the tail of the chain so setpts/tpad/lut3d
    # outputs don't fall through to a format VT rejects.
    output_guard = vfilter_output_guard(encoder)
    if output_guard:
        vfilter.append(output_guard)
    vf = ",".join(vfilter)

    final_dur = slot_dur
    fade = min(0.04, final_dur / 4) if final_dur > 0 else 0.0

    cmd = ["ffmpeg", "-y", "-v", "error"]
    audio_input_idx = None
    if keep_audio and not _source_has_audio(src):
        # Inject a silent stereo track matched to the source duration so every
        # prerendered clip carries an audio stream — keeps concat/sidechain stable.
        cmd += ["-f", "lavfi", "-t", f"{duration:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_input_idx = 0
    # B2: HW-decode the source when the encoder is a VT variant. The flags
    # must come *before* the `-i` for the input they apply to; the silent
    # anullsrc input above stays SW (CPU-side lavfi), which is what we want.
    cmd += hwaccel_input_flags(encoder)
    cmd += [
        "-ss", f"{entry['start_s']:.3f}",
        "-i", str(src),
        "-vf", vf,
    ]
    video_input_idx = 1 if audio_input_idx is not None else 0
    if keep_audio:
        a_steps: list[str] = []
        # Intelligent-audio gate: when this cut has little voice-band content,
        # mute it pre-mix so the duck sidechain stays clean and the music
        # plays unencumbered. Threshold from env (default 0.35).
        threshold = _audio_interest_threshold()
        interest = float(entry.get("audio_interest", 1.0))
        if threshold > 0 and interest < threshold:
            a_steps.append("volume=0")
        if speed != 1.0:
            # atempo cannot ramp linearly across a clip, so when the video
            # ramps we apply atempo at the average speed. Ramps tend to
            # coincide with action moments where music dominates the mix
            # anyway, so the small drift from a "true" audio ramp is
            # inaudible after the music duck.
            a_steps.append(f"atempo={max(0.5, min(2.0, speed)):.4f}")
        if pad_dur > 0.05:
            # Pad audio with silence so it matches the tpad'd video length.
            a_steps.append(f"apad=pad_dur={pad_dur:.3f}")
        if fade > 0:
            a_steps.append(f"afade=t=in:st=0:d={fade:.3f}")
            a_steps.append(f"afade=t=out:st={max(0.0, final_dur - fade):.3f}:d={fade:.3f}")
        # Hard-cap audio length so apad doesn't run on indefinitely.
        a_steps.append(f"atrim=duration={final_dur:.3f}")
        a_steps.append("asetpts=N/SR/TB")
        a_filter = ",".join(a_steps) if a_steps else "anull"
        if audio_input_idx is not None:
            cmd += ["-filter_complex",
                    f"[{video_input_idx}:v]{vf}[v];[{audio_input_idx}:a]{a_filter}[a]",
                    "-map", "[v]", "-map", "[a]"]
            # remove the -vf flag we already moved into filter_complex
            vf_idx = cmd.index("-vf")
            del cmd[vf_idx:vf_idx + 2]
        else:
            cmd += ["-af", a_filter]
        cmd += ["-c:a", "aac", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += [
        *encoder.video_args,
        "-pix_fmt", encoder.pix_fmt,
        "-t", f"{slot_dur:.3f}",
        str(out_clip),
    ]
    # Per-prerender frame target: planner-provided slot length × output fps.
    # `-progress pipe:1` emits `frame=N` per stats period, so percent-done
    # = min(1.0, frame / total_frames). Anything within ±5 % is fine for UX.
    total_frames = max(1, int(slot_dur * target_fps))

    def _on_block(block: dict) -> None:
        if progress_cb is None:
            return
        frame = block.get("frame") or 0
        # `progress=end` emits even when `frame` lagged; clamp at 1.0.
        frac = min(1.0, max(0.0, frame / total_frames))
        if block.get("progress") == "end":
            frac = 1.0
        progress_cb(ProgressEvent(
            stage="prerender",
            stage_index=stage_index,
            stage_total=stage_total,
            fraction_in_stage=frac,
            current_pid=block.get("_pid"),
        ))

    captured_pid: list[int] = []

    def _on_pid(pid: int) -> None:
        captured_pid.append(pid)
        if progress_cb is not None:
            progress_cb(ProgressEvent(
                stage="prerender",
                stage_index=stage_index,
                stage_total=stage_total,
                fraction_in_stage=0.0,
                current_pid=pid,
            ))

    try:
        run_with_progress(cmd, _on_block, check=True,
                          total_frames=total_frames, on_pid=_on_pid)
        return True
    except subprocess.CalledProcessError:
        log(f"  ! failed to render {src.name} — skipping")
        return False


def _prerender_worker(
    entry: dict,
    out_clip: Path,
    stage_index: int,
    stage_total: int,
    *,
    aspect: str,
    target_res: str,
    target_fps: int,
    lut: Path | None,
    keep_audio: bool,
    encoder: EncoderProfile,
    enable_reframe: bool,
) -> bool:
    """Picklable Adapter that calls `_prerender_clip` from a worker process.

    The parallel pool can't ferry a `ProgressCallback` across the process
    boundary (closures aren't picklable, and even module-level callables
    would need a reverse Queue to reach the GUI's progress lock). So this
    Adapter strips `progress_cb` — the parent process emits coalesced
    `ProgressEvent`s after each future resolves; see
    `aftermovie.render.parallel.parallel_prerender`.

    Stage_index/total are still threaded in so legacy logs that reference
    them stay accurate. They're not used to drive progress here.
    """
    return _prerender_clip(
        entry, out_clip,
        aspect=aspect, target_res=target_res, target_fps=target_fps,
        lut=lut, keep_audio=keep_audio,
        encoder=encoder,
        enable_reframe=enable_reframe,
        progress_cb=None,
        stage_index=stage_index,
        stage_total=stage_total,
    )


def _compensated_render_entry(entry: dict, *, transitions_active: bool,
                              is_first: bool) -> tuple[dict, float, float]:
    """Return (render_entry, planned_duration, prerender_duration).

    xfade/acrossfade overlaps the incoming clip by `transition_in.duration_s`.
    To preserve the planner's visible timeline length, the incoming clip must
    be prerendered longer by exactly that overlap.
    """
    planned_duration = float(entry.get(
        "out_duration_s",
        (entry["end_s"] - entry["start_s"]) / entry.get("speed", 1.0),
    ))
    render_duration = planned_duration
    if transitions_active and not is_first:
        transition = entry.get("transition_in") or {}
        if (transition.get("kind") or "cut") != "cut":
            render_duration += float(transition.get("duration_s") or 0.0)

    render_entry = dict(entry)
    render_entry["out_duration_s"] = render_duration
    return render_entry, planned_duration, render_duration


def cmd_render(args: argparse.Namespace, *,
               progress_cb: ProgressCallback | None = None) -> None:
    plan = json.loads(Path(args.plan).expanduser().read_text())
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lut = resolve_lut(plan.get("lut"))
    target_res = plan.get("resolution", DEFAULT_RES)
    target_fps = plan.get("fps", DEFAULT_FPS)
    aspect = plan.get("aspect", "16:9")
    audio_mix = plan.get("audio_mix", "music_only")
    titles = plan.get("titles", [])
    theme = plan.get("theme", "cinematic")
    entries = plan["entries"]
    enable_reframe = bool(plan.get("reframe", True))

    transitions_active = has_non_cut(entries)
    keep_audio = needs_clip_audio(audio_mix)
    use_filter_complex = transitions_active or bool(titles)

    # B1+B3: resolve the encoder profile once per render. detect_chip() is
    # cheap (one sysctl on darwin, nothing elsewhere) and select_from_env
    # honours AFTERMOVIE_VIDEO_CODEC while falling back gracefully when an
    # env value asks for a profile ffmpeg doesn't ship.
    chip = detect_chip()
    encoder = select_from_env(chip)
    # B4: pool size from the chip × encoder heuristic, honouring
    # AFTERMOVIE_RENDER_WORKERS for ops tuning / deterministic CI.
    max_workers = choose_max_workers(chip, encoder)

    log(f"Rendering {len(entries)} cuts → {output.name}")
    log(f"  res={target_res} fps={target_fps} aspect={aspect}"
        f" audio={audio_mix} transitions={'yes' if transitions_active else 'no'}"
        f" titles={'yes' if titles else 'no'}"
        f" encoder={encoder.name}"
        f" workers={max_workers}"
        f"{' lut=' + lut.name if lut else ''}")

    n_entries = len(entries)

    with tempfile.TemporaryDirectory(prefix="aftermovie_") as tmpdir:
        tmp = Path(tmpdir)
        # Pre-build the per-slot render entries + output paths. Doing this
        # upfront keeps the parallel pool ignorant of `_compensated_render_entry`
        # — the worker just gets a finished entry dict + path. It also lets
        # us preserve plan order for the post-parallel filter_complex pass
        # by indexing back into `render_entries` / `durations` using the
        # same slot index the parallel helper returns paths for.
        render_entries: list[dict] = []
        planned_durations: list[float] = []
        render_durations: list[float] = []
        work: list[tuple[dict, Path]] = []
        for i, entry in enumerate(entries):
            render_entry, planned_duration, render_duration = _compensated_render_entry(
                entry, transitions_active=transitions_active, is_first=(i == 0)
            )
            render_entries.append(render_entry)
            planned_durations.append(planned_duration)
            render_durations.append(render_duration)
            work.append((render_entry, tmp / f"clip_{i:04d}.mp4"))

        # B4: factory binds the encode flags so the parallel helper only
        # sees `(entry, out_clip, idx, total) -> bool`. `functools.partial`
        # over a module-level function is picklable, which keeps the
        # ProcessPoolExecutor path viable.
        factory = functools.partial(
            _prerender_worker,
            aspect=aspect, target_res=target_res, target_fps=target_fps,
            lut=lut, keep_audio=keep_audio,
            encoder=encoder,
            enable_reframe=enable_reframe,
        )

        prerender_results = parallel_prerender(
            work, factory,
            max_workers=max_workers,
            progress_cb=progress_cb,
            encoder_name=encoder.name,
        )

        # Filter dropped clips, preserving plan order. `parallel_prerender`
        # returns `None` in failed slots — same skip semantics as the
        # legacy for-loop's `if not ok: continue`.
        clip_paths: list[Path] = []
        durations: list[float] = []
        kept_render_entries: list[dict] = []
        planned_total = 0.0
        for idx, out_clip in enumerate(prerender_results):
            if out_clip is None:
                continue
            clip_paths.append(out_clip)
            kept_render_entries.append(render_entries[idx])
            planned_total += planned_durations[idx]
            durations.append(render_durations[idx])
        render_entries = kept_render_entries

        if not clip_paths:
            sys.exit("No clips rendered. Aborting.")

        intermediate = tmp / "intermediate.mp4"

        if use_filter_complex:
            frame_w, frame_h = _aspect_dims(aspect, target_res)
            resolved_titles = resolve_title_times(titles, planned_total)
            title_pngs: list[Path] = []
            for i, t in enumerate(resolved_titles):
                if not t.get("text"):
                    continue
                png = tmp / f"title_{i}.png"
                render_title_png(t["text"], theme, frame_w, frame_h, png)
                title_pngs.append(png)
            _render_filter_complex(
                clip_paths, durations, render_entries,
                title_pngs=title_pngs,
                title_times=[t for t in resolved_titles if t.get("text")],
                total_duration_s=planned_total,
                target_fps=target_fps,
                keep_audio=keep_audio,
                encoder=encoder,
                out=intermediate,
                progress_cb=progress_cb,
            )
        else:
            _render_concat(clip_paths, intermediate, keep_audio=keep_audio,
                           progress_cb=progress_cb)

        _final_mux(intermediate, plan, output, audio_mix=audio_mix,
                   keep_audio=keep_audio, target_fps=target_fps,
                   progress_cb=progress_cb)

    log(f"Done → {output}")


def _run_assemble_with_progress(
    cmd: list[str], progress_cb: ProgressCallback | None,
    *, stage: str = "assemble", total_frames: int | None = None,
) -> None:
    """Run an `assemble` / `mux` ffmpeg command and surface progress.

    We don't always know `total_frames` for these single-pass commands (the
    concat-copy path emits very few `frame=` ticks; final-mux copies video
    and re-encodes audio), so when `total_frames` is None we use the
    `out_time_ms` ratio against the planned duration the caller passes via
    `total_frames` instead. Either way the callback sees `fraction_in_stage`
    monotonically advancing from 0 → 1.
    """
    if progress_cb is None:
        run(cmd, check=True)
        return

    def _on_block(block: dict) -> None:
        frac = 0.0
        if total_frames and (block.get("frame") or 0) > 0:
            frac = min(1.0, max(0.0, block["frame"] / total_frames))
        if block.get("progress") == "end":
            frac = 1.0
        progress_cb(ProgressEvent(
            stage=stage, stage_index=1, stage_total=1,
            fraction_in_stage=frac, current_pid=block.get("_pid"),
        ))

    def _on_pid(pid: int) -> None:
        progress_cb(ProgressEvent(
            stage=stage, stage_index=1, stage_total=1,
            fraction_in_stage=0.0, current_pid=pid,
        ))

    run_with_progress(cmd, _on_block, check=True,
                      total_frames=total_frames, on_pid=_on_pid)


def _render_concat(clip_paths: list[Path], out: Path, *, keep_audio: bool,
                   progress_cb: ProgressCallback | None = None) -> None:
    concat_file = out.parent / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in clip_paths))
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(out),
    ]
    # concat-copy is so fast we rarely see more than one `-progress` block,
    # but plumbing the callback through keeps the assemble stage's % bar
    # ticking instead of stalling at 0 until the subprocess exits.
    _run_assemble_with_progress(cmd, progress_cb)


def _render_filter_complex(clip_paths: list[Path], durations: list[float],
                           entries: list[dict],
                           *, title_pngs: list[Path], title_times: list[dict],
                           total_duration_s: float,
                           target_fps: int, keep_audio: bool,
                           encoder: EncoderProfile,
                           out: Path,
                           progress_cb: ProgressCallback | None = None) -> None:
    """Build a single ffmpeg call with xfade transitions and PNG-overlay titles."""
    inputs: list[str] = []
    # Prerendered clips are already in the encoder's pixfmt, so we don't add
    # `-hwaccel videotoolbox` here — the HW-decode win lives in _prerender_clip
    # where the inputs are the heavy original sources (HEVC GoPro etc.).
    for p in clip_paths:
        inputs += ["-i", str(p)]
    for p in title_pngs:
        inputs += ["-loop", "1", "-i", str(p)]

    n_clips = len(clip_paths)
    xfade_graph, video_label = build_xfade_graph(
        n_clips, durations, entries, target_fps=target_fps,
    )

    parts: list[str] = []
    if xfade_graph:
        parts.append(xfade_graph)

    final_v = video_label
    if title_pngs:
        title_inputs = [(n_clips + i, t) for i, t in enumerate(title_times)]
        overlay = build_overlay_chain(video_label, title_inputs, out_label="v_out")
        if overlay:
            parts.append(overlay)
            final_v = "v_out"

    if keep_audio:
        # If any soft/auto crossfade transitions are active, glue audio with
        # acrossfade so seams match the video xfade. Otherwise plain concat.
        soft_audio = any(
            (e.get("transition_in", {}).get("kind") or "cut") != "cut"
            for e in entries
        )
        if soft_audio and n_clips > 1:
            last = "0:a"
            for i in range(1, n_clips):
                tdur = float(entries[i].get("transition_in", {}).get("duration_s") or 0.1)
                tdur = max(0.05, min(0.9, tdur))
                label = f"axf{i}" if i < n_clips - 1 else "a_out"
                parts.append(
                    f"[{last}][{i}:a]acrossfade=d={tdur:.3f}:c1=tri:c2=tri[{label}]"
                )
                last = label
        else:
            a_concat = "".join(f"[{i}:a]" for i in range(n_clips))
            parts.append(f"{a_concat}concat=n={n_clips}:v=0:a=1[a_out]")

    filter_complex = ";".join(parts) if parts else ""

    cmd = ["ffmpeg", "-y", "-v", "error"] + inputs
    if filter_complex:
        cmd += ["-filter_complex", filter_complex]
    # build_xfade_graph now normalizes the single-input case too, so final_v
    # is always a labeled stream (e.g. "v0"/"seg0"/"x42").
    cmd += ["-map", f"[{final_v}]"]
    if keep_audio:
        cmd += ["-map", "[a_out]", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += [
        *encoder.video_args,
        "-pix_fmt", encoder.pix_fmt,
        "-r", str(target_fps),
        "-t", f"{total_duration_s:.3f}",
        str(out),
    ]
    total_frames = max(1, int(total_duration_s * target_fps))
    _run_assemble_with_progress(cmd, progress_cb, stage="assemble",
                                total_frames=total_frames)


def _final_mux(intermediate: Path, plan: dict, output: Path,
               *, audio_mix: str, keep_audio: bool, target_fps: int = DEFAULT_FPS,
               progress_cb: ProgressCallback | None = None) -> None:
    song = plan["song"]
    music_db = plan.get("music_db", -8)
    a_filter = audio_filtergraph(audio_mix, music_db)

    # Probe intermediate duration so we can fade the final audio out at the tail.
    try:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(intermediate)],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        dur = 0.0
    # Scale the fade to the output length: 1.5s tail for normal-length edits,
    # shorter when the output is very short. Always at least 200ms.
    fade_out_d = max(0.2, min(1.5, dur * 0.05))
    if dur > fade_out_d + 0.1:
        fade_start = max(0.0, dur - fade_out_d)
        a_filter = (
            a_filter.replace("[a_out]", "[a_pre]")
            + f";[a_pre]afade=t=out:st={fade_start:.3f}:d={fade_out_d:.3f}[a_out]"
        )

    # Where in the song do the cuts actually live? The planner places the
    # first cut at song_meta.intro_end_s, so we seek the music there so
    # output t=0 lines up with the part of the song that's synced to.
    song_start_s = float(
        plan.get("song_start_s",
                 plan.get("song_meta", {}).get("intro_end_s", 0.0))
    )

    song_inputs: list[str] = []
    if song_start_s > 0.05:
        song_inputs += ["-ss", f"{song_start_s:.3f}"]
    song_inputs += ["-i", song]

    cmd = [
        "ffmpeg", "-y", "-v", "warning", "-stats",
        "-i", str(intermediate),
        *song_inputs,
        "-filter_complex", a_filter,
        "-map", "0:v",
        "-map", "[a_out]",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]
    # `-c:v copy` means video frames stream through verbatim; ffmpeg still
    # emits `frame=N` ticks though, so the mux bar fills smoothly. Use the
    # probed intermediate duration × target_fps as the frame target.
    total_frames = max(1, int(dur * target_fps)) if dur > 0 else None
    _run_assemble_with_progress(cmd, progress_cb, stage="mux",
                                total_frames=total_frames)
