"""Render pipeline.

Two paths:
    Fast (concat-demuxer)   — all transitions are hard cuts AND no titles
                              AND audio_mix == "music_only".
    Slow (filter_complex)   — anything else: crossfades, whips, titles, or
                              clip-audio mixing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from aftermovie.config import DEFAULT_FPS, DEFAULT_RES, resolve_lut
from aftermovie.ffmpeg_cmd import log, run
from aftermovie.render.audio_mix import filtergraph as audio_filtergraph
from aftermovie.render.audio_mix import needs_clip_audio
from aftermovie.render.filters import aspect_filter
from aftermovie.render.reframe import crop_x_expr_for_entry
from aftermovie.render.titles import (
    build_overlay_chain,
    render_title_png,
    resolve_title_times,
)
from aftermovie.render.transitions import build_xfade_graph, has_non_cut


def _aspect_dims(aspect: str, target_res: str) -> tuple[int, int]:
    w, h = (int(x) for x in target_res.split("x"))
    if aspect == "9:16":
        w, h = min(w, h), max(w, h)
        if w == h:
            w, h = 1080, 1920
    elif aspect == "1:1":
        w = h = min(w, h)
    return w, h


def _prerender_clip(entry: dict, out_clip: Path, *,
                    aspect: str, target_res: str, target_fps: int,
                    lut: Path | None, keep_audio: bool,
                    enable_reframe: bool = True) -> bool:
    src = Path(entry["source"])
    duration = entry["end_s"] - entry["start_s"]
    speed = entry.get("speed", 1.0)

    target_w, target_h = _aspect_dims(aspect, target_res)
    vfilter: list[str] = []

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
        vfilter.append(aspect_filter(aspect, target_res))

    if speed != 1.0:
        vfilter.append(f"setpts={1.0/speed:.4f}*PTS")
    if lut:
        vfilter.append(f"lut3d={lut.as_posix()}")
    vfilter.append(f"fps={target_fps}")
    vf = ",".join(vfilter)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{entry['start_s']:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vf", vf,
    ]
    if keep_audio:
        cmd += [
            "-af", f"atempo={max(0.5, min(2.0, speed)):.4f}" if speed != 1.0 else "anull",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
        ]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out_clip),
    ]
    try:
        run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        log(f"  ! failed to render {src.name} — skipping")
        return False


def cmd_render(args: argparse.Namespace) -> None:
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

    log(f"Rendering {len(entries)} cuts → {output.name}")
    log(f"  res={target_res} fps={target_fps} aspect={aspect}"
        f" audio={audio_mix} transitions={'yes' if transitions_active else 'no'}"
        f" titles={'yes' if titles else 'no'}"
        f"{' lut=' + lut.name if lut else ''}")

    with tempfile.TemporaryDirectory(prefix="aftermovie_") as tmpdir:
        tmp = Path(tmpdir)
        clip_paths: list[Path] = []
        durations: list[float] = []

        for i, entry in enumerate(entries):
            out_clip = tmp / f"clip_{i:04d}.mp4"
            ok = _prerender_clip(
                entry, out_clip,
                aspect=aspect, target_res=target_res, target_fps=target_fps,
                lut=lut, keep_audio=keep_audio,
                enable_reframe=enable_reframe,
            )
            if not ok:
                continue
            clip_paths.append(out_clip)
            durations.append((entry["end_s"] - entry["start_s"]) / entry.get("speed", 1.0))

        if not clip_paths:
            sys.exit("No clips rendered. Aborting.")

        intermediate = tmp / "intermediate.mp4"

        if use_filter_complex:
            frame_w, frame_h = _aspect_dims(aspect, target_res)
            resolved_titles = resolve_title_times(titles, sum(durations))
            title_pngs: list[Path] = []
            for i, t in enumerate(resolved_titles):
                if not t.get("text"):
                    continue
                png = tmp / f"title_{i}.png"
                render_title_png(t["text"], theme, frame_w, frame_h, png)
                title_pngs.append(png)
            _render_filter_complex(
                clip_paths, durations, entries[:len(clip_paths)],
                title_pngs=title_pngs,
                title_times=[t for t in resolved_titles if t.get("text")],
                total_duration_s=sum(durations),
                target_fps=target_fps,
                keep_audio=keep_audio,
                out=intermediate,
            )
        else:
            _render_concat(clip_paths, intermediate, keep_audio=keep_audio)

        _final_mux(intermediate, plan, output, audio_mix=audio_mix, keep_audio=keep_audio)

    log(f"Done → {output}")


def _render_concat(clip_paths: list[Path], out: Path, *, keep_audio: bool) -> None:
    concat_file = out.parent / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in clip_paths))
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(out),
    ]
    run(cmd)


def _render_filter_complex(clip_paths: list[Path], durations: list[float],
                           entries: list[dict],
                           *, title_pngs: list[Path], title_times: list[dict],
                           total_duration_s: float,
                           target_fps: int, keep_audio: bool,
                           out: Path) -> None:
    """Build a single ffmpeg call with xfade transitions and PNG-overlay titles."""
    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", str(p)]
    for p in title_pngs:
        inputs += ["-loop", "1", "-i", str(p)]

    n_clips = len(clip_paths)
    xfade_graph, video_label = build_xfade_graph(n_clips, durations, entries)

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
        a_concat = "".join(f"[{i}:a]" for i in range(n_clips))
        parts.append(f"{a_concat}concat=n={n_clips}:v=0:a=1[a_out]")

    filter_complex = ";".join(parts) if parts else ""

    cmd = ["ffmpeg", "-y", "-v", "error"] + inputs
    if filter_complex:
        cmd += ["-filter_complex", filter_complex]
    map_v = "0:v" if final_v == "0:v" else f"[{final_v}]"
    cmd += ["-map", map_v]
    if keep_audio:
        cmd += ["-map", "[a_out]", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        "-t", f"{total_duration_s:.3f}",
        str(out),
    ]
    run(cmd)


def _final_mux(intermediate: Path, plan: dict, output: Path,
               *, audio_mix: str, keep_audio: bool) -> None:
    song = plan["song"]
    music_db = plan.get("music_db", -8)
    a_filter = audio_filtergraph(audio_mix, music_db)

    cmd = [
        "ffmpeg", "-y", "-v", "warning", "-stats",
        "-i", str(intermediate),
        "-i", song,
        "-filter_complex", a_filter,
        "-map", "0:v",
        "-map", "[a_out]",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]
    run(cmd)
