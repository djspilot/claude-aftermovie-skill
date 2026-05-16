"""Render pipeline: trim each cut, concat, mux song."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from aftermovie.config import DEFAULT_FPS, DEFAULT_RES, resolve_lut
from aftermovie.ffmpeg_cmd import log, run
from aftermovie.render.filters import aspect_filter


def cmd_render(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).expanduser().read_text())
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lut = resolve_lut(plan.get("lut"))
    target_res = plan.get("resolution", DEFAULT_RES)
    target_fps = plan.get("fps", DEFAULT_FPS)
    aspect = plan.get("aspect", "16:9")

    log(f"Rendering {len(plan['entries'])} cuts → {output.name}")
    log(f"  res={target_res} fps={target_fps} aspect={aspect}"
        f"{' lut=' + lut.name if lut else ''}")

    with tempfile.TemporaryDirectory(prefix="aftermovie_") as tmpdir:
        tmp = Path(tmpdir)
        concat_lines = []
        for i, entry in enumerate(plan["entries"]):
            src = Path(entry["source"])
            out_clip = tmp / f"clip_{i:04d}.mp4"
            duration = entry["end_s"] - entry["start_s"]
            speed = entry.get("speed", 1.0)
            vfilter = [aspect_filter(aspect, target_res)]
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
                "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                str(out_clip),
            ]
            try:
                run(cmd, check=True)
            except subprocess.CalledProcessError:
                log(f"  ! failed to render cut {i} from {src.name} — skipping")
                continue
            concat_lines.append(f"file '{out_clip.as_posix()}'")

        if not concat_lines:
            sys.exit("No clips rendered. Aborting.")

        concat_file = tmp / "concat.txt"
        concat_file.write_text("\n".join(concat_lines))

        video_only = tmp / "video_only.mp4"
        run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(video_only),
        ])

        song = plan["song"]
        music_db = plan.get("music_db", -8)
        run([
            "ffmpeg", "-y", "-v", "warning", "-stats",
            "-i", str(video_only),
            "-i", song,
            "-filter_complex", f"[1:a]volume={music_db}dB[m]",
            "-map", "0:v", "-map", "[m]",
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(output),
        ])

    log(f"Done → {output}")
