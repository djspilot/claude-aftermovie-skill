"""Clip-level analysis: orchestrates probe + telemetry + motion + audio."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from aftermovie.analyze.audio import measure_audio_energy
from aftermovie.analyze.faces import available as faces_available
from aftermovie.analyze.faces import detect_per_second
from aftermovie.analyze.motion import measure_motion_energy
from aftermovie.config import VIDEO_EXTS
from aftermovie.ffmpeg_cmd import ffprobe_json, log
from aftermovie.telemetry.gpmf import extract_gpmf_track, parse_gpmf_motion
from aftermovie.telemetry.hilight import read_hilight_tags
from aftermovie.types import ClipInfo


def analyze_clip(path: Path) -> ClipInfo | None:
    """Run full feature extraction on a single video file."""
    try:
        info = ffprobe_json(path)
    except subprocess.CalledProcessError:
        log(f"  skip (probe failed): {path.name}")
        return None
    duration = float(info.get("format", {}).get("duration", 0))
    if duration < 0.3:
        log(f"  skip (too short): {path.name}")
        return None
    vstream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if not vstream:
        return None
    fps_str = vstream.get("avg_frame_rate", "30/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    width = int(vstream.get("width", 1920))
    height = int(vstream.get("height", 1080))

    is_short_form = duration < 4.0

    hilights = read_hilight_tags(path)
    gpmf_blob = extract_gpmf_track(path)
    has_gpmf = gpmf_blob is not None and len(gpmf_blob) > 100
    motion = parse_gpmf_motion(gpmf_blob) if has_gpmf else {"accl_mag": [], "gyro_mag": [], "gps_speed": []}

    log(f"  {path.name}  ({duration:.1f}s, {fps:.0f}fps"
        f"{', GPMF' if has_gpmf else ''}"
        f"{', ' + str(len(hilights)) + ' hilights' if hilights else ''})")

    motion_energy = measure_motion_energy(path, duration)
    audio_energy = measure_audio_energy(path, duration)
    face_bboxes: list[dict | None] = (
        detect_per_second(path, duration) if faces_available() else []
    )

    def per_second(arr: list[float], target_len: int) -> list[float]:
        if not arr or target_len == 0:
            return [0.0] * target_len
        bucket = max(1, len(arr) // target_len)
        return [
            (max(arr[i * bucket : (i + 1) * bucket]) if arr[i * bucket : (i + 1) * bucket] else 0.0)
            for i in range(target_len)
        ]

    n_sec = max(1, int(duration))

    return ClipInfo(
        path=str(path),
        duration_s=duration,
        fps=fps,
        width=width,
        height=height,
        has_gpmf=has_gpmf,
        hilight_tags_ms=hilights,
        motion_energy=motion_energy,
        audio_energy=audio_energy,
        accl_peaks=per_second(motion["accl_mag"], n_sec),
        gps_speed=per_second(motion["gps_speed"], n_sec),
        is_short_form=is_short_form,
        face_bboxes=face_bboxes,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    folder = Path(args.clips).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    if not files:
        sys.exit(f"No video files found in {folder} "
                 f"(looking for: {', '.join(sorted(VIDEO_EXTS))})")
    log(f"Analyzing {len(files)} clips in {folder}")
    catalog = []
    for f in files:
        info = analyze_clip(f)
        if info:
            catalog.append(asdict(info))
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clips": catalog}, indent=2))
    log(f"Wrote catalog → {out}")
