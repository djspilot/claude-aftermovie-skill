#!/usr/bin/env python3
"""
aftermovie.py — Beat-synced highlight video generator for macOS.

Three stages, runnable independently or together:
  analyze : scan a folder of video clips, extract telemetry + motion features
  score   : analyze the song, score clips, build an edit plan
  render  : execute the plan via ffmpeg

Or just `auto` to do all three in one shot.

Designed to handle mixed footage: GoPro (with GPMF telemetry + HiLight tags),
iPhone video, Live Photos, drone clips, screen recordings — anything ffmpeg
can read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# -------------------------------------------------------------------------
# Constants & utilities
# -------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".insv", ".lrv", ".MP4", ".MOV", ".M4V"}
# Live Photos and Motion Photos arrive as paired .heic/.jpg + .mov; we only
# need to read the .mov half.

DEFAULT_TARGET_LEN_S = 90
DEFAULT_FPS = 30
DEFAULT_RES = "1920x1080"

# Sub-clip candidate length range (seconds). Quik uses something like this.
MIN_CLIP_S = 0.4
MAX_CLIP_S = 4.0


def log(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Wrapper around subprocess.run with sensible defaults."""
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def ffprobe_json(path: Path) -> dict[str, Any]:
    """Return ffprobe metadata for a file as a dict."""
    res = run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture=True,
    )
    return json.loads(res.stdout)


# -------------------------------------------------------------------------
# GoPro HiLight tag reader (HMMT atom in MP4 udta box)
# -------------------------------------------------------------------------

def read_hilight_tags(path: Path) -> list[int]:
    """
    Extract HiLight tag timestamps (in milliseconds) from a GoPro MP4.

    HiLight tags live in the moov/udta/HMMT atom, NOT in the GPMF metadata
    stream. Structure:
        HMMT box header (8 bytes: size + 'HMMT')
        uint32_be count
        uint32_be timestamp_ms[count]

    Returns [] for non-GoPro files or files without HiLight tags.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return []
    idx = data.find(b"HMMT")
    if idx < 0:
        return []
    # The 4 bytes immediately before "HMMT" are the box size (big-endian uint32).
    try:
        box_size = struct.unpack(">I", data[idx - 4 : idx])[0]
    except struct.error:
        return []
    payload = data[idx + 4 : idx - 4 + box_size]
    if len(payload) < 4:
        return []
    count = struct.unpack(">I", payload[:4])[0]
    if count > 1000 or len(payload) < 4 + count * 4:
        return []
    return list(struct.unpack(f">{count}I", payload[4 : 4 + count * 4]))


# -------------------------------------------------------------------------
# GPMF telemetry — lightweight reader for ACCL, GYRO, GPS5
# -------------------------------------------------------------------------

def extract_gpmf_track(path: Path) -> bytes | None:
    """
    Pull the GPMF telemetry track out of a GoPro MP4 using ffmpeg.
    Returns the raw GPMF bytes, or None if the file has no telemetry track.
    """
    # First check whether the file has a 'gpmd' / 'GoPro MET' meta stream.
    try:
        info = ffprobe_json(path)
    except subprocess.CalledProcessError:
        return None
    gpmf_index = None
    for s in info.get("streams", []):
        codec = s.get("codec_tag_string", "").lower()
        if codec in ("gpmd", "meta") and s.get("codec_type") == "data":
            gpmf_index = s["index"]
            break
    if gpmf_index is None:
        return None
    # Extract that stream to stdout.
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out_path = tmp.name
    try:
        run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(path),
                "-map", f"0:{gpmf_index}",
                "-c", "copy", "-f", "data",
                out_path,
            ],
            check=False,
        )
        return Path(out_path).read_bytes()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def parse_gpmf_motion(blob: bytes) -> dict[str, list[float]]:
    """
    Parse a GPMF blob and return per-second motion summaries.

    GPMF is a nested KLV format: 4-byte FourCC key, 1 byte type, 1 byte
    structure size, 2 bytes repeat count, then payload (32-bit aligned).

    We only care about a few keys:
      ACCL — 3-axis accelerometer (used for jump detection)
      GYRO — 3-axis gyroscope     (used for steady-footage detection)
      GPS5 — GPS lat/lon/alt/2D-speed/3D-speed (used for speed peaks)

    For each, we return a flat list of magnitudes resampled at ~1 Hz.
    """
    result = {"accl_mag": [], "gyro_mag": [], "gps_speed": []}
    if not blob:
        return result
    pos = 0
    n = len(blob)
    scale_stack: list[list[float]] = [[1.0]]
    current_key = None
    while pos + 8 <= n:
        try:
            key = blob[pos : pos + 4].decode("ascii", errors="replace")
            type_char = chr(blob[pos + 4])
            struct_size = blob[pos + 5]
            repeat = struct.unpack(">H", blob[pos + 6 : pos + 8])[0]
        except (UnicodeDecodeError, struct.error):
            break
        payload_size = struct_size * repeat
        # 32-bit align
        aligned = (payload_size + 3) & ~3
        payload = blob[pos + 8 : pos + 8 + payload_size]
        pos += 8 + aligned

        if type_char == "\x00":
            # Nested container — recurse handled by linear scan.
            continue
        # Track scale for sensor data
        if key == "SCAL" and type_char in ("s", "S", "l", "L"):
            try:
                fmt = ">" + ("h" if type_char == "s" else "H" if type_char == "S" else "i" if type_char == "l" else "I") * repeat
                vals = struct.unpack(fmt, payload)
                scale_stack[-1] = [float(v) if v != 0 else 1.0 for v in vals]
            except struct.error:
                pass
            continue
        if key == "ACCL" and type_char == "s":
            scale = scale_stack[-1][0] if scale_stack[-1] else 1.0
            try:
                samples = struct.unpack(f">{repeat * (struct_size // 2)}h", payload)
            except struct.error:
                continue
            # 3 axes per sample
            for i in range(0, len(samples) - 2, 3):
                x, y, z = (samples[i] / scale, samples[i + 1] / scale, samples[i + 2] / scale)
                mag = (x * x + y * y + z * z) ** 0.5
                result["accl_mag"].append(mag)
        elif key == "GYRO" and type_char == "s":
            scale = scale_stack[-1][0] if scale_stack[-1] else 1.0
            try:
                samples = struct.unpack(f">{repeat * (struct_size // 2)}h", payload)
            except struct.error:
                continue
            for i in range(0, len(samples) - 2, 3):
                x, y, z = (samples[i] / scale, samples[i + 1] / scale, samples[i + 2] / scale)
                mag = (x * x + y * y + z * z) ** 0.5
                result["gyro_mag"].append(mag)
        elif key == "GPS5" and type_char == "l":
            # 5 ints per sample: lat, lon, alt, speed_2d, speed_3d (each scaled).
            try:
                ints = struct.unpack(f">{repeat * 5}i", payload)
            except struct.error:
                continue
            scales = scale_stack[-1] if len(scale_stack[-1]) >= 5 else [1.0] * 5
            for i in range(0, len(ints) - 4, 5):
                speed_2d = ints[i + 3] / (scales[3] if scales[3] else 1.0)
                result["gps_speed"].append(speed_2d)
    return result


# -------------------------------------------------------------------------
# Clip analysis
# -------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single candidate sub-clip (1-5 seconds within a source file)."""
    source: str
    start_s: float
    end_s: float
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    src_fps: float = 30.0  # original frame rate (used to detect slow-mo material)
    is_short: bool = False  # Live Photo / Motion Photo flag


@dataclass
class ClipInfo:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    has_gpmf: bool
    hilight_tags_ms: list[int]
    # Per-second feature arrays (length = ceil(duration_s)):
    motion_energy: list[float]
    audio_energy: list[float]
    accl_peaks: list[float]   # GPMF only
    gps_speed: list[float]    # GPMF only
    is_short_form: bool       # Live Photo / Motion Photo


def measure_motion_energy(path: Path, duration: float) -> list[float]:
    """
    Estimate per-second motion energy using ffmpeg's `signalstats` filter.
    Falls back to scene-detection scores if signalstats fails.

    Returns a list of per-second motion magnitude (arbitrary units, higher = more motion).
    """
    # Sample at 1 fps for speed. The `freezedetect`/`signalstats` filters give
    # us a frame-to-frame brightness/chroma delta that approximates motion.
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-vf", "fps=2,signalstats,metadata=mode=print:file=-",
        "-f", "null", "-",
    ]
    try:
        res = run(cmd, capture=True, check=False)
        text = res.stdout + res.stderr
    except Exception:
        return [0.0] * max(1, int(duration))
    # Parse YDIF (frame-to-frame luma diff) lines.
    pattern = re.compile(r"YDIF=([\d.]+)")
    diffs = [float(m.group(1)) for m in pattern.finditer(text)]
    if not diffs:
        return [0.0] * max(1, int(duration))
    # Bin into per-second buckets (we sampled at 2 fps so 2 diffs per second).
    n_sec = max(1, int(duration))
    binned = []
    for i in range(n_sec):
        chunk = diffs[i * 2 : (i + 1) * 2]
        binned.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return binned


def measure_audio_energy(path: Path, duration: float) -> list[float]:
    """Per-second RMS of the audio track (voices, cheering, music)."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f", "null", "-",
    ]
    try:
        res = run(cmd, capture=True, check=False)
        text = res.stdout + res.stderr
    except Exception:
        return [0.0] * max(1, int(duration))
    pattern = re.compile(r"RMS_level=(-?[\d.]+)")
    levels = [float(m.group(1)) for m in pattern.finditer(text)]
    if not levels:
        return [0.0] * max(1, int(duration))
    # RMS is in dB; convert to a positive 0-1 scale (silence ~ -60dB, loud ~ -6dB).
    normed = [max(0.0, min(1.0, (lvl + 60) / 54)) for lvl in levels]
    n_sec = max(1, int(duration))
    bucket = max(1, len(normed) // n_sec)
    binned = [
        sum(normed[i * bucket : (i + 1) * bucket]) / bucket
        for i in range(n_sec)
    ]
    return binned


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

    is_short_form = duration < 4.0  # Live Photos are typically ~3s

    hilights = read_hilight_tags(path)
    gpmf_blob = extract_gpmf_track(path)
    has_gpmf = gpmf_blob is not None and len(gpmf_blob) > 100
    motion = parse_gpmf_motion(gpmf_blob) if has_gpmf else {"accl_mag": [], "gyro_mag": [], "gps_speed": []}

    log(f"  {path.name}  ({duration:.1f}s, {fps:.0f}fps"
        f"{', GPMF' if has_gpmf else ''}"
        f"{', ' + str(len(hilights)) + ' hilights' if hilights else ''})")

    motion_energy = measure_motion_energy(path, duration)
    audio_energy = measure_audio_energy(path, duration)

    # Resample GPMF arrays to per-second.
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


# -------------------------------------------------------------------------
# Music analysis + scoring
# -------------------------------------------------------------------------

def analyze_song(song_path: Path) -> dict[str, Any]:
    """
    Use librosa to get tempo, beat times, and estimated downbeats.
    Returns: {duration, tempo, beats, downbeats, intro_end}
    """
    import numpy as np
    import librosa

    y, sr = librosa.load(str(song_path), sr=22050, mono=True)
    duration = len(y) / sr

    # Tempo + beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = beat_frames.tolist() if hasattr(beat_frames, "tolist") else list(beat_frames)
    tempo_val = float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])

    # Crude downbeat estimate: every 4th beat.
    downbeats = beats[::4] if len(beats) >= 4 else beats

    # Find first "energetic" moment (intro_end) by looking at onset strength.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)
    if len(onset_env) > 0:
        threshold = float(np.percentile(onset_env, 70))
        above = np.where(onset_env > threshold)[0]
        intro_end = float(onset_times[above[0]]) if len(above) else 0.0
    else:
        intro_end = 0.0

    return {
        "duration_s": duration,
        "tempo_bpm": tempo_val,
        "beats": beats,
        "downbeats": downbeats,
        "intro_end_s": intro_end,
    }


def build_candidates(catalog: dict[str, Any]) -> list[Candidate]:
    """Turn a catalog of clips into per-second candidate sub-clips with scores."""
    candidates: list[Candidate] = []
    for clip in catalog["clips"]:
        path = clip["path"]
        duration = clip["duration_s"]
        fps = clip.get("fps", 30.0)
        is_short = clip.get("is_short_form", False)
        # For very short clips (Live Photos), the whole clip is one candidate.
        if duration <= 4.0:
            score, reasons = score_window(clip, 0, int(duration))
            candidates.append(Candidate(
                source=path,
                start_s=0.0,
                end_s=duration,
                score=score,
                reasons=reasons,
                src_fps=fps,
                is_short=is_short,
            ))
            continue
        # Walk the clip in overlapping 2-second windows.
        n_sec = int(duration)
        step = 1
        win = 2
        for start in range(0, max(1, n_sec - win + 1), step):
            end = min(n_sec, start + win)
            score, reasons = score_window(clip, start, end)
            candidates.append(Candidate(
                source=path,
                start_s=float(start),
                end_s=float(end),
                score=score,
                reasons=reasons,
                src_fps=fps,
                is_short=False,
            ))
    return candidates


def score_window(clip: dict[str, Any], start: int, end: int) -> tuple[float, list[str]]:
    """
    Composite score for a window into a clip.
    Returns (score, list of reasons explaining the score).
    """
    reasons = []
    score = 0.0

    # Motion energy (universal, works for any footage).
    motion = clip.get("motion_energy", [])
    motion_avg = (
        sum(motion[start:end]) / (end - start)
        if motion[start:end] else 0.0
    )
    if motion_avg > 0:
        score += motion_avg * 1.5
        if motion_avg > (max(motion) * 0.7 if motion else 0):
            reasons.append("motion_peak")

    # Audio energy (cheering, voices, action sounds).
    audio = clip.get("audio_energy", [])
    audio_avg = (
        sum(audio[start:end]) / (end - start)
        if audio[start:end] else 0.0
    )
    score += audio_avg * 1.0
    if audio_avg > 0.7:
        reasons.append("loud_audio")

    # GoPro telemetry: accelerometer spike → jump/impact.
    accl = clip.get("accl_peaks", [])
    if accl[start:end]:
        accl_max = max(accl[start:end])
        # Rough threshold: at rest accl_mag ≈ 9.8 (gravity). 15+ is a real spike.
        if accl_max > 15:
            score += 3.0
            reasons.append("high_accel_jump")
        elif accl_max > 12:
            score += 1.5
            reasons.append("moderate_accel")

    # GoPro telemetry: GPS speed peak.
    speeds = clip.get("gps_speed", [])
    if speeds[start:end]:
        sp_max = max(speeds[start:end])
        sp_overall_max = max(speeds) if speeds else 0
        if sp_overall_max > 0 and sp_max > sp_overall_max * 0.8:
            score += 2.0
            reasons.append("speed_peak")

    # HiLight tag hits (the user said "this is interesting").
    win_ms_start = start * 1000
    win_ms_end = end * 1000
    for tag_ms in clip.get("hilight_tags_ms", []):
        if win_ms_start <= tag_ms <= win_ms_end:
            score += 10.0  # huge weight: user explicitly marked this
            reasons.append("hilight_tag")
            break

    # Penalty: very high gyro variance with no other signal = wobbly junk.
    # (We don't have direct gyro variance here; conservative skip.)

    return score, reasons


def cmd_score(args: argparse.Namespace) -> None:
    catalog = json.loads(Path(args.catalog).expanduser().read_text())
    song = analyze_song(Path(args.song).expanduser().resolve())
    log(f"Song: {song['tempo_bpm']:.0f} BPM, "
        f"{len(song['beats'])} beats, intro ends ~{song['intro_end_s']:.1f}s")

    candidates = build_candidates(catalog)
    log(f"Built {len(candidates)} candidate sub-clips")

    target_len = min(
        song["duration_s"],
        float(args.max_length) if args.max_length else DEFAULT_TARGET_LEN_S,
    )

    # Greedy fill: walk the downbeats (the "structural" cut points), and
    # for each gap pick the highest-scoring unused candidate that fits.
    cut_points = [b for b in song["beats"] if b >= song["intro_end_s"]]
    if not cut_points:
        cut_points = song["beats"]
    cut_points = [t for t in cut_points if t < target_len]
    cut_points.append(target_len)

    # Sort candidates by score descending.
    candidates.sort(key=lambda c: c.score, reverse=True)
    used_sources: dict[str, int] = {}  # source -> times-used (cap to avoid repetition)
    plan_entries = []

    for i in range(len(cut_points) - 1):
        beat_t = cut_points[i]
        next_t = cut_points[i + 1]
        gap = next_t - beat_t
        if gap < MIN_CLIP_S:
            continue
        # Pick best candidate that fits this gap and hasn't been overused.
        pick = None
        for c in candidates:
            clip_len = c.end_s - c.start_s
            if clip_len < MIN_CLIP_S:
                continue
            if used_sources.get(c.source, 0) >= 3:
                continue  # cap repeats per source
            # If gap < candidate, that's fine — we'll trim to gap.
            pick = c
            break
        if not pick:
            continue
        candidates.remove(pick)
        used_sources[pick.source] = used_sources.get(pick.source, 0) + 1

        # Decide if this clip gets slow-mo: high-fps source + on a downbeat
        # + has a motion/jump reason → ramp to 0.5x.
        on_downbeat = any(abs(beat_t - db) < 0.05 for db in song["downbeats"])
        is_high_fps = pick.src_fps >= 90
        wants_slowmo = (
            is_high_fps
            and on_downbeat
            and any(r in ("high_accel_jump", "motion_peak", "hilight_tag") for r in pick.reasons)
            and not args.no_speed_ramp
        )
        speed = 0.5 if wants_slowmo else 1.0
        # How much source time does `gap` seconds of output need?
        src_time_needed = gap * speed
        actual_end = min(pick.end_s, pick.start_s + src_time_needed)

        plan_entries.append({
            "source": pick.source,
            "start_s": pick.start_s,
            "end_s": actual_end,
            "out_duration_s": gap,
            "speed": speed,
            "beat_time_s": beat_t,
            "score": pick.score,
            "reasons": pick.reasons,
        })

    plan = {
        "song": str(Path(args.song).expanduser().resolve()),
        "song_meta": song,
        "target_length_s": target_len,
        "aspect": args.aspect,
        "resolution": args.res,
        "fps": args.fps,
        "lut": args.lut,
        "music_db": args.music_db,
        "clip_db": args.clip_db,
        "entries": plan_entries,
    }
    out = Path(args.out).expanduser().resolve()
    out.write_text(json.dumps(plan, indent=2))
    log(f"Plan: {len(plan_entries)} cuts over {target_len:.1f}s → {out}")


# -------------------------------------------------------------------------
# Render
# -------------------------------------------------------------------------

def resolve_lut(lut_arg: str | None, skill_dir: Path) -> Path | None:
    """Resolve LUT path. Accepts: absolute path, theme name, or None."""
    if not lut_arg:
        return skill_dir / "assets" / "luts" / "cinematic.cube"
    p = Path(lut_arg).expanduser()
    if p.is_file():
        return p
    # Try as a theme name
    themed = skill_dir / "assets" / "luts" / f"{lut_arg}.cube"
    if themed.is_file():
        return themed
    return None


def aspect_filter(aspect: str, target_res: str) -> str:
    """Build the ffmpeg filter to fit/crop to target aspect and resolution."""
    w, h = (int(x) for x in target_res.split("x"))
    if aspect == "9:16":
        w, h = min(w, h), max(w, h)
        if w == h:
            w, h = 1080, 1920
    elif aspect == "1:1":
        w = h = min(w, h)
    # scale-then-crop fills the frame without letterboxing.
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


def cmd_render(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).expanduser().read_text())
    skill_dir = Path(__file__).resolve().parent.parent
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lut = resolve_lut(plan.get("lut"), skill_dir)
    target_res = plan.get("resolution", DEFAULT_RES)
    target_fps = plan.get("fps", DEFAULT_FPS)
    aspect = plan.get("aspect", "16:9")

    log(f"Rendering {len(plan['entries'])} cuts → {output.name}")
    log(f"  res={target_res} fps={target_fps} aspect={aspect}"
        f"{' lut=' + lut.name if lut else ''}")

    # Build a temp dir for the per-clip pre-renders.
    with tempfile.TemporaryDirectory(prefix="aftermovie_") as tmpdir:
        tmp = Path(tmpdir)
        concat_lines = []
        for i, entry in enumerate(plan["entries"]):
            src = Path(entry["source"])
            out_clip = tmp / f"clip_{i:04d}.mp4"
            duration = entry["end_s"] - entry["start_s"]
            speed = entry.get("speed", 1.0)
            out_duration = entry.get("out_duration_s", duration / speed)
            # Build filter chain: trim, scale/crop, optional speed change, LUT.
            vfilter = [aspect_filter(aspect, target_res)]
            if speed != 1.0:
                # setpts scales playback duration; PTS*N slows by factor N.
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
                "-an",  # we strip clip audio here; mixed back in at concat step
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

        # Concat video silently, then mix in the song.
        video_only = tmp / "video_only.mp4"
        run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(video_only),
        ])

        song = plan["song"]
        music_db = plan.get("music_db", -8)
        # Final mux: video + song trimmed to video duration.
        # We don't bring back original clip audio in this MVP — keeping it
        # simple. The recipe notes how to add it back if desired.
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

    log(f"✓ Done → {output}")


# -------------------------------------------------------------------------
# Auto (the one-shot)
# -------------------------------------------------------------------------

def cmd_auto(args: argparse.Namespace) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="aftermovie_auto_"))
    log(f"Working dir: {workdir}")
    catalog_path = workdir / "catalog.json"
    plan_path = workdir / "plan.json"

    # Stage 1: analyze
    a = argparse.Namespace(clips=args.clips, out=str(catalog_path))
    cmd_analyze(a)

    # Stage 2: score
    s = argparse.Namespace(
        catalog=str(catalog_path),
        song=args.song,
        out=str(plan_path),
        max_length=args.max_length,
        aspect=args.aspect,
        res=args.res,
        fps=args.fps,
        lut=args.lut,
        music_db=args.music_db,
        clip_db=args.clip_db,
        no_speed_ramp=args.no_speed_ramp,
    )
    cmd_score(s)

    # Stage 3: render
    r = argparse.Namespace(plan=str(plan_path), output=args.output)
    cmd_render(r)

    # Keep the working dir so the user can inspect catalog.json / plan.json.
    log(f"Intermediate files preserved in: {workdir}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aftermovie",
        description="Beat-synced highlight video generator for macOS.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # analyze
    pa = sub.add_parser("analyze", help="Scan a folder of clips and extract features.")
    pa.add_argument("--clips", required=True, help="Folder containing video clips.")
    pa.add_argument("--out", required=True, help="Output catalog JSON path.")
    pa.set_defaults(func=cmd_analyze)

    # score
    ps = sub.add_parser("score", help="Build an edit plan from a catalog and song.")
    ps.add_argument("--catalog", required=True)
    ps.add_argument("--song", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--max-length", type=float, default=None)
    ps.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "1:1"])
    ps.add_argument("--res", default=DEFAULT_RES)
    ps.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ps.add_argument("--lut", default=None)
    ps.add_argument("--music-db", type=float, default=-8.0)
    ps.add_argument("--clip-db", type=float, default=-18.0)
    ps.add_argument("--no-speed-ramp", action="store_true")
    ps.set_defaults(func=cmd_score)

    # render
    pr = sub.add_parser("render", help="Execute a plan via ffmpeg.")
    pr.add_argument("--plan", required=True)
    pr.add_argument("--output", required=True)
    pr.set_defaults(func=cmd_render)

    # auto
    pu = sub.add_parser("auto", help="One-shot: analyze → score → render.")
    pu.add_argument("--clips", required=True)
    pu.add_argument("--song", required=True)
    pu.add_argument("--output", required=True)
    pu.add_argument("--max-length", type=float, default=None)
    pu.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "1:1"])
    pu.add_argument("--res", default=DEFAULT_RES)
    pu.add_argument("--fps", type=int, default=DEFAULT_FPS)
    pu.add_argument("--lut", default=None)
    pu.add_argument("--music-db", type=float, default=-8.0)
    pu.add_argument("--clip-db", type=float, default=-18.0)
    pu.add_argument("--no-speed-ramp", action="store_true")
    pu.add_argument("--theme", default=None,
                    help="Preset bundle (cinematic, punchy, chill, nostalgic).")
    pu.set_defaults(func=cmd_auto)

    return p


def main() -> None:
    args = build_parser().parse_args()
    # Apply theme presets if given
    if getattr(args, "theme", None):
        themes = {
            "cinematic": {"lut": "cinematic", "music_db": -10, "no_speed_ramp": False},
            "punchy":    {"lut": "punchy",    "music_db": -6,  "no_speed_ramp": False},
            "chill":     {"lut": "chill",     "music_db": -10, "no_speed_ramp": True},
            "nostalgic": {"lut": "nostalgic", "music_db": -10, "no_speed_ramp": False},
        }
        preset = themes.get(args.theme, {})
        for k, v in preset.items():
            if not getattr(args, k, None):
                setattr(args, k, v)
    args.func(args)


if __name__ == "__main__":
    main()
