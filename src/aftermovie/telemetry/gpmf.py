"""GPMF telemetry — lightweight reader for ACCL, GYRO, GPS5."""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path

from aftermovie.ffmpeg_cmd import ffprobe_json, run


def extract_gpmf_track(path: Path) -> bytes | None:
    """
    Pull the GPMF telemetry track out of a GoPro MP4 using ffmpeg.
    Returns the raw GPMF bytes, or None if the file has no telemetry track.
    """
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
    """
    result: dict[str, list[float]] = {"accl_mag": [], "gyro_mag": [], "gps_speed": []}
    if not blob:
        return result
    pos = 0
    n = len(blob)
    scale_stack: list[list[float]] = [[1.0]]
    while pos + 8 <= n:
        try:
            key = blob[pos : pos + 4].decode("ascii", errors="replace")
            type_char = chr(blob[pos + 4])
            struct_size = blob[pos + 5]
            repeat = struct.unpack(">H", blob[pos + 6 : pos + 8])[0]
        except (UnicodeDecodeError, struct.error):
            break
        payload_size = struct_size * repeat
        aligned = (payload_size + 3) & ~3
        payload = blob[pos + 8 : pos + 8 + payload_size]
        pos += 8 + aligned

        if type_char == "\x00":
            continue
        if key == "SCAL" and type_char in ("s", "S", "l", "L"):
            try:
                fmt = ">" + ("h" if type_char == "s" else "H" if type_char == "S"
                             else "i" if type_char == "l" else "I") * repeat
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
            for i in range(0, len(samples) - 2, 3):
                x, y, z = (samples[i] / scale, samples[i + 1] / scale, samples[i + 2] / scale)
                result["accl_mag"].append((x * x + y * y + z * z) ** 0.5)
        elif key == "GYRO" and type_char == "s":
            scale = scale_stack[-1][0] if scale_stack[-1] else 1.0
            try:
                samples = struct.unpack(f">{repeat * (struct_size // 2)}h", payload)
            except struct.error:
                continue
            for i in range(0, len(samples) - 2, 3):
                x, y, z = (samples[i] / scale, samples[i + 1] / scale, samples[i + 2] / scale)
                result["gyro_mag"].append((x * x + y * y + z * z) ** 0.5)
        elif key == "GPS5" and type_char == "l":
            try:
                ints = struct.unpack(f">{repeat * 5}i", payload)
            except struct.error:
                continue
            scales = scale_stack[-1] if len(scale_stack[-1]) >= 5 else [1.0] * 5
            for i in range(0, len(ints) - 4, 5):
                speed_2d = ints[i + 3] / (scales[3] if scales[3] else 1.0)
                result["gps_speed"].append(speed_2d)
    return result
