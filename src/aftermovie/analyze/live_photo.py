"""Single-file Live Photo / Motion Photo video extraction.

iPhone Live Photos exist in two export shapes:
  * **Paired**  — `IMG_xxxx.HEIC` + `IMG_xxxx.MOV` siblings. Already handled in
    analyze.stills (the HEIC is dropped, the MOV is treated as video).
  * **Single-file** — HEIC with the MOV embedded as a metadata box (some
    AirDrop / iCloud paths), or Google Pixel "Motion Photo" JPGs with an MP4
    appended after the JPEG end-of-image marker.

This module probes a single image file and, if it contains an extractable
video, writes that video to a content-hash-keyed cache and returns the path.

Detection tags (Apple Live Photo):
  - `MakerNotes:ContentIdentifier` indicates the file was once a Live Photo.
    *Presence does not guarantee* the MOV is in the file — many exports
    drop the motion side.

Extraction methods tried, in order:
  1. `exiftool -b -EmbeddedVideoFile` (covers Apple HEIC with embedded MOV box)
  2. ffmpeg track extraction (multi-track HEIC with a video track)

Returns None silently if nothing extractable is found, or if exiftool is
missing. The caller is responsible for surfacing a single aggregate hint
to the user.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from aftermovie.analyze.stills import _cache_key
from aftermovie.config import data_dir
from aftermovie.ffmpeg_cmd import log


_EXIFTOOL_WARNED = False


def _live_photo_cache_dir() -> Path:
    d = data_dir() / "cache" / "live_photos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def _exiftool_warn_once() -> None:
    global _EXIFTOOL_WARNED
    if not _EXIFTOOL_WARNED:
        log("  ! exiftool not found — single-file Live Photos will be used as "
            "stills only. Install it (brew install exiftool) to enable motion "
            "extraction.")
        _EXIFTOOL_WARNED = True


def has_live_photo_marker(heic: Path) -> bool:
    """True if `heic` carries an Apple ContentIdentifier (was a Live Photo)."""
    if not exiftool_available():
        return False
    try:
        out = subprocess.run(
            ["exiftool", "-s", "-s", "-s", "-ContentIdentifier", str(heic)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return bool(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _try_exiftool_embedded(heic: Path, out: Path) -> bool:
    """Try `exiftool -b -EmbeddedVideoFile`. Returns True if a non-trivial MOV
    landed at `out`."""
    try:
        with out.open("wb") as fh:
            subprocess.run(
                ["exiftool", "-b", "-EmbeddedVideoFile", str(heic)],
                check=False, stdout=fh, stderr=subprocess.DEVNULL, timeout=30,
            )
    except subprocess.TimeoutExpired:
        return False
    if not out.is_file() or out.stat().st_size < 4096:
        return False
    return _ffprobe_has_motion(out)


def _try_ffmpeg_video_track(heic: Path, out: Path) -> bool:
    """Some single-file HEICs expose a real video track ffmpeg can extract."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(heic)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    # Need at least one track with > 1 frame to call it motion.
    has_motion_track = any(
        n.strip().isdigit() and int(n.strip()) > 1 for n in probe
    )
    if not has_motion_track:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(heic),
             "-map", "0:v:0", "-c", "copy", str(out)],
            check=True, capture_output=True, timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return out.is_file() and out.stat().st_size >= 4096 and _ffprobe_has_motion(out)


def _ffprobe_has_motion(path: Path) -> bool:
    """True if `path` parses as a video with > 1 frame."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    if not out:
        return False
    nb = out[0].strip()
    if nb.isdigit() and int(nb) > 1:
        return True
    # Some containers report 'N/A' frames but a positive duration — accept.
    if len(out) > 1:
        try:
            return float(out[1].strip()) > 0.2
        except ValueError:
            return False
    return False


def live_photo_video_path(heic: Path) -> Path | None:
    """Return a cached MOV extracted from `heic`, or None if nothing usable.

    Cache key includes path, size, mtime so re-exports invalidate. Negative
    results (no extractable video) are NOT cached — re-probing each run is
    cheap and avoids stale "no" answers after the user re-exports.
    """
    if not exiftool_available():
        _exiftool_warn_once()
        return None
    cache_dir = _live_photo_cache_dir()
    key = _cache_key(heic, duration_s=0.0, target_res="motion")
    out = cache_dir / f"{key}.mov"
    if out.is_file() and out.stat().st_size >= 4096 and _ffprobe_has_motion(out):
        return out

    # Wipe any prior failed-extraction stub before retrying.
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass

    if _try_exiftool_embedded(heic, out):
        return out
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass
    if _try_ffmpeg_video_track(heic, out):
        return out
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass
    return None
