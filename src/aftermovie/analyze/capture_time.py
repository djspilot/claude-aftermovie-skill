"""Extract a capture-time timestamp for sorting clips chronologically.

For stills: EXIF `DateTimeOriginal` via PIL/pillow-heif.
For videos: ffprobe `format.tags.creation_time`.
For materialized stills (mp4 in our cache): we can't read it from the
output — caller should pass the *source* still path instead.

Falls back to file mtime when no embedded timestamp is found.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _from_ffprobe(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format_tags=creation_time",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        ct = (data.get("format", {}) or {}).get("tags", {}).get("creation_time")
        if not ct:
            return None
        # ISO 8601: '2025-05-13T17:14:33.000000Z'
        # Python <3.11 doesn't handle the trailing Z natively.
        if ct.endswith("Z"):
            ct = ct[:-1] + "+00:00"
        return datetime.fromisoformat(ct).timestamp()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, ValueError, KeyError):
        return None


def _from_pil_exif(path: Path) -> float | None:
    try:
        from PIL import Image
        try:
            import pillow_heif  # noqa: F401 — registers HEIC support
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(path) as img:
            exif = img.getexif() if hasattr(img, "getexif") else None
            if not exif:
                return None
            # Tag 36867 = DateTimeOriginal; 36868 = DateTimeDigitized; 306 = DateTime.
            for tag in (36867, 36868, 306):
                raw = exif.get(tag)
                if not raw:
                    continue
                # EXIF format: 'YYYY:MM:DD HH:MM:SS', no timezone — assume local.
                try:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").timestamp()
                except ValueError:
                    continue
    except (OSError, ValueError, ImportError):
        return None
    return None


def captured_at_for(path: Path) -> float | None:
    """Return a Unix timestamp for `path`'s capture moment, or None.

    Tries EXIF first for image extensions, ffprobe first for video extensions,
    then the other strategy as a fallback. Falls back to mtime if both fail —
    caller can decide whether that's good enough.
    """
    ext = path.suffix.lower()
    is_image = ext in (".heic", ".heif", ".jpg", ".jpeg", ".png")
    is_video = ext in (".mp4", ".mov", ".m4v", ".insv", ".lrv")

    if is_image:
        t = _from_pil_exif(path)
        if t is not None:
            return t
    if is_video:
        t = _from_ffprobe(path)
        if t is not None:
            return t
    # Cross-try the other strategy in case the file lies about its extension.
    t = _from_pil_exif(path) or _from_ffprobe(path)
    if t is not None:
        return t
    try:
        return path.stat().st_mtime
    except OSError:
        return None
