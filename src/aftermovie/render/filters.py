"""ffmpeg filter string builders (aspect, lut, speed)."""
from __future__ import annotations


def aspect_filter(aspect: str, target_res: str) -> str:
    """Build the ffmpeg filter to fit/crop to target aspect and resolution."""
    w, h = (int(x) for x in target_res.split("x"))
    if aspect == "9:16":
        w, h = min(w, h), max(w, h)
        if w == h:
            w, h = 1080, 1920
    elif aspect == "1:1":
        w = h = min(w, h)
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
