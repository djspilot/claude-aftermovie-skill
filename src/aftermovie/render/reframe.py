"""Face-aware reframing for vertical (9:16) output.

When the target aspect is 9:16 and the source has face bbox data, we crop to
a window that tracks the smoothed face centroid. Without face data we fall
back to a centered crop.
"""
from __future__ import annotations

from typing import Iterable


def _smooth(values: list[float], alpha: float = 0.4) -> list[float]:
    """Simple exponential smoothing."""
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def crop_x_expr_for_entry(entry: dict, source_w: int, source_h: int,
                          target_w: int, target_h: int) -> str | None:
    """Build an ffmpeg `crop` filter expression for the entry's source slice.

    Returns a string like `crop=W:H:x_expr:y_expr` or None to mean "default
    centered crop". The x_expr is a per-frame eval that lerps toward the
    face centroid for the second it falls into.
    """
    faces: list[dict | None] = entry.get("face_bboxes") or []
    if not faces or all(f is None for f in faces):
        return None

    # Centers per second, smoothed; fall back to last known when None.
    centers: list[float] = []
    last = 0.5
    for f in faces:
        if f and f.get("cx") is not None:
            last = float(f["cx"])
        centers.append(last)
    centers = _smooth(centers, alpha=0.4)

    # Width of the crop window in source pixels (we want target_w/target_h aspect).
    target_aspect = target_w / target_h
    crop_w = int(round(source_h * target_aspect))
    crop_w = max(2, min(source_w, crop_w))
    half = crop_w / 2.0

    # Build an ffmpeg expression: for each integer second `floor(t)`, use that
    # second's smoothed center clamped to keep the crop window inside the frame.
    # ffmpeg's expression language supports `if`, `floor`, `mod`, arithmetic.
    # We assemble a chain of nested if() calls picking the second's target.
    targets = [max(half, min(source_w - half, c * source_w)) for c in centers]
    # x = target - half, clamped to [0, source_w - crop_w]
    xs = [max(0, min(source_w - crop_w, t - half)) for t in targets]

    if len(xs) == 1:
        x_expr = f"{xs[0]:.1f}"
    else:
        # Build chained if() expression keyed on floor(t).
        expr = f"{xs[-1]:.1f}"
        for i in range(len(xs) - 2, -1, -1):
            expr = f"if(lt(floor(t),{i+1}),{xs[i]:.1f},{expr})"
        x_expr = expr

    return f"crop={crop_w}:{source_h}:'{x_expr}':0"
