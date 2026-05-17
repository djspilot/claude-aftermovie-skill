"""Auto-orientation fallback: detect the correct rotation for a PIL image
when EXIF doesn't tell us (screenshots, edited photos, missing tags).

Strategy:
1. Caller has already applied `ImageOps.exif_transpose`.
2. If MediaPipe FaceDetector is available, run face detection on the image
   plus 3 rotated copies (90/180/270°). The rotation whose face-confidence
   sum is highest wins — that's "upright" because faces are easier to
   detect when eyes-above-nose-above-mouth orientation holds.
3. Conservative threshold: only override the EXIF result when the
   alternative rotation beats it by at least 30%. Otherwise trust EXIF.

The cost: one downscaled detection per rotation. We downscale to 256 px on
the longest side first so each detection is ~5-10ms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aftermovie.analyze.faces import available as faces_available


_DETECTOR = None
_OVERRIDE_MARGIN = 1.30  # alt-rotation must beat EXIF result by 30%+


def _detector() -> Any:
    """Lazy-init a singleton FaceDetector — model load is ~30ms, not free."""
    global _DETECTOR
    if _DETECTOR is None:
        from aftermovie.analyze.faces import _make_detector
        _DETECTOR = _make_detector()
    return _DETECTOR


def _face_confidence(img: "Any") -> float:
    """Sum of face-detection confidences on `img` (a PIL.Image).
    Higher = more / more-confident faces detected.
    """
    if not faces_available():
        return 0.0
    try:
        import mediapipe as mp
        import numpy as np

        small = img.copy()
        small.thumbnail((256, 256))
        arr = np.array(small.convert("RGB"))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        det = _detector().detect(mp_img)
    except Exception:  # noqa: BLE001
        return 0.0
    if not det.detections:
        return 0.0
    total = 0.0
    for d in det.detections:
        if d.categories:
            total += float(d.categories[0].score)
    return total


def auto_orient(img: "Any", source_path: Path | None = None) -> "Any":
    """Return `img` rotated to its canonical orientation.

    Caller should have already run `ImageOps.exif_transpose`. This helper
    only fires when face detection is available and finds a STRONGER face
    response at a different rotation than what EXIF produced.
    """
    if not faces_available():
        return img
    base_conf = _face_confidence(img)
    best_angle = 0
    best_conf = base_conf
    for angle in (90, 180, 270):
        rotated = img.rotate(-angle, expand=True)
        conf = _face_confidence(rotated)
        if conf > best_conf * _OVERRIDE_MARGIN:
            best_conf = conf
            best_angle = angle
    if best_angle == 0:
        return img
    # Silent — the analyzer already logs lots; an extra line per still
    # would be noisy. The dimension shift is visible in the output.
    return img.rotate(-best_angle, expand=True)
