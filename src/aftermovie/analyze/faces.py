"""Per-second face detection via MediaPipe Tasks FaceDetector.

We sample 1 frame/sec via ffmpeg image2pipe → MJPEG → PIL → numpy, then run
FaceDetector and store the top face's bbox for each second of the clip.

The whole feature is gated: if mediapipe is missing or fails to load the
model, faces is detected as unavailable() and analyze_clip skips it. The
reframe pipeline then falls back to a centered crop.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

from aftermovie.config import models_dir

_MODEL_NAME = "blaze_face_short_range.tflite"


def available() -> bool:
    """Cheap check for `mediapipe + model file`."""
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return (models_dir() / _MODEL_NAME).is_file()


def _make_detector():
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    model_path = models_dir() / _MODEL_NAME
    opts = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(opts)


def detect_per_second(path: Path, duration_s: float) -> list[dict[str, Any] | None]:
    """Return a list of length max(1, int(duration_s)) with the top face bbox
    per sampled second (or None if no face was found that second).

    Each bbox is a dict {x, y, w, h, score, cx, cy} where the coordinates
    are normalized to the source resolution.
    """
    if not available():
        return [None] * max(1, int(duration_s))

    import mediapipe as mp
    import numpy as np
    from PIL import Image

    detector = _make_detector()
    n_sec = max(1, int(duration_s))

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(path),
        "-vf", "fps=1",
        "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return [None] * n_sec

    # Split the MJPEG stream into individual JPEG frames.
    raw = proc.stdout
    soi = b"\xff\xd8"
    eoi = b"\xff\xd9"
    frames: list[bytes] = []
    i = 0
    while True:
        s = raw.find(soi, i)
        if s < 0:
            break
        e = raw.find(eoi, s)
        if e < 0:
            break
        frames.append(raw[s : e + 2])
        i = e + 2

    results: list[dict[str, Any] | None] = []
    for f in frames[:n_sec]:
        try:
            img = Image.open(io.BytesIO(f)).convert("RGB")
            arr = np.array(img)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            det = detector.detect(mp_image)
        except Exception:
            results.append(None)
            continue
        if not det.detections:
            results.append(None)
            continue
        best = max(det.detections, key=lambda d: d.categories[0].score if d.categories else 0)
        bbox = best.bounding_box
        w_img, h_img = img.size
        cx = (bbox.origin_x + bbox.width / 2) / w_img
        cy = (bbox.origin_y + bbox.height / 2) / h_img
        results.append({
            "x": bbox.origin_x / w_img,
            "y": bbox.origin_y / h_img,
            "w": bbox.width / w_img,
            "h": bbox.height / h_img,
            "score": float(best.categories[0].score) if best.categories else 0.0,
            "cx": cx, "cy": cy,
        })

    # Pad if we got fewer frames than expected seconds.
    while len(results) < n_sec:
        results.append(None)
    return results
