"""Semantic clip embeddings via MediaPipe ImageEmbedder (optional).

dHash catches near-identical FRAMES; it cannot catch "the same scene shot
from a different angle". This module embeds one representative frame per
clip with a MobileNetV3 image embedder — cosine similarity between those
vectors is a cheap semantic same-scene signal the scorer uses as an extra
duplicate-suppression layer on top of the phash one.

Fully gated, same contract as analyze/faces.py: requires the optional
`mediapipe` dependency AND the embedder model file in assets/models. When
either is missing, `available()` is False and analyze skips embeddings —
catalogs simply carry `embedding: null` and the scorer's semantic filter
is a no-op.

Model (≈4MB, not bundled — drop it in assets/models to enable):
https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite
"""
from __future__ import annotations

from pathlib import Path

from aftermovie.analyze.duplicates import _extract_frame_png, _probe_duration
from aftermovie.config import models_dir
from aftermovie.optional_dep import optional_import

_MODEL_NAME = "mobilenet_v3_small.tflite"

_MEDIAPIPE = optional_import(
    "mediapipe",
    warning="  ! mediapipe not installed — semantic duplicate detection "
            "disabled (pip install mediapipe).",
)

_embedder = None  # lazy singleton; embedder init is ~100ms


def available() -> bool:
    """Cheap check for `mediapipe + embedder model file`."""
    if not _MEDIAPIPE.available:
        return False
    return (models_dir() / _MODEL_NAME).is_file()


def _get_embedder():
    global _embedder
    if _embedder is None:
        import mediapipe as mp  # noqa: F401  (asserts the import works)
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        opts = vision.ImageEmbedderOptions(
            base_options=BaseOptions(
                model_asset_path=str(models_dir() / _MODEL_NAME)),
            l2_normalize=True,
        )
        _embedder = vision.ImageEmbedder.create_from_options(opts)
    return _embedder


def _embed_image_file(img_path: Path) -> list[float] | None:
    """Embed a PNG/JPG on disk; None on any failure (never raises)."""
    try:
        import mediapipe as mp
        img = mp.Image.create_from_file(str(img_path))
        result = _get_embedder().embed(img)
        return [float(v) for v in result.embeddings[0].embedding]
    except Exception:
        return None


def embed_for_clip(path: Path, origin_still: Path | None = None) -> list[float] | None:
    """L2-normalized embedding of a representative frame, or None.

    Stills embed the original image (mediapipe reads png/jpg; HEIC and
    other exotics fail gracefully to None). Videos embed their midpoint
    frame via the same ffmpeg extraction the phash uses.
    """
    if not available():
        return None
    if origin_still is not None:
        return _embed_image_file(origin_still)
    dur = _probe_duration(path)
    if dur is None:
        return None
    tmp_png = _extract_frame_png(path, dur / 2.0)
    if tmp_png is None:
        return None
    try:
        return _embed_image_file(tmp_png)
    finally:
        try:
            tmp_png.unlink()
        except OSError:
            pass


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Vectors are l2-normalized at embed time, so this
    is a plain dot product; mismatched lengths count as fully dissimilar."""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
