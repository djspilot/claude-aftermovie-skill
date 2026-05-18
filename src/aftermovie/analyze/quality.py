"""Per-second sharpness + exposure metrics for video clips.

Both signals run on frames sampled at 1 Hz via OpenCV (cv2). When cv2 is not
importable both functions short-circuit to an empty list and we log a single
warning the first time we're called — the scorer treats missing data as
"neutral" and skips the corresponding penalties.

Sharpness uses the variance of the Laplacian on a grayscale frame: textured /
in-focus frames produce wide gradient distributions, motion-blurred or
defocused frames produce narrow ones. The raw values span several orders of
magnitude across cameras and exposures, so we min-max normalize the result
WITHIN each clip into [0, 1] before handing it to the scorer.

Exposure is the mean luminance of the frame, in [0, 1]. Pitch-black frames
sit at 0; pure-white blown-out frames sit at 1; well-exposed mid-tones land
near 0.5. The scorer's job is to penalize the extremes (`< 0.25` or
`> 0.85`); the analyzer just reports the raw signal.
"""
from __future__ import annotations

from pathlib import Path

from aftermovie.ffmpeg_cmd import log

try:
    import cv2  # type: ignore[import-not-found]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_CV2_WARNED = False


def available() -> bool:
    """True iff cv2 imported. Sibling analyzers use the same pattern."""
    return _CV2_AVAILABLE


def _warn_once() -> None:
    global _CV2_WARNED
    if not _CV2_WARNED:
        log("! cv2 not installed — sharpness/exposure skipped "
            "(pip install opencv-python-headless)")
        _CV2_WARNED = True


def _sampled_grayscale_frames(path: Path, duration: float, fps: float):
    """Yield (sec_idx, grayscale_frame) pairs at ~1 Hz from `path`.

    Picks one frame per integer second by stepping `fps` frames at a time
    rather than re-seeking — VideoCapture seek on H.264 hops to the nearest
    keyframe and would skew sharpness toward the I-frames.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    try:
        n_sec = max(1, int(duration))
        step = max(1, int(round(fps)))
        next_target = 0  # frame index we want next
        cur = 0
        emitted = 0
        while emitted < n_sec:
            ok, frame = cap.read()
            if not ok:
                break
            if cur == next_target:
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                yield emitted, gray
                emitted += 1
                next_target += step
            cur += 1
    finally:
        cap.release()


def sharpness_per_second(path: Path, duration: float, fps: float) -> list[float]:
    """Per-second sharpness in [0, 1], min-max normalized within the clip.

    Returns [] if cv2 is unavailable or the file can't be opened.
    """
    if not _CV2_AVAILABLE:
        _warn_once()
        return []
    raw: list[float] = []
    for _, gray in _sampled_grayscale_frames(path, duration, fps):
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        raw.append(float(lap.var()))
    if not raw:
        return []
    lo = min(raw)
    hi = max(raw)
    if hi - lo < 1e-9:
        # Flat distribution — every second equally (un)sharp. Mid-value keeps
        # the scorer from flagging the whole clip as blurry.
        return [0.5] * len(raw)
    return [(v - lo) / (hi - lo) for v in raw]


def exposure_per_second(path: Path, duration: float, fps: float) -> list[float]:
    """Per-second mean luminance in [0, 1]. 0 = pitch black, 1 = blown white.

    Returns [] if cv2 is unavailable or the file can't be opened.
    """
    if not _CV2_AVAILABLE:
        _warn_once()
        return []
    out: list[float] = []
    for _, gray in _sampled_grayscale_frames(path, duration, fps):
        out.append(float(gray.mean()) / 255.0)
    return out


def sharpness_for_image(path: Path) -> float | None:
    """Single sharpness value for a still image, scaled into [0, 1].

    Stills don't have per-clip context to normalize against, so we map raw
    Laplacian variance through a saturating curve: most photos land between
    50 (soft) and 1500 (tack sharp). Returns None if cv2 is unavailable.
    """
    if not _CV2_AVAILABLE:
        _warn_once()
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    # Saturating map: 0 → 0, 500 → 0.5, ∞ → 1. Avoids per-clip normalization.
    return var / (var + 500.0)


def exposure_for_image(path: Path) -> float | None:
    """Single mean-luminance value for a still image in [0, 1]. None if cv2 missing."""
    if not _CV2_AVAILABLE:
        _warn_once()
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(img.mean()) / 255.0
