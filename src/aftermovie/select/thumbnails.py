"""Cached 256x256 JPG thumbnail generator for the `select` web GUI.

Stills (HEIC/JPG/PNG/HEIF/JPEG) are decoded via PIL + pillow-heif and
center-cropped via `ImageOps.fit`. Videos (MP4/MOV/M4V/INSV/LRV) get a
single-frame grab from ffmpeg at t=1s (falls back to t=0 if the clip is
shorter than 1 second).

The cache lives at `~/.skills-data/aftermovie/cache/thumbs/` keyed by
`sha1(path | size | mtime)` so re-exports invalidate but unchanged files
hit the cache on subsequent requests.

Concurrency: the server can call `thumb_path_for(src)` from multiple
threads (one per HTTP request); generation is idempotent — the worst case
is two threads producing the same JPG to the same cache file, which the
last writer wins. We use a tiny global ThreadPoolExecutor for caller-side
warmup if/when needed; the per-request path is synchronous.
"""
from __future__ import annotations

import hashlib
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aftermovie.config import data_dir
from aftermovie.ffmpeg_cmd import log

# Decoder lock for PIL operations on the same source file — pillow-heif's C
# decoders aren't always thread-safe for the same handle, so we serialize per
# absolute path. A global lock would over-serialize on big folders.
_PER_PATH_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# Bounded pool used by callers that want to pre-warm thumbs in parallel
# (e.g. the /api/sources handler). Per-thumb work is bounded by I/O + ffmpeg,
# so 4 is a reasonable default for laptop-class machines.
_POOL: ThreadPoolExecutor | None = None
_POOL_GUARD = threading.Lock()

THUMB_SIZE = (256, 256)
FFMPEG_TIMEOUT_S = 10

STILL_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".insv", ".lrv"}


def _thumbs_cache_dir() -> Path:
    d = data_dir() / "cache" / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(path: Path) -> str:
    try:
        st = path.stat()
        seed = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        seed = str(path)
    return hashlib.sha1(seed.encode()).hexdigest()


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _PER_PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PER_PATH_LOCKS[key] = lock
        return lock


def _ensure_pillow_heif() -> bool:
    try:
        import pillow_heif  # type: ignore[import-not-found]
        pillow_heif.register_heif_opener()
        return True
    except ImportError:
        return False


def _make_still_thumb(src: Path, out: Path) -> bool:
    """PIL-based path: works for HEIC/JPG/PNG (HEIC needs pillow-heif)."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        log("  ! Pillow not available; cannot make still thumbnail")
        return False

    if src.suffix.lower() in (".heic", ".heif"):
        _ensure_pillow_heif()

    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            thumb = ImageOps.fit(img, THUMB_SIZE, Image.Resampling.LANCZOS)
            thumb.save(out, format="JPEG", quality=85, optimize=True)
        return True
    except (OSError, ValueError) as e:
        log(f"  ! cannot decode {src.name}: {e}")
        return False


def _make_video_thumb(src: Path, out: Path) -> bool:
    """ffmpeg single-frame grab at t=1 (falls back to t=0 on short clips)."""
    # Use scale + pad so vertical iPhone clips fit the 256x256 square nicely
    # without distortion. force_original_aspect_ratio=decrease + pad to box.
    vf = (
        "scale=256:256:force_original_aspect_ratio=decrease,"
        "pad=256:256:(ow-iw)/2:(oh-ih)/2:color=black,"
        "format=yuvj420p"
    )
    for ss in ("1", "0"):
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", ss, "-i", str(src),
            "-frames:v", "1",
            "-vf", vf,
            "-q:v", "3",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           timeout=FFMPEG_TIMEOUT_S)
            if out.is_file() and out.stat().st_size > 0:
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    log(f"  ! ffmpeg could not extract a thumbnail frame from {src.name}")
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass
    return False


def thumb_path_for(source: Path) -> Path | None:
    """Generate (or hit cache) and return the JPG path for `source`.

    Returns None if the source is missing or thumbnail generation failed.
    The caller is responsible for surfacing the failure (e.g. serving a 404
    or a placeholder).
    """
    src = Path(source)
    try:
        if not src.is_file():
            return None
    except OSError:
        return None

    cache_dir = _thumbs_cache_dir()
    key = _cache_key(src)
    out = cache_dir / f"{key}.jpg"
    if out.is_file() and out.stat().st_size > 0:
        return out

    ext = src.suffix.lower()
    lock = _path_lock(src)
    with lock:
        # Re-check after acquiring the lock — another thread may have just
        # generated the same thumbnail.
        if out.is_file() and out.stat().st_size > 0:
            return out
        ok = False
        if ext in STILL_EXTS:
            ok = _make_still_thumb(src, out)
        elif ext in VIDEO_EXTS:
            ok = _make_video_thumb(src, out)
        else:
            log(f"  ! unsupported file type for thumb: {src.name}")
            return None
    return out if ok else None


def get_pool() -> ThreadPoolExecutor:
    """Lazy global thumbnail-generation pool (4 workers)."""
    global _POOL
    with _POOL_GUARD:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="thumbgen")
        return _POOL


def shutdown_pool() -> None:
    """Tests use this to drain the pool between runs."""
    global _POOL
    with _POOL_GUARD:
        if _POOL is not None:
            _POOL.shutdown(wait=True)
            _POOL = None
