"""Still image → short clip materializer + Live Photo pairing logic.

This module turns iPhone-style mixed folders into a clip-only catalog the rest
of the pipeline can use:

  * `.heic`/`.heif`/`.jpg`/`.jpeg`/`.png` files become 2.5-second mp4 clips
    with a subtle Ken Burns zoom, cached under `~/.skills-data/aftermovie/cache/stills/`.
  * Live Photos exported as paired files (e.g. `IMG_0488.HEIC` + `IMG_0488.MOV`)
    are detected by shared stem; the still is dropped because the MOV already
    carries the motion.
  * Single-file Live Photos (HEIC with the MOV in a metadata box) currently
    degrade to the still frame — extracting the embedded MOV needs exiftool.

HEIC support: homebrew's stock ffmpeg can't demux HEIC as a loopable image
(no libheif), so we always pre-decode images via PIL (with pillow-heif
registered) into a temp PNG, then feed THAT to ffmpeg.
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

from aftermovie.config import data_dir
from aftermovie.ffmpeg_cmd import log, run

# Register HEIC/AVIF support on PIL as a side-effect import (no-op if missing).
try:
    import pillow_heif  # type: ignore[import-not-found]
    pillow_heif.register_heif_opener()
    _PIL_HEIC_AVAILABLE = True
except ImportError:
    _PIL_HEIC_AVAILABLE = False

STILL_EXTS = {
    ".heic", ".heif", ".jpg", ".jpeg", ".png",
    ".HEIC", ".HEIF", ".JPG", ".JPEG", ".PNG",
}
LIVE_PHOTO_VIDEO_EXTS = {".mov", ".MOV"}
DEFAULT_STILL_DURATION_S = 2.5

# Filenames at the source-folder root that look like prior aftermovie outputs.
# Matched as case-insensitive substring against the file name.
OUTPUT_NAME_HINTS = ("aftermovie", "_aftermovie", "highlight_reel", "recap")

# Subdirectories that should not be recursed into. These are workaround /
# staging folders the user (or a prior session) created for source
# conversions; if we ingest both the original AND the staging copy we get
# every clip listed under two different paths and the per-source repetition
# cap is silently doubled.
SKIP_DIR_NAMES = {
    "_aftermovie_src", "_aftermovie", "aftermovie_workdir",
    "__pycache__", ".git", "node_modules", ".cache",
}


def _stills_cache_dir() -> Path:
    d = data_dir() / "cache" / "stills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(path: Path, duration_s: float, target_res: str) -> str:
    try:
        st = path.stat()
        seed = f"{path}|{st.st_size}|{int(st.st_mtime)}|{duration_s}|{target_res}"
    except OSError:
        seed = f"{path}|{duration_s}|{target_res}"
    return hashlib.sha1(seed.encode()).hexdigest()[:16]


def _under_skipped_dir(p: Path, root: Path) -> bool:
    """True if any ancestor between p and root is in SKIP_DIR_NAMES."""
    try:
        rel = p.relative_to(root)
    except ValueError:
        return False
    return any(part in SKIP_DIR_NAMES for part in rel.parts[:-1])


def find_live_photos_and_stills(folder: Path) -> tuple[list[Path], list[Path], int]:
    """Split a folder's still files into (extracted_live_photo_movs, stills, n_orphan_live_marker).

    For each HEIC/JPG that has no same-stem MOV sibling, probe for an embedded
    Live Photo / Motion Photo video and extract it into the cache. Successful
    extractions are returned as MOV paths the analyzer should treat as video.
    Failed extractions (or non-Live-Photo files) fall through to the stills list.

    `n_orphan_live_marker` counts HEICs that have an Apple ContentIdentifier
    but no extractable video — i.e. were Live Photos whose motion portion was
    dropped during export. The caller surfaces this as a hint to re-export.
    """
    from aftermovie.analyze.live_photo import (
        exiftool_available,
        has_live_photo_marker,
        live_photo_video_path,
    )

    stills = find_stills_excluding_live_pairs(folder)
    movs: list[Path] = []
    real_stills: list[Path] = []
    orphan_markers = 0
    et_ok = exiftool_available()

    for s in stills:
        mov = live_photo_video_path(s) if et_ok else None
        if mov is not None:
            movs.append(mov)
            continue
        if et_ok and s.suffix.lower() in (".heic", ".heif") and has_live_photo_marker(s):
            orphan_markers += 1
        real_stills.append(s)
    return movs, real_stills, orphan_markers


def find_stills_excluding_live_pairs(folder: Path) -> list[Path]:
    """Return the still files that have NO same-stem video sibling.

    A `IMG_0488.HEIC` next to `IMG_0488.MOV` is treated as a Live Photo and the
    HEIC is dropped — the MOV is what the catalog should hold.
    """
    if not folder.is_dir():
        return []
    by_stem: dict[str, list[Path]] = {}
    for p in folder.rglob("*"):
        if (p.is_file()
                and not p.name.startswith(".")
                and not _under_skipped_dir(p, folder)):
            by_stem.setdefault(p.stem, []).append(p)
    out: list[Path] = []
    for stem, files in by_stem.items():
        stills = [f for f in files if f.suffix in STILL_EXTS]
        videos = [f for f in files if f.suffix in LIVE_PHOTO_VIDEO_EXTS]
        if videos:
            continue  # Live Photo pair — skip stills
        out.extend(stills)
    return sorted(out)


def _decode_to_png(src: Path) -> Path | None:
    """Decode any PIL-readable image (HEIC/JPG/PNG/etc.) to a temp PNG.

    Returns None if PIL can't open it (most commonly: HEIC without
    pillow-heif). The caller is responsible for unlinking the returned path.
    """
    from PIL import Image

    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            img.save(tmp.name, format="PNG")
            return Path(tmp.name)
    except (OSError, ValueError) as e:
        if src.suffix.lower() in (".heic", ".heif") and not _PIL_HEIC_AVAILABLE:
            log(f"  ! cannot decode {src.name} — install pillow-heif "
                f"(pip install pillow-heif)")
        else:
            log(f"  ! cannot decode {src.name}: {e}")
        return None


def materialize_still(path: Path, duration_s: float = DEFAULT_STILL_DURATION_S,
                      target_res: str = "1920x1080",
                      force: bool = False) -> Path | None:
    """Render a still to a cached mp4 with a subtle Ken Burns zoom.

    Returns the cached path, or None if the image can't be decoded.

    Strategy: always pre-decode via PIL → temp PNG, then ffmpeg from the PNG.
    This works for any PIL-readable format (HEIC via pillow-heif), and dodges
    homebrew ffmpeg's lack of libheif.
    """
    cache_dir = _stills_cache_dir()
    key = _cache_key(path, duration_s, target_res)
    out = cache_dir / f"{key}.mp4"
    if out.is_file() and not force:
        return out

    png_path = _decode_to_png(path)
    if png_path is None:
        return None

    w, h = (int(x) for x in target_res.split("x"))
    fps = 30
    n_frames = max(2, int(round(duration_s * fps)))
    vf = (
        f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
        f"crop={w*2}:{h*2},"
        f"zoompan=z='1+0.1*on/{n_frames}':d={n_frames}:fps={fps}:s={w}x{h},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-i", str(png_path),
        "-t", f"{duration_s:.3f}",
        "-vf", vf,
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    try:
        run(cmd, check=True)
        return out
    except subprocess.CalledProcessError:
        log(f"  ! ffmpeg failed on materialized PNG for {path.name}")
        if out.is_file():
            try:
                out.unlink()
            except OSError:
                pass
        return None
    finally:
        try:
            png_path.unlink()
        except OSError:
            pass


def _is_excluded_output(path: Path) -> bool:
    """Heuristic: filter out our own previous renders that landed in the source folder."""
    name = path.name.lower()
    if path.suffix.lower() not in (".mp4", ".mov"):
        return False
    return any(hint in name for hint in OUTPUT_NAME_HINTS)
