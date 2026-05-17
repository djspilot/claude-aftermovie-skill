"""Still image → short clip materializer + Live Photo pairing logic.

Turns iPhone-style mixed folders into a clip-only catalog: HEIC/JPG/PNG become
2.5-second mp4 clips with a Quik-style display variant (push / pull / pan /
fit_pad / ...) and are cached under `~/.skills-data/aftermovie/cache/stills/`.
Live Photo pairs (`IMG_0488.HEIC` + `IMG_0488.MOV`) drop the still — the MOV
already carries the motion. Single-file Live Photos with an embedded MOV need
exiftool to extract; otherwise they degrade to the still frame.

HEIC: homebrew ffmpeg lacks libheif, so we pre-decode every image via PIL
(with pillow-heif registered) into a temp PNG and feed THAT to ffmpeg.

Filter-chain construction (variant pick + zoompan/pad/blur graph) lives in
`still_filters.py`; this module is just discovery + IO + the cache.
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

from aftermovie.analyze.still_filters import _build_still_filter, _pick_still_variant
from aftermovie.config import data_dir
from aftermovie.ffmpeg_cmd import log, run

# Register HEIC/AVIF support on PIL as a side-effect import (no-op if missing).
try:
    import pillow_heif  # type: ignore[import-not-found]
    pillow_heif.register_heif_opener()
    _PIL_HEIC_AVAILABLE = True
except ImportError:
    _PIL_HEIC_AVAILABLE = False

STILL_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png",
              ".HEIC", ".HEIF", ".JPG", ".JPEG", ".PNG"}
LIVE_PHOTO_VIDEO_EXTS = {".mov", ".MOV"}
DEFAULT_STILL_DURATION_S = 2.5

# Case-insensitive substring match against filename for prior aftermovie outputs.
OUTPUT_NAME_HINTS = ("aftermovie", "_aftermovie", "highlight_reel", "recap")

# Subdirectories not to recurse into. Staging copies would otherwise double-count
# every clip under two paths, silently doubling the per-source repetition cap.
SKIP_DIR_NAMES = {"_aftermovie_src", "_aftermovie", "aftermovie_workdir",
                  "__pycache__", ".git", "node_modules", ".cache"}


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
    """Split stills into (extracted_live_photo_movs, real_stills, orphan_markers).

    For each HEIC/JPG without a same-stem MOV sibling, probe for an embedded
    Live Photo video and extract it into the cache. Successful extractions are
    returned as MOV paths the analyzer should treat as video. `orphan_markers`
    counts HEICs with an Apple ContentIdentifier but no extractable video —
    surfaced to the user as a hint to re-export.
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
    """Decode any PIL-readable image to a temp PNG (caller unlinks). None on failure."""
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
            log(f"  ! cannot decode {src.name} — install pillow-heif (pip install pillow-heif)")
        else:
            log(f"  ! cannot decode {src.name}: {e}")
        return None


def materialize_still(path: Path, duration_s: float = DEFAULT_STILL_DURATION_S,
                      target_res: str = "1920x1080",
                      force: bool = False) -> Path | None:
    """Render a still to a cached mp4 using a per-file Quik-style variant.

    Pre-decodes via PIL → PNG (so HEIC works via pillow-heif), then feeds it
    to ffmpeg with the variant chain from `still_filters`. Returns None if
    the image can't be decoded.
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
    variant, seed = _pick_still_variant(path, w, h)
    spec = _build_still_filter(variant, seed, n_frames, fps, w, h)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-loop", "1", "-i", str(png_path),
           "-t", f"{duration_s:.3f}", "-vf", spec.chain, "-an",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(out)]
    try:
        run(cmd, check=True)
        # Stamp the materialized clip's mtime with the source still's capture
        # time so downstream code that asks "when was this clip captured?"
        # via the cached mp4 gets a meaningful answer (and chronological
        # sorting Just Works).
        try:
            import os
            from aftermovie.analyze.capture_time import captured_at_for
            ts = captured_at_for(path)
            if ts is not None:
                os.utime(out, (ts, ts))
        except Exception:  # noqa: BLE001
            pass
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
    """Filter out our own previous renders that landed in the source folder."""
    if path.suffix.lower() not in (".mp4", ".mov"):
        return False
    return any(hint in path.name.lower() for hint in OUTPUT_NAME_HINTS)
