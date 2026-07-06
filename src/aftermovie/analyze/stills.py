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
    """Decode any PIL-readable image to a temp PNG (caller unlinks). None on failure.

    Applies EXIF orientation, then a face-detection-based auto-orient
    fallback for cases where EXIF is missing or wrong.
    """
    from PIL import Image, ImageOps
    from aftermovie.analyze.orient import auto_orient

    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img = auto_orient(img, source_path=src)
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


# Two portrait photos shot close together become ONE side-by-side shot on a
# landscape canvas — the Quik-style duo. Only pairs within this window (same
# scene, not months apart in a chat export) are combined.
DUO_PAIR_WINDOW_S = 600.0


def _is_portrait(path: Path) -> bool:
    """True when the image renders taller than wide (EXIF-orientation aware).
    Header-only read; any failure counts as not-portrait."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            # Orientations 5-8 rotate 90° — displayed dims are swapped.
            if (img.getexif() or {}).get(274, 1) in (5, 6, 7, 8):
                w, h = h, w
        return h > w
    except Exception:
        return False


def pair_portrait_stills(stills: list[Path]) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Split stills into (portrait duo pairs, remaining singles).

    Greedy over capture-time order: two DIRECTLY consecutive portrait stills
    captured within DUO_PAIR_WINDOW_S pair up; everything else stays single.
    """
    from aftermovie.analyze.capture_time import captured_at_for

    timed = sorted(stills, key=lambda p: (captured_at_for(p) or float("inf"),
                                          str(p)))
    times = {p: captured_at_for(p) for p in timed}
    pairs: list[tuple[Path, Path]] = []
    singles: list[Path] = []
    i = 0
    while i < len(timed):
        a = timed[i]
        b = timed[i + 1] if i + 1 < len(timed) else None
        if (b is not None and _is_portrait(a) and _is_portrait(b)
                and times[a] is not None and times[b] is not None
                and (times[b] - times[a]) <= DUO_PAIR_WINDOW_S):
            pairs.append((a, b))
            i += 2
        else:
            singles.append(a)
            i += 1
    return pairs, singles


def materialize_still_duo(a: Path, b: Path,
                          duration_s: float = DEFAULT_STILL_DURATION_S,
                          target_res: str = "1920x1080",
                          force: bool = False) -> Path | None:
    """Render two portrait stills side-by-side into one cached mp4.

    Each half scale-fills width/2 × height (center crop), then hstack.
    ponytail: static duo — no Ken Burns; the split itself is the visual.
    Returns None if either image can't be decoded."""
    cache_dir = _stills_cache_dir()
    key = hashlib.sha1(
        f"duo|{_cache_key(a, duration_s, target_res)}"
        f"|{_cache_key(b, duration_s, target_res)}".encode()
    ).hexdigest()[:16]
    out = cache_dir / f"duo_{key}.mp4"
    if out.is_file() and not force:
        return out

    png_a = _decode_to_png(a)
    png_b = _decode_to_png(b)
    if png_a is None or png_b is None:
        return None

    w, h = (int(x) for x in target_res.split("x"))
    half = (w // 2) // 2 * 2  # even width for yuv420p
    fill = (f"scale={half}:{h}:force_original_aspect_ratio=increase,"
            f"crop={half}:{h},setsar=1")
    graph = (f"[0:v]{fill}[l];[1:v]{fill}[r];"
             f"[l][r]hstack=inputs=2,fps=30,format=yuv420p")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-loop", "1", "-i", str(png_a),
           "-loop", "1", "-i", str(png_b),
           "-t", f"{duration_s:.3f}", "-filter_complex", graph, "-an",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           str(out)]
    try:
        run(cmd, check=True)
        try:
            import os
            from aftermovie.analyze.capture_time import captured_at_for
            ts = captured_at_for(a)
            if ts is not None:
                os.utime(out, (ts, ts))
        except Exception:  # noqa: BLE001
            pass
        return out
    except subprocess.CalledProcessError:
        log(f"  ! ffmpeg failed on duo still {a.name} + {b.name}")
        if out.is_file():
            try:
                out.unlink()
            except OSError:
                pass
        return None
    finally:
        for p in (png_a, png_b):
            try:
                p.unlink()
            except OSError:
                pass


def materialize_still(path: Path, duration_s: float = DEFAULT_STILL_DURATION_S,
                      target_res: str = "1920x1080",
                      force: bool = False) -> Path | None:
    """Render a still to a cached mp4 using a per-file Quik-style variant.

    Pre-decodes via PIL → PNG (so HEIC works via pillow-heif), then feeds it
    to ffmpeg with the variant chain from `still_filters`. Returns None if
    the image can't be decoded.
    """
    cache_dir = _stills_cache_dir()
    # Cache key suffix bumped to invalidate when orientation logic changes.
    key = _cache_key(path, duration_s, target_res) + "_o1"
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
