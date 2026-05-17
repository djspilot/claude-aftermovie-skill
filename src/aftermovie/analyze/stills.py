"""Still image → short clip materializer + Live Photo pairing logic.

This module turns iPhone-style mixed folders into a clip-only catalog the rest
of the pipeline can use:

  * `.heic`/`.heif`/`.jpg`/`.jpeg`/`.png` files become 2.5-second mp4 clips
    with a subtle Ken Burns zoom, cached under `~/.skills-data/aftermovie/cache/stills/`.
  * Live Photos exported as paired files (e.g. `IMG_0488.HEIC` + `IMG_0488.MOV`)
    are detected by shared stem; the still is dropped because the MOV already
    carries the motion.
  * Live Photos exported as a single file with the video track *embedded* in
    the HEIC: ffmpeg can read the first frame but the embedded MOV isn't
    extractable without exiftool; for now those degrade to stills.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from aftermovie.config import data_dir
from aftermovie.ffmpeg_cmd import log, run

STILL_EXTS = {
    ".heic", ".heif", ".jpg", ".jpeg", ".png",
    ".HEIC", ".HEIF", ".JPG", ".JPEG", ".PNG",
}
LIVE_PHOTO_VIDEO_EXTS = {".mov", ".MOV"}
DEFAULT_STILL_DURATION_S = 2.5


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


def find_stills_excluding_live_pairs(folder: Path) -> list[Path]:
    """Return the still files that have NO same-stem video sibling.

    A `IMG_0488.HEIC` next to `IMG_0488.MOV` is treated as a Live Photo and the
    HEIC is dropped — the MOV is what the catalog should hold.
    """
    if not folder.is_dir():
        return []
    by_stem: dict[str, list[Path]] = {}
    for p in folder.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            by_stem.setdefault(p.stem, []).append(p)
    out: list[Path] = []
    for stem, files in by_stem.items():
        stills = [f for f in files if f.suffix in STILL_EXTS]
        videos = [f for f in files if f.suffix in LIVE_PHOTO_VIDEO_EXTS]
        if videos:
            continue  # Live Photo pair — skip stills
        out.extend(stills)
    return sorted(out)


def materialize_still(path: Path, duration_s: float = DEFAULT_STILL_DURATION_S,
                      target_res: str = "1920x1080",
                      force: bool = False) -> Path | None:
    """Render a still to a cached mp4 with a subtle Ken Burns zoom.

    Returns the cached path, or None if ffmpeg can't read the source.
    """
    cache_dir = _stills_cache_dir()
    key = _cache_key(path, duration_s, target_res)
    out = cache_dir / f"{key}.mp4"
    if out.is_file() and not force:
        return out

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
        "-loop", "1", "-i", str(path),
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
        log(f"  ! could not materialize still {path.name}")
        if out.is_file():
            try:
                out.unlink()
            except OSError:
                pass
        return None
