"""Visual duplicate detection via 8x8 dHash + union-find grouping.

Per-clip perceptual hash so the scorer can collapse near-identical shots
(e.g. ten near-identical photos of the same sunset, or two GoPros pointed
at the same trick) BEYOND the existing timestamp-burst suppression. The
two filters compose: burst-dedup catches same-moment-from-same-camera,
duplicate-grouping catches same-look-from-anywhere-in-the-folder.

Hash algorithm: dHash (Zauner 2010). Load → grayscale → resize to 9x8 →
compare each pixel to its right neighbour → 64 bits → 16-char hex. dHash
is cheap, robust to mild crop / exposure shifts, and gives us a useful
Hamming distance metric.

Frame source: for video files we grab a single frame from the middle of
the clip via ffmpeg into a temp PNG, then hash that. For still images
(materialized stills are mp4s by the time they reach the catalog, so this
mostly matters for direct callers / tests) we load straight via PIL.

Dependency posture: PIL is already a required dep; ffmpeg is the project's
universal hard requirement. If either is somehow missing at call time we
log ONCE and return None — the catalog still writes, downstream phash
consumers treat None as "singleton, leave alone".
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from aftermovie.ffmpeg_cmd import log

# One-shot warning flags so a missing dep doesn't spam the log for every
# clip in the folder. First clip logs, the rest are silent.
_WARNED_NO_PIL = False
_WARNED_NO_FFMPEG = False

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
                  ".insv", ".lrv"}
STILL_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp",
                  ".bmp", ".tiff", ".tif"}


def _extract_middle_frame_png(path: Path) -> Path | None:
    """Grab a single PNG frame from the middle of a video into a temp file.

    Caller is responsible for unlinking the returned path. Returns None if
    ffprobe or ffmpeg fails (corrupt file, no ffmpeg on PATH, etc.)."""
    global _WARNED_NO_FFMPEG
    # Probe duration so we can seek to the midpoint. A failed probe usually
    # means a corrupt clip — bail rather than guess.
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True, timeout=10,
        )
        dur = float(probe.stdout.strip() or "0")
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, ValueError):
        if not _WARNED_NO_FFMPEG:
            log("phash: ffprobe unavailable or failed — perceptual hashing disabled")
            _WARNED_NO_FFMPEG = True
        return None
    mid = max(0.0, dur / 2.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    out = Path(tmp.name)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{mid:.3f}", "-i", str(path),
        "-frames:v", "1",
        "-vf", "scale=64:64:force_original_aspect_ratio=decrease",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        try:
            out.unlink()
        except OSError:
            pass
        return None
    if not out.is_file() or out.stat().st_size == 0:
        try:
            out.unlink()
        except OSError:
            pass
        return None
    return out


def _dhash_from_pil_image(img) -> str:
    """Compute 64-bit dHash → 16-char zero-padded hex.

    Compares each pixel in the 9x8 grayscale image to its right neighbour;
    bit = 1 if left > right. Reading order is row-major (top-left first).
    """
    from PIL import Image  # local import: kept off the module top so a
                            # broken PIL install can't break `import aftermovie.analyze.duplicates`.

    small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = small.load()
    bits = 0
    for y in range(8):
        for x in range(8):
            bit = 1 if px[x, y] > px[x + 1, y] else 0
            bits = (bits << 1) | bit
    return f"{bits:016x}"


def compute_phash(path: Path) -> str | None:
    """Return the 16-hex-char dHash of `path`, or None if hashing failed.

    Videos are sampled at their midpoint frame via ffmpeg. Stills load
    straight through PIL (with pillow-heif's HEIC opener if registered).
    Any failure mode (missing PIL, missing ffmpeg, unreadable file, weird
    codec) returns None — never raises — so callers can keep going.
    """
    global _WARNED_NO_PIL
    try:
        from PIL import Image
    except ImportError:
        if not _WARNED_NO_PIL:
            log("phash: Pillow unavailable — perceptual hashing disabled")
            _WARNED_NO_PIL = True
        return None

    suffix = path.suffix.lower()
    tmp_png: Path | None = None
    try:
        if suffix in STILL_SUFFIXES:
            # Best-effort HEIC registration — silent if pillow-heif missing
            # since PIL will raise UnidentifiedImageError below and we'll
            # return None gracefully.
            if suffix in (".heic", ".heif"):
                try:
                    import pillow_heif  # type: ignore[import-not-found]
                    pillow_heif.register_heif_opener()
                except ImportError:
                    pass
            try:
                with Image.open(path) as img:
                    return _dhash_from_pil_image(img)
            except (OSError, ValueError):
                return None
        elif suffix in VIDEO_SUFFIXES:
            tmp_png = _extract_middle_frame_png(path)
            if tmp_png is None:
                return None
            try:
                with Image.open(tmp_png) as img:
                    return _dhash_from_pil_image(img)
            except (OSError, ValueError):
                return None
        else:
            # Unknown extension — try PIL as a last resort.
            try:
                with Image.open(path) as img:
                    return _dhash_from_pil_image(img)
            except (OSError, ValueError):
                return None
    finally:
        if tmp_png is not None:
            try:
                tmp_png.unlink()
            except OSError:
                pass


def hamming_distance(a: str, b: str) -> int:
    """Bit-level Hamming distance between two equal-length hex strings.

    Mismatched lengths return 64 (i.e. "definitely different") rather than
    raising — keeps the grouping code linear and crash-free against any
    legacy catalog rows with a different hash format."""
    if len(a) != len(b):
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def group_duplicates(items: list[tuple[str, str | None]],
                     threshold: int = 8) -> dict[str, str | None]:
    """Group paths whose phashes are within `threshold` bits of each other.

    `items` is a list of (path, phash | None). Items without a phash map to
    None (they're treated as singletons; the renderer mustn't dedupe them).
    Singletons (no near-twin) also map to None — group ids are only minted
    when at least two paths cluster.

    Grouping uses union-find: hash distance is not transitive, but we WANT
    transitive closure here ("if A≈B and B≈C, treat A,B,C as one shot")
    because long sequences of nearly-identical frames otherwise drip-feed
    into the plan one at a time.

    Returns `{path: group_id_or_None}`. Group ids are deterministic strings
    `"g1", "g2", ...` numbered by first-occurrence order in `items`.
    """
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Lower index wins so group numbering follows input order.
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    # O(n^2) is fine here — a folder of 5000 clips is 12.5M cheap XOR ops;
    # we already pay way more than that on motion/audio extraction.
    for i in range(n):
        _, hi = items[i]
        if hi is None:
            continue
        for j in range(i + 1, n):
            _, hj = items[j]
            if hj is None:
                continue
            if hamming_distance(hi, hj) <= threshold:
                union(i, j)

    # Count cluster sizes so we can suppress singleton "groups".
    sizes: dict[int, int] = {}
    for i in range(n):
        if items[i][1] is None:
            continue
        sizes[find(i)] = sizes.get(find(i), 0) + 1

    # Assign group ids deterministically by first-occurrence of the root.
    group_id_for_root: dict[int, str] = {}
    next_id = 1
    result: dict[str, str | None] = {}
    for i, (path, phash) in enumerate(items):
        if phash is None:
            result[path] = None
            continue
        root = find(i)
        if sizes.get(root, 0) < 2:
            result[path] = None
            continue
        gid = group_id_for_root.get(root)
        if gid is None:
            gid = f"g{next_id}"
            next_id += 1
            group_id_for_root[root] = gid
        result[path] = gid
    return result
