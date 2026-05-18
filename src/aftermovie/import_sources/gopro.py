"""Adapter over any mounted volume that carries a `DCIM/` subtree.

The DCIM convention is universal across GoPro, iPhone-as-camera, drones,
mirrorless cameras — anything that wants a generic photo-import workflow
on macOS exposes its filesystem like this. We treat one mounted volume as
one Adapter instance; `detect_gopro_mounts()` enumerates them.

Capture-time strategy mirrors `analyze.capture_time`:
  1. exiftool `DateTimeOriginal` / `CreateDate` (when exiftool is on PATH)
  2. ffprobe `format.tags.creation_time` for videos
  3. file mtime as the final fallback

We skip GoPro proxy / thumbnail siblings (`*.LRV`, `*.THM`) by default —
they're low-res derivatives of the primary `.MP4` and would only inflate
the copy time. A future flag could opt them in.

Live Photos that landed on the card as paired HEIC+MOV siblings (rare but
possible: AirDrop, iPhone-as-camera) are surfaced as kind="live_photo"
with the MOV under `extra["live_photo_mov"]`, same shape as the Photos
library Adapter — so the downstream copy path is identical.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from aftermovie.ffmpeg_cmd import log
from aftermovie.import_sources.base import (
    CopyResult,
    ImportItem,
    ProgressCb,
    copy_files,
)
from aftermovie.optional_dep import optional_command


_EXIFTOOL = optional_command(
    "exiftool",
    warning="  ! exiftool not found — falling back to ffprobe + mtime for "
            "GoPro capture times. brew install exiftool for accuracy.",
)


# Extensions we copy. LRV/THM are the GoPro proxy + thumbnail siblings;
# they're not interesting source material on their own.
_INCLUDED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".insv", ".360"}
_INCLUDED_STILL_EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".dng", ".raw"}
_SKIPPED_EXTS = {".lrv", ".thm"}


def detect_gopro_mounts() -> list[Path]:
    """Return every `/Volumes/<name>` that contains a `DCIM/` subtree.

    Stable alphabetical ordering — so two registry passes in one process
    list the same mounts in the same order even if the OS reshuffles them.
    """
    volumes = Path("/Volumes")
    if not volumes.is_dir():
        return []
    out: list[Path] = []
    try:
        for vol in sorted(volumes.iterdir()):
            if not vol.is_dir():
                continue
            if (vol / "DCIM").is_dir():
                out.append(vol)
    except OSError:
        return []
    return out


def _capture_time_for(path: Path) -> float:
    """Best-effort POSIX timestamp for `path`. Falls back to mtime.

    Tries exiftool first (handles HEIC/MP4/DNG uniformly), then ffprobe for
    videos, then mtime. We log nothing per file — the caller will see the
    one-shot exiftool warning if it's missing.
    """
    if _EXIFTOOL.available:
        t = _from_exiftool(path)
        if t is not None:
            return t
    if path.suffix.lower() in _INCLUDED_VIDEO_EXTS:
        t = _from_ffprobe(path)
        if t is not None:
            return t
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _from_exiftool(path: Path) -> float | None:
    """Read DateTimeOriginal / CreateDate via exiftool -j."""
    try:
        out = subprocess.run(
            ["exiftool", "-j", "-DateTimeOriginal", "-CreateDate", str(path)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        if not isinstance(data, list) or not data:
            return None
        entry = data[0]
        for key in ("DateTimeOriginal", "CreateDate"):
            raw = entry.get(key)
            if not raw:
                continue
            # exiftool format: 'YYYY:MM:DD HH:MM:SS' (sometimes with subsec).
            # Strip any timezone suffix exiftool may emit.
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
                try:
                    return datetime.strptime(raw.split("+")[0].split("-0")[0].strip(), fmt).timestamp()
                except ValueError:
                    continue
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, ValueError):
        return None
    return None


def _from_ffprobe(path: Path) -> float | None:
    """Mirror of analyze.capture_time._from_ffprobe — kept local to avoid
    coupling import_sources to the analyze package."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format_tags=creation_time",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        ct = (data.get("format", {}) or {}).get("tags", {}).get("creation_time")
        if not ct:
            return None
        if ct.endswith("Z"):
            ct = ct[:-1] + "+00:00"
        return datetime.fromisoformat(ct).timestamp()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, ValueError, KeyError):
        return None


def _kind_for(path: Path) -> str:
    """Map an extension to the ImportItem.kind vocabulary."""
    ext = path.suffix.lower()
    if ext in _INCLUDED_VIDEO_EXTS:
        return "video"
    return "still"


class GoProAdapter:
    """ImportSource over a single mounted volume with a DCIM/ subtree.

    Construction is cheap — just records the mount path. The expensive
    DCIM walk happens in `list_in_range` so callers can iterate the
    registry without paying for I/O on every volume.
    """

    name = "gopro"

    def __init__(self, mount: Path) -> None:
        self.mount = Path(mount)
        self.label = f"GoPro ({self.mount.name})"

    def available(self) -> bool:
        return (self.mount / "DCIM").is_dir()

    def list_in_range(self, since: datetime, until: datetime) -> list[ImportItem]:
        dcim = self.mount / "DCIM"
        if not dcim.is_dir():
            return []
        since_ts = since.timestamp()
        until_ts = until.timestamp()
        items: list[ImportItem] = []
        # Build a side-lookup of HEIC stems so paired-MOV siblings can be
        # collapsed into one Live Photo ImportItem.
        all_files: list[Path] = []
        try:
            for p in dcim.rglob("*"):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in _SKIPPED_EXTS:
                    continue
                if ext in _INCLUDED_VIDEO_EXTS or ext in _INCLUDED_STILL_EXTS:
                    all_files.append(p)
        except OSError as e:
            log(f"  ! gopro: cannot walk {dcim}: {e}")
            return []

        # Live-Photo pairing: a HEIC + same-stem MOV in the same dir.
        heic_stems = {
            (p.parent, p.stem): p
            for p in all_files if p.suffix.lower() in (".heic", ".heif")
        }
        consumed_movs: set[Path] = set()
        for p in all_files:
            if p.suffix.lower() == ".mov":
                heic = heic_stems.get((p.parent, p.stem))
                if heic is not None:
                    consumed_movs.add(p)

        for p in all_files:
            if p in consumed_movs:
                continue  # surfaced via the HEIC's live_photo_mov
            ts = _capture_time_for(p)
            if ts < since_ts or ts > until_ts:
                continue
            extra: dict = {}
            kind = _kind_for(p)
            if p.suffix.lower() in (".heic", ".heif"):
                mov_sibling = p.with_suffix(".MOV")
                if not mov_sibling.is_file():
                    mov_sibling = p.with_suffix(".mov")
                if mov_sibling.is_file():
                    extra["live_photo_mov"] = str(mov_sibling)
                    kind = "live_photo"
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            items.append(ImportItem(
                src_path=str(p),
                captured_at=ts,
                kind=kind,
                size_bytes=size,
                source_label=self.label,
                extra=extra,
            ))
        items.sort(key=lambda it: (it.captured_at, it.src_path))
        return items

    def copy_into(
        self,
        items: list[ImportItem],
        dest_folder: Path,
        progress_cb: ProgressCb | None = None,
    ) -> CopyResult:
        return copy_files(items, dest_folder, progress_cb)
