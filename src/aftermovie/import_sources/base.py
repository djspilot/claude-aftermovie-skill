"""Contracts shared by every ImportSource Adapter.

The `ImportItem` + `CopyResult` dataclasses are the Interface across the
Module boundary — the CLI, the future GUI, and the future HTTP service all
consume them so each Adapter only needs to satisfy this one shape.

`all_sources()` is the registry: photos_library first (if osxphotos is
importable), then one GoProAdapter per mounted volume under /Volumes/ that
carries a DCIM/ subtree. Ordering is stable so UI lists don't shuffle
between runs.

Helpers (`copy_files`) live here because both Adapters share the same
idempotent-copy invariant: skip when dest already exists with same size,
log + skip when sizes differ, preserve mtime via shutil.copy2.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from aftermovie.ffmpeg_cmd import log


# Progress callback shape: (done, total, current_path_or_none).
ProgressCb = Callable[[int, int, str | None], None]


@dataclass
class ImportItem:
    """One file the Adapter has identified as in-range for import.

    `src_path` is an absolute path on disk (osxphotos-managed paths count —
    they live under the Photos library bundle but still resolve via shutil).
    `kind` is the coarse media class the GUI uses for filtering; we keep it
    a string (not an Enum) so JSON serialization is one-liner-trivial.
    `extra` is the Adapter-specific escape hatch — Live Photos drop the
    paired MOV path under `live_photo_mov`, GoPro proxies could drop an LRV
    sibling here in future, etc.
    """
    src_path: str
    captured_at: float          # POSIX seconds; we don't need timezone fidelity
    kind: str                   # "video" | "still" | "live_photo"
    size_bytes: int             # 0 if unknown
    source_label: str           # human label e.g. "Photos library" / "GoPro (Untitled)"
    extra: dict = field(default_factory=dict)


@dataclass
class CopyResult:
    """Aggregate outcome of `copy_into`. The CLI prints this verbatim."""
    copied: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_written: int = 0
    dest_folder: str = ""


class ImportSource(Protocol):
    """Adapter Interface every device gateway honors.

    The `name` is a stable id the CLI consumes through `--sources` (must not
    contain commas). `label` is the human-readable form for UIs.
    `available()` is the cheap predicate the registry uses to gate listing
    — false means we'll skip the source entirely without surfacing it.
    """
    name: str
    label: str

    def available(self) -> bool:
        ...

    def list_in_range(self, since: datetime, until: datetime) -> list[ImportItem]:
        ...

    def copy_into(
        self,
        items: list[ImportItem],
        dest_folder: Path,
        progress_cb: ProgressCb | None = None,
    ) -> CopyResult:
        ...


# ---------------------------------------------------------------------------
# Shared copy helper. Both Adapters' `copy_into` route through this so the
# idempotent-skip + log-on-conflict invariants live in one place.
# ---------------------------------------------------------------------------


def _safe_copy(src: Path, dest: Path) -> tuple[bool, int]:
    """Copy `src` → `dest` if it doesn't already exist with the same size.

    Returns (copied, bytes_written). `copied=False` is the
    idempotent-skip path; we never overwrite a dest of a different size,
    we log + skip instead so re-runs are safe.
    """
    try:
        src_size = src.stat().st_size
    except OSError as e:
        log(f"  ! import: cannot stat source {src}: {e}")
        raise
    if dest.exists():
        try:
            dest_size = dest.stat().st_size
        except OSError:
            dest_size = -1
        if dest_size == src_size:
            return False, 0
        log(
            f"  ! import: {dest.name} already exists with different size "
            f"({dest_size} vs {src_size}); skipping (not overwriting)."
        )
        return False, 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)  # preserves mtime
    return True, src_size


def copy_files(
    items: list[ImportItem],
    dest_folder: Path,
    progress_cb: ProgressCb | None = None,
    *,
    log_every: int = 10,
) -> CopyResult:
    """Copy each item's src_path (and any paired Live Photo MOV) into dest.

    The Adapter-shared invariants:
      - Live Photos: copy both HEIC and the paired MOV, preserving the same
        stem so downstream `analyze.stills` re-pairs them.
      - Skip if `dest / src.name` already exists with same size.
      - Never overwrite an existing dest file that has different size.
      - Preserve mtime via shutil.copy2.
      - Log one line per `log_every` files copied + a final summary.
    """
    dest_folder = dest_folder.expanduser().resolve()
    dest_folder.mkdir(parents=True, exist_ok=True)
    res = CopyResult(dest_folder=str(dest_folder))
    total = sum(1 + (1 if it.extra.get("live_photo_mov") else 0) for it in items)
    done = 0
    for item in items:
        # Primary file.
        src = Path(item.src_path)
        dest = dest_folder / src.name
        try:
            copied, bytes_w = _safe_copy(src, dest)
        except OSError:
            res.failed += 1
            done += 1
            if progress_cb:
                progress_cb(done, total, str(src))
            continue
        if copied:
            res.copied += 1
            res.bytes_written += bytes_w
            if res.copied % log_every == 0:
                log(f"  copied {res.copied} files ({res.bytes_written / 1e6:.1f} MB)")
        else:
            res.skipped += 1
        done += 1
        if progress_cb:
            progress_cb(done, total, str(src))

        # Paired Live Photo MOV — only when the primary was a HEIC/JPG.
        mov_path = item.extra.get("live_photo_mov")
        if mov_path:
            mov_src = Path(mov_path)
            mov_dest = dest_folder / mov_src.name
            try:
                copied, bytes_w = _safe_copy(mov_src, mov_dest)
            except OSError:
                res.failed += 1
                done += 1
                if progress_cb:
                    progress_cb(done, total, str(mov_src))
                continue
            if copied:
                res.copied += 1
                res.bytes_written += bytes_w
                if res.copied % log_every == 0:
                    log(f"  copied {res.copied} files ({res.bytes_written / 1e6:.1f} MB)")
            else:
                res.skipped += 1
            done += 1
            if progress_cb:
                progress_cb(done, total, str(mov_src))
    return res


# ---------------------------------------------------------------------------
# Registry. photos_library always first so the GUI's default expansion shows
# the iCloud catalog before any inserted SD card.
# ---------------------------------------------------------------------------


def all_sources() -> list[ImportSource]:
    """Return every ImportSource present on this machine, in stable order.

    photos_library lands first if `osxphotos` is importable; one
    GoProAdapter follows per `/Volumes/<name>/DCIM/` mount the OS exposes;
    finally one GoProICCAdapter per HERO/GoPro reached over MTP via
    Apple's ImageCaptureCore (when `pyobjc-framework-ImageCaptureCore` is
    importable). Adapter constructors must be cheap — `available()` is the
    predicate callers use to gate UI/listing work, NOT construction.

    Dedup: if a HERO presents BOTH as a Mass-Storage mount AND as an MTP
    camera (rare — user manually toggled the mode), we skip the ICC
    Adapter so the simpler filesystem path wins; the uuid match is the
    handle for that comparison.
    """
    # Local imports so a missing optional dep (osxphotos / pyobjc) doesn't
    # break `from aftermovie.import_sources import all_sources` at module
    # load.
    from aftermovie.import_sources.photos import PhotosLibraryAdapter
    from aftermovie.import_sources.gopro import GoProAdapter, detect_gopro_mounts

    out: list[ImportSource] = [PhotosLibraryAdapter()]
    ms_mounts = list(detect_gopro_mounts())
    for mount in ms_mounts:
        out.append(GoProAdapter(mount))

    # ICC GoPros — only added when the optional dep loads. Browse failures
    # (no camera, ICC unavailable) degrade to an empty list.
    try:
        from aftermovie.import_sources.gopro_icc import detect_icc_gopros
        icc_adapters = detect_icc_gopros()
    except Exception as e:
        log(f"  ! gopro_icc: registry skipped ({e})")
        icc_adapters = []

    # Dedup against Mass-Storage mounts. The only stable handle we have on
    # both sides is the volume name vs. the camera name; both surface
    # "HERO9" / "HERO10" so a substring match across the existing labels
    # is the cheap-and-correct check.
    ms_labels = {a.label for a in out if isinstance(a, GoProAdapter)}
    for icc in icc_adapters:
        if any(icc.camera_name in lbl or lbl.split("(")[-1].rstrip(")") in icc.camera_name
               for lbl in ms_labels):
            continue
        out.append(icc)
    return out
