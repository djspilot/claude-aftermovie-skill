"""Adapter over the macOS Photos library, via the optional `osxphotos` dep.

iCloud-synced iPhone photos and videos all land in
`~/Pictures/Photos Library.photoslibrary`. We don't parse the SQLite
ourselves — `osxphotos.PhotosDB` does the heavy lifting. The Adapter just
filters by date and maps PhotoInfo → ImportItem.

Live Photos: when `photo.live_photo` is True, the paired MOV path comes
from `photo.path_live_photo`. We surface BOTH paths in one ImportItem with
the MOV under `extra["live_photo_mov"]` so the shared `copy_files` helper
copies them as a pair (same stem). Downstream `analyze.stills` already
knows to re-pair HEIC+MOV siblings.

osxphotos is an optional dep — install with `pip install aftermovie[import]`.
When it's missing, `available()` returns False and the registry silently
skips us. We never raise at import time so the rest of the CLI stays usable
on machines without macOS Photos.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from aftermovie.ffmpeg_cmd import log
from aftermovie.import_sources.base import (
    CopyResult,
    ImportItem,
    ProgressCb,
    copy_files,
)
from aftermovie.optional_dep import optional_import


_OSXPHOTOS = optional_import(
    "osxphotos",
    warning="  ! osxphotos not installed — Photos library import disabled. "
            "Install with: pip install aftermovie[import]",
)


DEFAULT_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary"


class PhotosLibraryAdapter:
    """ImportSource over a macOS Photos library bundle.

    `library_path` defaults to the standard `~/Pictures/Photos Library.photoslibrary`
    location; callers can point at an alternate library by passing one
    explicitly (useful for tests + multi-library users).
    """

    name = "photos_library"
    label = "Photos library"

    def __init__(self, library_path: Path | None = None) -> None:
        self.library_path = (library_path or DEFAULT_LIBRARY).expanduser()

    def available(self) -> bool:
        """True iff osxphotos imports AND the library bundle exists on disk."""
        if not _OSXPHOTOS.available:
            return False
        return self.library_path.is_dir()

    def _open_db(self):
        """Return an `osxphotos.PhotosDB` or None (with one-shot warn)."""
        osxphotos = _OSXPHOTOS.require()
        if osxphotos is None:
            return None
        try:
            return osxphotos.PhotosDB(library_path=str(self.library_path))
        except Exception as e:
            # Library locked, sqlite missing, etc. — degrade to "no items".
            log(f"  ! photos_library: cannot open {self.library_path}: {e}")
            return None

    def list_in_range(self, since: datetime, until: datetime) -> list[ImportItem]:
        """Return every PhotoInfo whose `date` falls in [since, until]."""
        db = self._open_db()
        if db is None:
            return []
        items: list[ImportItem] = []
        for photo in db.photos():
            # photo.date is a timezone-aware datetime in local time. We
            # normalize both sides to naive-local for the comparison so the
            # caller can pass dates without thinking about timezones.
            pdate = photo.date
            if pdate is None:
                continue
            pdate_naive = pdate.replace(tzinfo=None) if pdate.tzinfo else pdate
            since_naive = since.replace(tzinfo=None) if since.tzinfo else since
            until_naive = until.replace(tzinfo=None) if until.tzinfo else until
            if pdate_naive < since_naive or pdate_naive > until_naive:
                continue

            src = photo.path or photo.path_edited
            if not src:
                # Original not downloaded from iCloud yet — skip silently.
                continue
            src_path = Path(src)
            if not src_path.is_file():
                continue

            extra: dict = {}
            if getattr(photo, "live_photo", False):
                mov = photo.path_live_photo
                if mov and Path(mov).is_file():
                    extra["live_photo_mov"] = str(mov)
                    kind = "live_photo"
                else:
                    kind = "still"
            elif getattr(photo, "ismovie", False):
                kind = "video"
            else:
                kind = "still"

            try:
                size = src_path.stat().st_size
            except OSError:
                size = 0

            items.append(ImportItem(
                src_path=str(src_path),
                captured_at=pdate.timestamp(),
                kind=kind,
                size_bytes=size,
                source_label=self.label,
                extra=extra,
            ))
        # Stable order: chronological, then path — keeps progress logs sane.
        items.sort(key=lambda it: (it.captured_at, it.src_path))
        return items

    def copy_into(
        self,
        items: list[ImportItem],
        dest_folder: Path,
        progress_cb: ProgressCb | None = None,
    ) -> CopyResult:
        return copy_files(items, dest_folder, progress_cb)
