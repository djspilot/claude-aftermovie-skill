"""Import sources — pull mixed footage off connected devices into a folder.

The Module owns one job: given a date range, find every still / video the
user might want to feed into `aftermovie auto` and copy it into a Source
folder. The actual analyze → score → render pipeline never sees this code;
the only output is files on disk.

Two Adapters ship today:

  * `PhotosLibraryAdapter` — iCloud-synced iPhone library at
    `~/Pictures/Photos Library.photoslibrary` via the optional `osxphotos`
    Python module.
  * `GoProAdapter` — one instance per mounted volume under `/Volumes/` that
    carries a `DCIM/` subtree (the GoPro / iPhone DCIM convention).

Both fulfil the `ImportSource` Protocol so the CLI, the future GUI, and a
future HTTP endpoint can iterate `all_sources()` and treat each Adapter
uniformly. New devices (DJI drone SD card, Android MTP) bolt on by adding a
new file under this package and prepending the registry tuple — no churn in
the call sites.
"""
from __future__ import annotations

from aftermovie.import_sources.base import (
    CopyResult,
    ImportItem,
    ImportSource,
    all_sources,
)

__all__ = [
    "CopyResult",
    "ImportItem",
    "ImportSource",
    "all_sources",
]
