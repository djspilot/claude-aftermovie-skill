"""GoPro HiLight tag reader (HMMT atom in MP4 udta box)."""
from __future__ import annotations

import struct
from pathlib import Path


def read_hilight_tags(path: Path) -> list[int]:
    """
    Extract HiLight tag timestamps (in milliseconds) from a GoPro MP4.

    HiLight tags live in the moov/udta/HMMT atom, NOT in the GPMF metadata
    stream. Structure:
        HMMT box header (8 bytes: size + 'HMMT')
        uint32_be count
        uint32_be timestamp_ms[count]

    Returns [] for non-GoPro files or files without HiLight tags.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return []
    idx = data.find(b"HMMT")
    if idx < 0:
        return []
    try:
        box_size = struct.unpack(">I", data[idx - 4 : idx])[0]
    except struct.error:
        return []
    payload = data[idx + 4 : idx - 4 + box_size]
    if len(payload) < 4:
        return []
    count = struct.unpack(">I", payload[:4])[0]
    if count > 1000 or len(payload) < 4 + count * 4:
        return []
    return list(struct.unpack(f">{count}I", payload[4 : 4 + count * 4]))
