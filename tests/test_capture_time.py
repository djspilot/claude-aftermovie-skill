"""Filename-based capture-time extraction (WhatsApp strips EXIF)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from aftermovie.analyze.capture_time import _from_filename, captured_at_for


def test_whatsapp_export_style_full_timestamp():
    ts = _from_filename(Path("00003021-PHOTO-2026-04-08-08-43-11.jpg"))
    assert ts == datetime(2026, 4, 8, 8, 43, 11).timestamp()
    vid = _from_filename(Path("00003032-VIDEO-2026-04-09-23-38-23.mp4"))
    assert vid == datetime(2026, 4, 9, 23, 38, 23).timestamp()
    # Photos and videos from the same chat interleave by real send moment.
    assert ts < vid


def test_whatsapp_inapp_style_date_plus_sequence():
    a = _from_filename(Path("IMG-20260408-WA0012.jpg"))
    b = _from_filename(Path("VID-20260408-WA0034.mp4"))
    assert a is not None and b is not None
    assert a < b  # same day; WA sequence keeps the send order
    assert a == datetime(2026, 4, 8).timestamp() + 12


def test_no_timestamp_in_name_returns_none():
    assert _from_filename(Path("GOPR0042.MP4")) is None
    assert _from_filename(Path("IMG_1234.HEIC")) is None
    # Garbage date must not raise.
    assert _from_filename(Path("x-2026-99-99-99-99-99.jpg")) is None


def test_captured_at_prefers_filename_over_mtime(tmp_path: Path):
    """A metadata-less file whose NAME carries the moment must not fall
    back to (download-time) mtime."""
    p = tmp_path / "00000001-PHOTO-2026-04-08-08-43-11.jpg"
    p.write_bytes(b"not a real jpeg")  # EXIF read fails → next strategies
    ts = captured_at_for(p)
    assert ts == datetime(2026, 4, 8, 8, 43, 11).timestamp()
