"""Telemetry tests: HMMT atom parsing + GPMF parser sanity."""
from __future__ import annotations

import struct
from pathlib import Path

from aftermovie.telemetry.gpmf import parse_gpmf_motion
from aftermovie.telemetry.hilight import read_hilight_tags


def test_hilight_finds_injected_tag(hilight_sample: Path):
    tags = read_hilight_tags(hilight_sample)
    assert tags == [500]


def test_hilight_returns_empty_for_plain_clip(clip_a: Path):
    assert read_hilight_tags(clip_a) == []


def test_hilight_returns_empty_for_missing_file(tmp_path: Path):
    assert read_hilight_tags(tmp_path / "does-not-exist.mp4") == []


def test_gpmf_parser_returns_empty_for_empty_blob():
    result = parse_gpmf_motion(b"")
    assert result == {"accl_mag": [], "gyro_mag": [], "gps_speed": []}


def test_gpmf_parser_handles_garbage_without_crashing():
    # Random bytes — parser should bail cleanly, not raise.
    blob = b"\x00\x01\x02\x03" * 64
    result = parse_gpmf_motion(blob)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"accl_mag", "gyro_mag", "gps_speed"}


def test_gpmf_parser_extracts_accl_from_synthetic_klv():
    """Build a tiny GPMF blob with one ACCL sample and verify magnitude."""
    # Header: 'SCAL' (set scale to 1000) so accl values divide cleanly.
    scal_payload = struct.pack(">h", 1000)
    scal_payload += b"\x00\x00"  # pad to 32-bit alignment
    scal_header = b"SCAL" + b"s" + bytes([2]) + struct.pack(">H", 1)
    scal_block = scal_header + scal_payload

    # ACCL: 3 int16 samples [3000, 4000, 0] → divided by SCAL 1000 → (3,4,0) → mag 5.
    accl_payload = struct.pack(">hhh", 3000, 4000, 0)
    accl_payload += b"\x00\x00"  # pad to 32-bit alignment
    accl_header = b"ACCL" + b"s" + bytes([6]) + struct.pack(">H", 1)
    accl_block = accl_header + accl_payload

    blob = scal_block + accl_block
    result = parse_gpmf_motion(blob)
    assert len(result["accl_mag"]) == 1
    assert abs(result["accl_mag"][0] - 5.0) < 0.001
