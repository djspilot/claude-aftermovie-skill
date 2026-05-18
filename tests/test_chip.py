"""Tests for the chip-detection Module (B3)."""
from __future__ import annotations

import platform
from unittest.mock import patch

import pytest

from aftermovie.render.chip import ChipInfo, detect_chip


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="sysctl is darwin-only")
def test_detect_chip_on_darwin_returns_non_empty_brand():
    info = detect_chip()
    assert isinstance(info, ChipInfo)
    assert info.brand and info.brand != "Unknown"
    assert info.arch in ("arm64", "x86_64")


def test_detect_chip_off_darwin_returns_generic():
    with patch("aftermovie.render.chip.platform.system", return_value="Linux"), \
         patch("aftermovie.render.chip.platform.machine", return_value="x86_64"):
        info = detect_chip()
    assert info.brand == "Generic"
    assert info.media_engines == 0
    assert info.is_apple_silicon is False


def test_detect_chip_apple_m5_pro_heuristic():
    """Stubbed sysctl returning 'Apple M5 Pro' yields a sensible ChipInfo."""
    def fake_sysctl(key: str) -> str | None:
        return {
            "machdep.cpu.brand_string": "Apple M5 Pro",
            "hw.perflevel0.physicalcpu": "10",
            "hw.perflevel1.physicalcpu": "5",
        }.get(key)

    with patch("aftermovie.render.chip.platform.system", return_value="Darwin"), \
         patch("aftermovie.render.chip.platform.machine", return_value="arm64"), \
         patch("aftermovie.render.chip._sysctl", side_effect=fake_sysctl):
        info = detect_chip()
    assert info.brand == "Apple M5 Pro"
    assert info.arch == "arm64"
    # Best-effort heuristic: Pro-class chips have >= 8 perf cores.
    assert info.perf_cores >= 8
    assert info.eff_cores >= 1
    assert info.media_engines >= 1
    assert info.is_apple_silicon is True


def test_detect_chip_m1_max_has_more_media_engines():
    def fake_sysctl(key: str) -> str | None:
        return {
            "machdep.cpu.brand_string": "Apple M1 Max",
            "hw.perflevel0.physicalcpu": "8",
            "hw.perflevel1.physicalcpu": "2",
        }.get(key)

    with patch("aftermovie.render.chip.platform.system", return_value="Darwin"), \
         patch("aftermovie.render.chip.platform.machine", return_value="arm64"), \
         patch("aftermovie.render.chip._sysctl", side_effect=fake_sysctl):
        info = detect_chip()
    assert info.media_engines == 2


def test_detect_chip_bare_m_has_single_engine():
    def fake_sysctl(key: str) -> str | None:
        return {
            "machdep.cpu.brand_string": "Apple M2",
            "hw.perflevel0.physicalcpu": "4",
            "hw.perflevel1.physicalcpu": "4",
        }.get(key)

    with patch("aftermovie.render.chip.platform.system", return_value="Darwin"), \
         patch("aftermovie.render.chip.platform.machine", return_value="arm64"), \
         patch("aftermovie.render.chip._sysctl", side_effect=fake_sysctl):
        info = detect_chip()
    assert info.media_engines == 1
