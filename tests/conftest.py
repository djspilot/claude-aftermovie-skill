"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def clip_a(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clip_a.mp4"


@pytest.fixture
def clip_b(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clip_b.mp4"


@pytest.fixture
def clip_c(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clip_c.mp4"


@pytest.fixture
def tone(fixtures_dir: Path) -> Path:
    return fixtures_dir / "tone.wav"


@pytest.fixture
def hilight_sample(fixtures_dir: Path) -> Path:
    return fixtures_dir / "hilight_sample.mp4"
