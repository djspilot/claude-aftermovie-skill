"""Chip detection Module.

A tiny Adapter over `sysctl` (darwin) that exposes a `ChipInfo` Interface
the renderer's encoder-selection Seam can read without touching subprocess
itself. Non-darwin platforms still get a well-shaped `ChipInfo` so callers
don't have to special-case the OS.

The `media_engines` count is an approximation — Apple does not publish a
direct sysctl for it, so we map chip family → heuristic. Conservative
default is 1; selection code that relies on it should treat the value as
a hint rather than a hard bound.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ChipInfo:
    """Hardware fingerprint used by the encoder-selection Seam.

    Attributes:
        brand:         Free-form brand string ("Apple M5 Pro", "Intel...",
                       or "Generic" off-darwin).
        arch:          `platform.machine()` (e.g. "arm64", "x86_64").
        perf_cores:    Performance-core count when knowable, else 0.
        eff_cores:     Efficiency-core count when knowable, else 0.
        media_engines: Approximate VideoToolbox media-engine count
                       (heuristic; conservative default 1).
    """

    brand: str
    arch: str
    perf_cores: int
    eff_cores: int
    media_engines: int

    @property
    def is_apple_silicon(self) -> bool:
        return self.arch == "arm64" and self.brand.lower().startswith("apple")


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sysctl_int(key: str) -> int:
    raw = _sysctl(key)
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _media_engines_for(brand: str) -> int:
    """Approximate media-engine count for an Apple Silicon chip.

    Sources: Apple's tech briefs at chip-launch time. The 'Pro/Max/Ultra'
    suffix is the strongest signal; bare 'M1'/'M2'/'M3' have a single
    engine, 'Pro' typically has 1 engine that's wider than the base, 'Max'
    doubles it, 'Ultra' doubles again. We round down on uncertainty so
    parallelism heuristics don't oversubscribe.
    """
    b = brand.lower()
    if "ultra" in b:
        return 4
    if "max" in b:
        return 2
    if "pro" in b:
        # M1 Pro / M2 Pro / M3 Pro / M5 Pro: 1 engine in practice (wider).
        # M2 Max-era marketing claims '2x media' for Pro on newer gens, but
        # the safe assumption for parallel encode oversubscription is 1.
        return 1
    if re.search(r"\bapple m\d+\b", b):
        return 1
    return 1


def detect_chip() -> ChipInfo:
    """Best-effort hardware probe.

    On darwin we call `sysctl` for brand + per-perflevel core counts.
    On any other OS, we return a generic ChipInfo so the caller's
    encoder-selection Seam falls through to x264.
    """
    arch = platform.machine() or "unknown"
    if platform.system() != "Darwin":
        return ChipInfo(
            brand="Generic",
            arch=arch,
            perf_cores=os.cpu_count() or 0,
            eff_cores=0,
            media_engines=0,
        )

    brand = _sysctl("machdep.cpu.brand_string") or "Unknown"
    # Apple's perflevel naming has drifted across generations: M1–M3 use
    # perflevel0 for performance and perflevel1 for efficiency, while the
    # M5 Pro labels them "Super" + "Performance" with the wider cluster on
    # perflevel1. Treat the larger physical-cpu count as `perf_cores` and
    # the smaller as `eff_cores` so the doctor output matches user
    # expectations across all M-series chips.
    level0 = _sysctl_int("hw.perflevel0.physicalcpu")
    level1 = _sysctl_int("hw.perflevel1.physicalcpu")
    perf = max(level0, level1)
    eff = min(level0, level1)
    if perf == 0 and eff == 0:
        # Pre-Apple-Silicon Mac (or sysctl shape we don't know): fall back
        # to total physical cores so perf_cores still reflects something.
        perf = _sysctl_int("hw.physicalcpu") or (os.cpu_count() or 0)
    engines = _media_engines_for(brand) if arch == "arm64" else 0
    return ChipInfo(
        brand=brand,
        arch=arch,
        perf_cores=perf,
        eff_cores=eff,
        media_engines=engines,
    )


__all__ = ["ChipInfo", "detect_chip"]
