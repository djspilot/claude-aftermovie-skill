"""Per-second motion energy via ffmpeg signalstats."""
from __future__ import annotations

import re
from pathlib import Path

from aftermovie.ffmpeg_cmd import run


def measure_motion_energy(path: Path, duration: float) -> list[float]:
    """
    Estimate per-second motion energy using ffmpeg's `signalstats` filter.

    Returns a list of per-second motion magnitude (arbitrary units, higher = more motion).
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-vf", "fps=2,signalstats,metadata=mode=print:file=-",
        "-f", "null", "-",
    ]
    try:
        res = run(cmd, capture=True, check=False)
        text = res.stdout + res.stderr
    except Exception:
        return [0.0] * max(1, int(duration))
    pattern = re.compile(r"YDIF=([\d.]+)")
    diffs = [float(m.group(1)) for m in pattern.finditer(text)]
    if not diffs:
        return [0.0] * max(1, int(duration))
    n_sec = max(1, int(duration))
    binned = []
    for i in range(n_sec):
        chunk = diffs[i * 2 : (i + 1) * 2]
        binned.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return binned
