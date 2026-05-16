"""Per-second audio RMS via ffmpeg astats."""
from __future__ import annotations

import re
from pathlib import Path

from aftermovie.ffmpeg_cmd import run


def measure_audio_energy(path: Path, duration: float) -> list[float]:
    """Per-second RMS of the audio track (voices, cheering, music)."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f", "null", "-",
    ]
    try:
        res = run(cmd, capture=True, check=False)
        text = res.stdout + res.stderr
    except Exception:
        return [0.0] * max(1, int(duration))
    pattern = re.compile(r"RMS_level=(-?[\d.]+)")
    levels = [float(m.group(1)) for m in pattern.finditer(text)]
    if not levels:
        return [0.0] * max(1, int(duration))
    normed = [max(0.0, min(1.0, (lvl + 60) / 54)) for lvl in levels]
    n_sec = max(1, int(duration))
    bucket = max(1, len(normed) // n_sec)
    return [sum(normed[i * bucket : (i + 1) * bucket]) / bucket for i in range(n_sec)]
