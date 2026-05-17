"""Per-second audio RMS via ffmpeg astats."""
from __future__ import annotations

import re
from pathlib import Path

from aftermovie.ffmpeg_cmd import run


def _rms_per_second(path: Path, duration: float, prefilter: str = "") -> list[float]:
    """Per-second normalized RMS [0,1] of `path`, optionally bandpass-filtered.

    `prefilter` is an ffmpeg audio filter chain inserted before astats — e.g.
    `highpass=f=200,lowpass=f=3000` to isolate the voice band.
    """
    af = (prefilter + "," if prefilter else "") + \
        "astats=metadata=1:reset=1," \
        "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-af", af,
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


def measure_audio_energy(path: Path, duration: float) -> list[float]:
    """Per-second broadband RMS (full spectrum)."""
    return _rms_per_second(path, duration)


def measure_voice_energy(path: Path, duration: float) -> list[float]:
    """Per-second RMS limited to the 200–3000 Hz voice band.

    Wind / motor rumble lives below 200 Hz; cymbal hiss above 3000 Hz. The
    in-band energy approximates speech, laughter and sharp impacts — the
    sounds we usually want to surface over the music.
    """
    return _rms_per_second(path, duration, prefilter="highpass=f=200,lowpass=f=3000")
