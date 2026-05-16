"""Music analysis via librosa: tempo, beats, downbeats, intro boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def analyze_song(song_path: Path) -> dict[str, Any]:
    """
    Use librosa to get tempo, beat times, and estimated downbeats.
    Returns: {duration_s, tempo_bpm, beats, downbeats, intro_end_s}
    """
    import numpy as np
    import librosa

    y, sr = librosa.load(str(song_path), sr=22050, mono=True)
    duration = len(y) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = beat_frames.tolist() if hasattr(beat_frames, "tolist") else list(beat_frames)
    tempo_val = float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])

    downbeats = beats[::4] if len(beats) >= 4 else beats

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)
    if len(onset_env) > 0:
        threshold = float(np.percentile(onset_env, 70))
        above = np.where(onset_env > threshold)[0]
        intro_end = float(onset_times[above[0]]) if len(above) else 0.0
    else:
        intro_end = 0.0

    return {
        "duration_s": duration,
        "tempo_bpm": tempo_val,
        "beats": beats,
        "downbeats": downbeats,
        "intro_end_s": intro_end,
    }
