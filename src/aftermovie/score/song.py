"""Music analysis via librosa: tempo, beats, downbeats, intro boundary, sections."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Section taxonomy (Phase 4 / Phase C5):
# - intro:  the song's quiet opener (energy below song-median for the first N s)
# - verse:  the default "everything in between" (medium energy)
# - build:  a rising-energy run that leads into a drop (gradient > 0 for 4-8s)
# - drop:   a local energy peak (>1.4x median, preceded by a >50% rise in 4s)
# - outro:  the song's tail where energy drops below the median for >=4s
SECTION_KINDS = ("intro", "verse", "build", "drop", "outro")

# Detection knobs. Tuned for typical aftermovie songs (90-130 BPM, 60-300s).
# They are deliberately conservative: the *only* job here is to split the
# song into a handful of musically meaningful spans for the pacing-aware
# allocator. A track with no obvious drop should fall back to intro/verse/outro.
_INTRO_MAX_S = 12.0          # only the first 12s may be labelled intro
_OUTRO_MIN_TAIL_S = 4.0      # outro requires >=4s of below-median energy
_OUTRO_MAX_S = 12.0          # only the last 12s may be labelled outro
_DROP_MIN_RATIO = 1.4        # local-window energy >= 1.4x song median
_DROP_RISE_FRACTION = 0.5    # >50% rise across the prior 4s
_DROP_RISE_WINDOW_S = 4.0
_DROP_WINDOW_S = 1.0         # smoothing window for "local energy"
_BUILD_MIN_S = 4.0
_BUILD_MAX_S = 8.0


@dataclass(frozen=True)
class Section:
    """A musically meaningful span of the Song.

    `kind` is one of `SECTION_KINDS`. `intensity` in [0, 1] is a rough peak
    energy within the span (relative to the song's max), used by the scorer
    to size cut-density bumps inside builds and drops.
    """
    kind: str
    start_s: float
    end_s: float
    intensity: float

    def __post_init__(self) -> None:
        if self.kind not in SECTION_KINDS:
            raise ValueError(
                f"Section.kind={self.kind!r} not in {SECTION_KINDS!r}"
            )
        if self.end_s <= self.start_s:
            raise ValueError(
                f"Section.end_s ({self.end_s}) must exceed start_s ({self.start_s})"
            )
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(
                f"Section.intensity ({self.intensity}) must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_sections(
    duration_s: float,
    tempo_bpm: float,
    beats: list[float],
    energy_per_s: list[float],
) -> list[Section]:
    """Split a song into a deterministic list of Sections.

    Inputs are the same numbers `analyze_song` already computes — this
    Module has no dependency on librosa so it can be unit-tested with a
    synthetic energy curve in `tests/test_song_sections.py`. The
    `tempo_bpm` / `beats` arguments are accepted for forward-compat with
    smarter detectors; the current heuristics only consume `duration_s`
    and `energy_per_s`.

    Returns a contiguous, non-overlapping list of `Section` values
    covering `[0, duration_s)`. Always returns at least one Section
    (everything is `verse` when the energy curve is degenerate).
    """
    sec_count = int(duration_s) if duration_s > 0 else len(energy_per_s)
    if sec_count <= 0 or not energy_per_s:
        return [Section(kind="verse", start_s=0.0,
                        end_s=max(float(duration_s), 1.0), intensity=0.0)]

    # Clamp the energy curve to `sec_count` so off-by-one curves (the song
    # ends mid-second) don't blow up our indexing below.
    energy = [float(e) for e in energy_per_s[:sec_count]]
    if len(energy) < sec_count:
        energy = energy + [energy[-1]] * (sec_count - len(energy))

    # Median is robust to a single late spike; mean would bias intro/outro
    # detection when the chorus dominates the curve.
    sorted_e = sorted(energy)
    median = sorted_e[len(sorted_e) // 2]
    emax = max(energy) if energy else 1.0
    if emax <= 0:
        return [Section(kind="verse", start_s=0.0,
                        end_s=float(duration_s) or 1.0, intensity=0.0)]

    # --- intro: first run of below-median energy at the head of the song ---
    intro_end = 0
    intro_cap = min(_INTRO_MAX_S, sec_count)
    for i in range(int(intro_cap)):
        if energy[i] < median:
            intro_end = i + 1
        else:
            break

    # --- outro: last run of below-median energy at the tail, length >=4s ---
    outro_start = sec_count
    tail_run = 0
    outro_cap = max(0, sec_count - int(_OUTRO_MAX_S))
    for i in range(sec_count - 1, outro_cap - 1, -1):
        if energy[i] < median:
            tail_run += 1
            outro_start = i
        else:
            break
    if tail_run < _OUTRO_MIN_TAIL_S:
        outro_start = sec_count
        tail_run = 0

    # --- drops: local energy peaks preceded by a rising edge ---
    # A "drop second" is one whose 1s-window energy is >=1.4x the median
    # AND whose energy rose by >50% across the prior 4s. We walk left-to-right,
    # collapse adjacent drop seconds into a single drop span, and skip any
    # second that already belongs to the intro or outro.
    drop_seconds: list[int] = []
    for i in range(intro_end, outro_start):
        if energy[i] < median * _DROP_MIN_RATIO:
            continue
        rise_start = max(0, i - int(_DROP_RISE_WINDOW_S))
        prior = energy[rise_start:i] or [energy[i]]
        prior_mean = sum(prior) / len(prior)
        if prior_mean <= 0:
            continue
        rise = (energy[i] - prior_mean) / max(prior_mean, 1e-6)
        if rise < _DROP_RISE_FRACTION:
            continue
        drop_seconds.append(i)

    # Merge adjacent drop seconds into spans. A gap of 1-2s within a drop
    # is normal (the chorus has a brief breath); a gap >=3s closes the span.
    drop_spans: list[tuple[int, int]] = []  # [start_s, end_s) in seconds
    for s in drop_seconds:
        if drop_spans and s - drop_spans[-1][1] <= 2:
            drop_spans[-1] = (drop_spans[-1][0], s + 1)
        else:
            drop_spans.append((s, s + 1))

    # --- builds: rising-energy runs that lead into a drop start ---
    # For each drop's start `s0`, walk backwards until either the gradient
    # goes negative or we've consumed `_BUILD_MAX_S` seconds. The span
    # must be at least `_BUILD_MIN_S` to count.
    build_spans: list[tuple[int, int]] = []
    for s0, _s1 in drop_spans:
        b_end = s0
        b_start = s0
        for j in range(s0 - 1, max(intro_end - 1, s0 - int(_BUILD_MAX_S) - 1), -1):
            if j < intro_end:
                break
            if j + 1 >= sec_count:
                continue
            if energy[j + 1] - energy[j] <= 0:
                # gradient went non-positive; stop extending the build
                break
            b_start = j
        if b_end - b_start >= int(_BUILD_MIN_S):
            build_spans.append((b_start, b_end))

    # --- compose the timeline as a contiguous list of Sections ---
    # Walk seconds, label each, then collapse runs of identical labels into
    # spans. Precedence: outro > drop > build > intro > verse.
    drop_set: set[int] = set()
    for s0, s1 in drop_spans:
        drop_set.update(range(s0, s1))
    build_set: set[int] = set()
    for s0, s1 in build_spans:
        build_set.update(range(s0, s1))

    labels: list[str] = []
    for i in range(sec_count):
        if i >= outro_start:
            labels.append("outro")
        elif i in drop_set:
            labels.append("drop")
        elif i in build_set:
            labels.append("build")
        elif i < intro_end:
            labels.append("intro")
        else:
            labels.append("verse")

    sections: list[Section] = []
    run_start = 0
    for i in range(1, sec_count):
        if labels[i] != labels[i - 1]:
            kind = labels[i - 1]
            peak = max(energy[run_start:i]) if i > run_start else 0.0
            sections.append(Section(
                kind=kind,
                start_s=float(run_start),
                end_s=float(i),
                intensity=float(peak / emax) if emax > 0 else 0.0,
            ))
            run_start = i
    # Trailing run runs through the song's true end (not int truncation).
    kind = labels[-1]
    tail_peak = max(energy[run_start:]) if sec_count > run_start else 0.0
    sections.append(Section(
        kind=kind,
        start_s=float(run_start),
        end_s=float(duration_s) if duration_s > run_start else float(sec_count),
        intensity=float(tail_peak / emax) if emax > 0 else 0.0,
    ))
    return sections


def analyze_song(song_path: Path) -> dict[str, Any]:
    """
    Use librosa to get tempo, beat times, and estimated downbeats.
    Returns: {duration_s, tempo_bpm, beats, downbeats, intro_end_s,
              energy_per_s, onset_peaks, sections}
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

    # Strong onset peaks — used by the soft-transition heuristic to force a
    # hard cut on snare/kick hits so the visual cut lands with the song.
    onset_peaks: list[float] = []
    if len(onset_env) > 0:
        peak_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units="frames",
        )
        if len(peak_frames):
            # 98th percentile is much stricter than 92 — keeps only the
            # genuine snare/kick hits, not every beat with a transient.
            strong_thr = float(np.percentile(onset_env, 98))
            peak_frames_arr = np.asarray(peak_frames, dtype=int)
            strong_mask = onset_env[peak_frames_arr] > strong_thr
            strong_frames = peak_frames_arr[strong_mask]
            if len(strong_frames):
                peak_times = onset_times[strong_frames]
                onset_peaks = sorted(float(t) for t in peak_times.tolist())

    # Per-second RMS energy — used for pace=auto so cuts pack tighter during
    # loud sections and breathe during quieter ones. Normalised to [0, 1].
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    sec_count = int(np.ceil(duration))
    energy_per_s = np.zeros(sec_count, dtype=float)
    if sec_count > 0 and len(rms) > 0:
        for i in range(sec_count):
            mask = (rms_times >= i) & (rms_times < i + 1)
            if mask.any():
                energy_per_s[i] = float(rms[mask].mean())
        emax = float(energy_per_s.max())
        if emax > 0:
            energy_per_s = energy_per_s / emax

    energy_list = energy_per_s.tolist()
    sections = [
        s.to_dict()
        for s in detect_sections(duration, tempo_val, beats, energy_list)
    ]

    return {
        "duration_s": duration,
        "tempo_bpm": tempo_val,
        "beats": beats,
        "downbeats": downbeats,
        "intro_end_s": intro_end,
        "energy_per_s": energy_list,
        "onset_peaks": onset_peaks,
        "sections": sections,
    }
