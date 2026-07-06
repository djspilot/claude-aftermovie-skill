"""xfade-based transitions between cuts.

Three kinds:
    cut       — hard cut, no fade. Uses the concat-demuxer fast path.
    crossfade — xfade=fade, beat-scaled (~0.4 beat).
    whip      — xfade=wipeleft / wiperight, beat-scaled (~0.5 beat).

When any entry has a non-cut transition_in we switch the whole render to
filter_complex; xfade requires both source streams to be live.
"""
from __future__ import annotations

from typing import Any

# Transition durations are expressed in BEATS and converted per song, so a
# crossfade eats the same musical fraction at 90 and at 160 BPM. The beat
# fractions are chosen to reproduce the old fixed-seconds feel at 120 BPM
# (beat = 0.5s): e.g. crossfade 0.4 beat × 0.5s = the old 0.2s.
# Absolute clamps keep extreme tempos sane; the 0.9s ceiling matches the
# audio acrossfade clamp in pipeline.py.
_TDUR_MIN_S = 0.12
_TDUR_MAX_S = 0.9
# A transition may cover at most this fraction of the incoming clip's slot —
# without it a calm 0.85s dissolve can be LONGER than a tight drop cut,
# erasing the cut entirely.
_MAX_SLOT_FRACTION = 0.4


def _beat_gap_s(song_meta: dict[str, Any] | None) -> float:
    bpm = float((song_meta or {}).get("tempo_bpm") or 0.0)
    return 60.0 / bpm if bpm > 0 else 0.5


def _fit_tdur(beats: float, beat_gap_s: float, entry: dict[str, Any]) -> float:
    """Beat-count → seconds, clamped to absolute bounds and to a fraction of
    the incoming entry's slot. May return < 0.05, which build_xfade_graph
    treats as a hard cut — that's intentional for very tight slots."""
    tdur = max(_TDUR_MIN_S, min(beats * beat_gap_s, _TDUR_MAX_S))
    slot = float(entry.get("out_duration_s") or 0.0)
    if slot > 0:
        tdur = min(tdur, _MAX_SLOT_FRACTION * slot)
    return tdur

# Every input to xfade must share time_base, frame rate, pixel format, and SAR.
# concat preserves the FIRST input's timebase; xfade rewrites output to
# 1/AV_TIME_BASE (1/1000000). Chaining them without per-node normalization
# blows up with "First input link main timebase do not match the corresponding
# second input link xfade timebase". We normalize every label that feeds into
# xfade — raw inputs AND concat outputs — to AVTB.
_XFADE_NORMALIZE = "settb=AVTB,fps={fps},format=yuv420p,setsar=1"


def decide_transitions(entries: list[dict[str, Any]],
                       song_meta: dict[str, Any] | None = None,
                       *, mode: str = "auto") -> None:
    """Assign a transition_in dict to each entry in-place.

    mode="auto" (default):
      - Every entry defaults to a hard cut.
      - One whip on the highest-scoring entry overall (per 8-cut block, max 3).
      - Crossfade on every 4th entry as a structural beat marker.

    mode="soft":
      - Every entry except the first is a short (0.15s) crossfade. No whips.
        Produces a relaxed, glide-y feel — recommended for stills-heavy or
        chill-vibe edits where hard cuts feel choppy.
    """
    if mode == "soft":
        _decide_soft(entries, song_meta or {})
        return
    if not entries:
        return
    n = len(entries)
    beat_gap = _beat_gap_s(song_meta)
    for e in entries:
        e["transition_in"] = {"kind": "cut", "duration_s": 0.0}

    # Structural crossfades: on cuts that land on a musical downbeat (bar
    # start), spaced at least 4 cuts apart so most entries stay hard cuts.
    # 0.4 beat ≈ the old fixed 0.2s at 120 BPM. Without downbeat data we
    # fall back to the legacy every-4th-cut rule.
    downbeats = list((song_meta or {}).get("downbeats") or [])
    marks: list[int] = []
    if downbeats:
        last = 0
        for i in range(1, n):
            beat_t = float(entries[i].get("beat_time_s", -1.0))
            if i - last >= 4 and any(abs(beat_t - db) < 0.15 for db in downbeats):
                marks.append(i)
                last = i
    if not marks:
        marks = list(range(4, n, 4))
    for i in marks:
        entries[i]["transition_in"] = {
            "kind": "crossfade",
            "duration_s": _fit_tdur(0.4, beat_gap, entries[i]),
        }

    # Whip on the highest-scoring entry in each 8-cut block, max 3 whips total.
    whips = 0
    for block_start in range(0, n, 8):
        if whips >= 3:
            break
        block = list(enumerate(entries[block_start:block_start + 8], start=block_start))
        if not block:
            continue
        best_i, best_e = max(block, key=lambda kv: kv[1].get("score", 0))
        if best_i == 0:
            continue  # nothing to transition INTO at index 0
        direction = "wipeleft" if best_i % 2 == 0 else "wiperight"
        entries[best_i]["transition_in"] = {
            "kind": "whip",
            "duration_s": _fit_tdur(0.5, beat_gap, entries[best_i]),
            "direction": direction,
        }
        whips += 1


_PEAK_REASONS = {"high_accel_jump", "motion_peak", "hilight_tag", "audio_peak"}


def _decide_soft(entries: list[dict[str, Any]],
                 song_meta: dict[str, Any]) -> None:
    """Variable-duration soft transitions: dissolve length tracks song energy,
    downbeat proximity, and whether the cut lands on an action peak.

    The shape is what the user actually feels as "Quik-like" — long breathy
    dissolves on quiet sections, near-cuts on loud sections, and hard cuts on
    real action peaks so they hit harder.
    """
    if not entries:
        return
    energy: list[float] = list(song_meta.get("energy_per_s") or [])
    downbeats: list[float] = list(song_meta.get("downbeats") or [])
    onset_peaks: list[float] = list(song_meta.get("onset_peaks") or [])
    beat_gap = _beat_gap_s(song_meta)
    scores = sorted(float(e.get("score", 0)) for e in entries)
    top_idx = max(0, int(0.85 * len(scores)) - 1)
    top_score = scores[top_idx] if scores else 0.0

    entries[0]["transition_in"] = {"kind": "cut", "duration_s": 0.0}
    for e in entries[1:]:
        beat_t = float(e.get("beat_time_s", 0.0))
        score = float(e.get("score", 0))
        reasons = set(e.get("reasons", []) or [])

        # Hard cut if a strong onset hits within ±60ms of the beat — lines
        # the visual cut up with a snare/kick/vocal entrance. Tight window
        # because onsets are common; only the closest hits should trigger.
        if any(abs(beat_t - p) < 0.06 for p in onset_peaks):
            e["transition_in"] = {"kind": "cut", "duration_s": 0.0}
            continue

        # Hard cut on real action peaks — lands harder than a dissolve.
        if reasons & _PEAK_REASONS and score >= top_score:
            e["transition_in"] = {"kind": "cut", "duration_s": 0.0}
            continue

        e_val = 0.5
        if energy:
            idx = max(0, min(int(beat_t), len(energy) - 1))
            e_val = float(energy[idx])

        on_downbeat = any(abs(beat_t - db) < 0.15 for db in downbeats)

        # Durations in BEATS (converted by _fit_tdur), tuned to match the
        # old fixed seconds at 120 BPM — the user kept asking for slower
        # transitions, so the calm end stays breathy. On fast tracks these
        # now shrink with the beat instead of smearing across it, and
        # _fit_tdur additionally caps each fade to 40% of the incoming slot
        # so a dissolve can never swallow a tight drop cut.
        if on_downbeat and e_val >= 0.6:
            beats = 1.3   # structural-beat marker (0.65s @ 120 BPM)
        elif e_val < 0.35:
            beats = 1.7   # calm dissolve (0.85s @ 120 BPM)
        elif e_val >= 0.75:
            beats = 0.6   # tight glide in loud sections (0.30s @ 120 BPM)
        else:
            beats = 0.9   # default mid-tempo (0.45s @ 120 BPM)
        e["transition_in"] = {
            "kind": "crossfade",
            "duration_s": _fit_tdur(beats, beat_gap, e),
        }


def has_non_cut(entries: list[dict[str, Any]]) -> bool:
    return any(e.get("transition_in", {}).get("kind", "cut") != "cut" for e in entries)


def build_xfade_graph(n_inputs: int, durations: list[float],
                      entries: list[dict[str, Any]],
                      *, target_fps: int = 30) -> tuple[str, str]:
    """Return (filter_complex_string, final_label).

    Algorithm:
        1. Normalize every raw input to AVTB timebase / target fps / yuv420p
           so concat and xfade can be chained without timebase clashes.
        2. Split clips into segments at non-cut transition boundaries.
        3. Concat (filter, not demuxer) within each segment, re-normalize
           the concat output (concat preserves the first input's timebase
           but xfade demands AVTB).
        4. xfade between segments. xfade outputs at AVTB which already
           matches the normalized input timebase, so no extra step between
           successive xfades.

    `durations` is the per-cut output duration after speed-ramping. Each input
    stream is assumed to be at index 0..n_inputs-1.
    """
    norm = _XFADE_NORMALIZE.format(fps=target_fps)

    if n_inputs == 1:
        # Normalize even the single-input case so downstream overlay/title
        # filters always see a labeled stream.
        return f"[0:v]{norm}[v0]", "v0"

    # Pre-normalize every raw video input to label v{i}.
    parts: list[str] = [f"[{i}:v]{norm}[v{i}]" for i in range(n_inputs)]

    # Identify segment boundaries: each entry whose transition_in is non-cut
    # marks the START of a new segment.
    segments: list[list[int]] = [[0]]
    transitions: list[dict[str, Any]] = []
    for i in range(1, n_inputs):
        t = entries[i].get("transition_in", {"kind": "cut", "duration_s": 0.0})
        kind = t.get("kind", "cut")
        tdur = float(t.get("duration_s", 0.0))
        if kind != "cut" and tdur >= 0.05:
            segments.append([i])
            transitions.append({"kind": kind, "duration_s": tdur,
                                "direction": t.get("direction", "wipeleft")})
        else:
            segments[-1].append(i)

    seg_labels: list[str] = []
    seg_durations: list[float] = []

    for seg_idx, seg in enumerate(segments):
        if len(seg) == 1:
            label = f"v{seg[0]}"
        else:
            in_labels = "".join(f"[v{i}]" for i in seg)
            raw = f"segraw{seg_idx}"
            label = f"seg{seg_idx}"
            # Concat preserves first-input timebase, NOT AVTB — re-normalize
            # before feeding to xfade. This is the load-bearing line.
            parts.append(
                f"{in_labels}concat=n={len(seg)}:v=1:a=0[{raw}];"
                f"[{raw}]{norm}[{label}]"
            )
        seg_labels.append(label)
        seg_durations.append(sum(durations[i] for i in seg))

    if len(seg_labels) == 1:
        return ";".join(parts) if parts else "", seg_labels[0]

    cum = seg_durations[0]
    last = seg_labels[0]
    for i in range(1, len(seg_labels)):
        t = transitions[i - 1]
        tdur = t["duration_s"]
        xtype = "fade" if t["kind"] == "crossfade" else t.get("direction", "wipeleft")
        offset = max(0.0, cum - tdur)
        out_label = f"x{i}"
        parts.append(
            f"[{last}][{seg_labels[i]}]xfade=transition={xtype}:"
            f"duration={tdur:.4f}:offset={offset:.4f}[{out_label}]"
        )
        last = out_label
        cum = offset + seg_durations[i]

    return ";".join(parts), last
