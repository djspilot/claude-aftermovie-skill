"""xfade-based transitions between cuts.

Three kinds:
    cut       — hard cut, no fade. Uses the concat-demuxer fast path.
    crossfade — xfade=fade, 200ms.
    whip      — xfade=wipeleft / wiperight, 250ms.

When any entry has a non-cut transition_in we switch the whole render to
filter_complex; xfade requires both source streams to be live.
"""
from __future__ import annotations

from typing import Any

XFADE_TYPES = {
    "cut": (None, 0.0),
    "crossfade": ("fade", 0.2),
    "whip": ("wipeleft", 0.25),
    "whip_right": ("wiperight", 0.25),
}


def decide_transitions(entries: list[dict[str, Any]],
                       song_meta: dict[str, Any] | None = None) -> None:
    """Assign a transition_in dict to each entry in-place.

    Heuristic:
      - Every entry defaults to a hard cut.
      - One whip on the highest-scoring entry overall (per 8-cut block, max 3).
      - Crossfade on every 4th entry as a structural beat marker.
    """
    if not entries:
        return
    n = len(entries)
    for e in entries:
        e["transition_in"] = {"kind": "cut", "duration_s": 0.0}

    # Crossfade every 4th cut, skipping the first one.
    for i in range(4, n, 4):
        entries[i]["transition_in"] = {"kind": "crossfade", "duration_s": 0.2}

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
            "kind": "whip", "duration_s": 0.25, "direction": direction,
        }
        whips += 1


def has_non_cut(entries: list[dict[str, Any]]) -> bool:
    return any(e.get("transition_in", {}).get("kind", "cut") != "cut" for e in entries)


def build_xfade_graph(n_inputs: int, durations: list[float],
                      entries: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (filter_complex_string, final_label).

    Algorithm:
        1. Split clips into segments at non-cut transition boundaries.
        2. Concat (filter, not demuxer) within each segment.
        3. xfade between segments.

    `durations` is the per-cut output duration after speed-ramping. Each input
    stream is assumed to be at index 0..n_inputs-1.
    """
    if n_inputs == 1:
        return "", "0:v"

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

    parts: list[str] = []
    seg_labels: list[str] = []
    seg_durations: list[float] = []

    for seg_idx, seg in enumerate(segments):
        if len(seg) == 1:
            label = f"{seg[0]}:v"
        else:
            in_labels = "".join(f"[{i}:v]" for i in seg)
            label = f"seg{seg_idx}"
            parts.append(f"{in_labels}concat=n={len(seg)}:v=1:a=0[{label}]")
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
