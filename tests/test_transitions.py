"""Unit tests for the transition heuristic + xfade graph builder."""
from __future__ import annotations

import pytest

from aftermovie.render.transitions import (
    build_xfade_graph,
    decide_transitions,
    has_non_cut,
)
from aftermovie.render.pipeline import _compensated_render_entry


def _e(score: float = 1.0) -> dict:
    return {"score": score, "transition_in": {"kind": "cut", "duration_s": 0.0}}


def test_decide_transitions_places_at_least_one_crossfade():
    """At minimum, the heuristic must add some structure (crossfade or whip) when
    given enough cuts to work with."""
    entries = [_e(score=float(i)) for i in range(12)]
    decide_transitions(entries)
    kinds = [e["transition_in"]["kind"] for e in entries]
    assert "crossfade" in kinds or "whip" in kinds
    assert kinds[0] == "cut"  # first entry never gets a transition_in
    assert kinds.count("cut") >= 6  # most entries stay as cuts


def test_decide_transitions_caps_whips():
    entries = [_e(score=float(i)) for i in range(40)]
    decide_transitions(entries)
    whips = [e for e in entries if e["transition_in"]["kind"] == "whip"]
    assert len(whips) <= 3


def test_auto_crossfades_land_on_downbeats():
    """With downbeat data, structural crossfades sit on entries whose cut
    time is a musical downbeat — not on an arbitrary every-4th index."""
    entries = [_e(score=1.0) for _ in range(12)]
    for i, e in enumerate(entries):
        e["beat_time_s"] = float(i)
    song_meta = {"tempo_bpm": 120, "downbeats": [5.0, 9.0]}
    decide_transitions(entries, song_meta)
    fades = [i for i, e in enumerate(entries)
             if e["transition_in"]["kind"] == "crossfade"]
    assert fades == [5, 9]


def test_auto_crossfades_fall_back_without_downbeats():
    """No downbeat data → legacy every-4th-cut placement."""
    # Ascending scores park the whips on 7 and 11, away from the fades.
    entries = [_e(score=float(i)) for i in range(12)]
    decide_transitions(entries)
    fades = [i for i, e in enumerate(entries)
             if e["transition_in"]["kind"] == "crossfade"]
    assert fades == [4, 8]


def test_transition_durations_scale_with_tempo():
    """The same crossfade slot must be shorter on a fast track than a slow
    one: durations are beats, not fixed seconds."""
    slow = [_e(score=float(i)) for i in range(12)]
    fast = [_e(score=float(i)) for i in range(12)]
    decide_transitions(slow, {"tempo_bpm": 80})
    decide_transitions(fast, {"tempo_bpm": 160})
    slow_d = slow[4]["transition_in"]["duration_s"]
    fast_d = fast[4]["transition_in"]["duration_s"]
    assert slow[4]["transition_in"]["kind"] == "crossfade"
    assert fast_d < slow_d
    # 0.4 beat: 80 BPM → 0.3s, 160 BPM → 0.15s.
    assert abs(slow_d - 0.3) < 1e-6
    assert abs(fast_d - 0.15) < 1e-6


def test_transition_clamped_to_fraction_of_slot():
    """A fade may cover at most 40% of the incoming clip's slot, so tight
    drop cuts stay crisp instead of being swallowed by a long dissolve."""
    entries = [_e(score=float(i)) for i in range(12)]
    entries[4]["out_duration_s"] = 0.4  # very tight slot
    decide_transitions(entries, {"tempo_bpm": 60})  # 0.4 beat = 0.4s unclamped
    assert entries[4]["transition_in"]["duration_s"] <= 0.4 * 0.4 + 1e-9


def test_has_non_cut_detects_real_transitions():
    a = [_e(), _e()]
    a[1]["transition_in"] = {"kind": "crossfade", "duration_s": 0.2}
    assert has_non_cut(a)
    b = [_e(), _e()]
    assert not has_non_cut(b)


def test_xfade_graph_uses_concat_for_pure_cut_run():
    entries = [
        {"transition_in": {"kind": "cut", "duration_s": 0.0}},
        {"transition_in": {"kind": "cut", "duration_s": 0.0}},
        {"transition_in": {"kind": "cut", "duration_s": 0.0}},
    ]
    graph, label = build_xfade_graph(3, [1.0, 1.0, 1.0], entries)
    assert "concat=n=3:v=1:a=0" in graph
    assert "xfade" not in graph
    assert label == "seg0"


def test_xfade_graph_splits_on_crossfade():
    entries = [
        {"transition_in": {"kind": "cut", "duration_s": 0.0}},
        {"transition_in": {"kind": "cut", "duration_s": 0.0}},
        {"transition_in": {"kind": "crossfade", "duration_s": 0.2}},
        {"transition_in": {"kind": "cut", "duration_s": 0.0}},
    ]
    graph, label = build_xfade_graph(4, [1.0, 1.0, 1.0, 1.0], entries)
    assert "xfade" in graph
    assert "concat=n=2:v=1:a=0" in graph  # first segment is 2 clips
    # Last segment is 2 clips (indices 2 and 3) — also a concat.
    assert graph.count("concat=") == 2


def test_transition_prerender_duration_compensates_for_overlap():
    """Centered fades: half the entry's own overlap at its head, half the
    NEXT entry's overlap at its tail. Timeline total stays the planner's."""
    entry = {
        "source": "/clip.mp4",
        "start_s": 0.0,
        "end_s": 2.0,
        "out_duration_s": 2.0,
        "transition_in": {"kind": "crossfade", "duration_s": 0.45},
    }
    nxt = {"transition_in": {"kind": "crossfade", "duration_s": 0.3}}

    render_entry, planned, prerender = _compensated_render_entry(
        entry, transitions_active=True, is_first=False, next_entry=nxt,
    )

    assert planned == 2.0
    assert prerender == pytest.approx(2.0 + 0.45 / 2 + 0.3 / 2)
    assert render_entry["out_duration_s"] == prerender
    assert entry["out_duration_s"] == 2.0

    # Last entry: no tail extension.
    _, _, tail = _compensated_render_entry(
        entry, transitions_active=True, is_first=False, next_entry=None,
    )
    assert tail == pytest.approx(2.0 + 0.45 / 2)


def test_centered_fade_midpoint_lands_on_planned_boundary():
    """End-to-end offset math: with half-extended segments, the xfade
    midpoint (offset + tdur/2) sits exactly on the planned cut boundary,
    and the total telescopes back to the planned sum."""
    tdur = 0.4
    entries = [
        {"source": "/a.mp4", "start_s": 0.0, "end_s": 2.0, "out_duration_s": 2.0,
         "transition_in": {"kind": "cut", "duration_s": 0.0}},
        {"source": "/b.mp4", "start_s": 0.0, "end_s": 2.0, "out_duration_s": 2.0,
         "transition_in": {"kind": "crossfade", "duration_s": tdur}},
    ]
    rendered = []
    for i, e in enumerate(entries):
        re_, _, dur = _compensated_render_entry(
            e, transitions_active=True, is_first=(i == 0),
            next_entry=entries[i + 1] if i + 1 < len(entries) else None,
        )
        rendered.append((re_, dur))
    durations = [d for _, d in rendered]
    graph, _ = build_xfade_graph(2, durations, [r for r, _ in rendered])
    import re
    m = re.search(r"xfade=transition=fade:duration=([\d.]+):offset=([\d.]+)", graph)
    assert m, graph
    dur_s, offset = float(m.group(1)), float(m.group(2))
    # Planned boundary is at 2.0; midpoint of the fade must sit there.
    assert offset + dur_s / 2 == pytest.approx(2.0)
    # Total output length telescopes to the planned 4.0.
    assert offset + durations[1] == pytest.approx(4.0)
