"""Unit tests for the transition heuristic + xfade graph builder."""
from __future__ import annotations

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
    entry = {
        "source": "/clip.mp4",
        "start_s": 0.0,
        "end_s": 2.0,
        "out_duration_s": 2.0,
        "transition_in": {"kind": "crossfade", "duration_s": 0.45},
    }

    render_entry, planned, prerender = _compensated_render_entry(
        entry, transitions_active=True, is_first=False,
    )

    assert planned == 2.0
    assert prerender == 2.45
    assert render_entry["out_duration_s"] == 2.45
    assert entry["out_duration_s"] == 2.0
