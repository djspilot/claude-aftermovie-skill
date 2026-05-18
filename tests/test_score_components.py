"""End-to-end smoke tests for the score-component breakdown plumbing.

These tests confirm `components` survives every hop along the scoring
pipeline:

    score_window
        → Candidate (via build_candidates)
        → plan entry dict (via build_plan)
        → plan.json on disk
        → PlanEntry.from_dict (forward and back-compat)
"""
from __future__ import annotations

import json

from aftermovie.score.scorer import build_candidates, build_plan
from aftermovie.types import Candidate, PlanEntry


def _clip(path: str, duration: float = 8.0, **overrides):
    """Mirror of the helper in test_scorer.py — kept local so the two test
    files stay independently editable."""
    n = int(duration)
    base = {
        "path": path,
        "duration_s": duration,
        "fps": 60.0,
        "width": 1920,
        "height": 1080,
        "has_gpmf": False,
        "hilight_tags_ms": [],
        "motion_energy": [0.5] * n,
        "audio_energy": [0.3] * n,
        "accl_peaks": [9.8] * n,
        "gps_speed": [0.0] * n,
        "is_short_form": False,
    }
    base.update(overrides)
    return base


def test_build_candidates_attaches_components():
    """Every Candidate emitted by build_candidates must carry a components
    dict whose values sum to the candidate's score."""
    catalog = {"clips": [
        _clip("/m.mp4", motion_energy=[0.4] * 8, audio_energy=[0.5] * 8,
              hilight_tags_ms=[2500]),
    ]}
    candidates = build_candidates(catalog)
    assert candidates, "build_candidates returned nothing"
    for c in candidates:
        assert isinstance(c, Candidate)
        assert isinstance(c.components, dict)
        # Score must equal sum of component contributions (float epsilon).
        assert abs(sum(c.components.values()) - c.score) < 1e-9, \
            f"candidate {c.source}@{c.start_s} sum(components)={sum(c.components.values())} != score={c.score}"


def test_build_plan_entries_carry_components_dict():
    """build_plan entries must include a `components` key forwarded from the
    winning candidate. The keys must be non-empty when the candidate scored
    above zero on any tracked signal."""
    catalog = {"clips": [
        _clip("/m.mp4", motion_energy=[0.6] * 8, audio_energy=[0.4] * 8,
              hilight_tags_ms=[1500]),
        _clip("/n.mp4", motion_energy=[0.6] * 8, audio_energy=[0.4] * 8),
    ]}
    song = {
        "duration_s": 12.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(24)],
        "downbeats": [i * 2.0 for i in range(6)],
        "intro_end_s": 0.0,
    }
    entries = build_plan(catalog, song, target_len=8.0, no_speed_ramp=True,
                         pace="medium", source_cap=3)
    assert entries, "plan came out empty"
    for entry in entries:
        assert "components" in entry, f"entry missing components: {entry}"
        assert isinstance(entry["components"], dict)
        # sum must match the entry's score within FP epsilon.
        assert abs(sum(entry["components"].values()) - entry["score"]) < 1e-9
    # At least one entry should reflect the hilight_tag we planted.
    assert any("hilight_tag" in e["components"] for e in entries), \
        "expected at least one entry to inherit the hilight_tag component"


def test_plan_json_roundtrip_preserves_components(tmp_path):
    """When the plan dict is dumped to JSON and reloaded via PlanEntry.from_dict,
    the components dict must survive untouched."""
    catalog = {"clips": [
        _clip("/p.mp4", motion_energy=[0.7] * 8, audio_energy=[0.5] * 8,
              hilight_tags_ms=[2500]),
    ]}
    song = {
        "duration_s": 12.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(24)],
        "downbeats": [i * 2.0 for i in range(6)],
        "intro_end_s": 0.0,
    }
    entries = build_plan(catalog, song, target_len=8.0, no_speed_ramp=True,
                         pace="medium", source_cap=3)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"entries": entries}, indent=2))

    raw = json.loads(plan_path.read_text())
    for entry_dict in raw["entries"]:
        assert "components" in entry_dict
        pe = PlanEntry.from_dict(entry_dict)
        assert isinstance(pe.components, dict)
        # The reloaded breakdown must still sum to the score it was written
        # with — guards against from_dict dropping or mistyping the values.
        assert abs(sum(pe.components.values()) - pe.score) < 1e-9


def test_plan_entry_from_dict_tolerates_legacy_plans_without_components():
    """A plan.json written before this field existed (no `components` key)
    must still load — from_dict should default to an empty dict."""
    legacy = {
        "source": "/legacy.mp4",
        "start_s": 0.0,
        "end_s": 2.0,
        "out_duration_s": 2.0,
        "speed": 1.0,
        "beat_time_s": 0.0,
        "score": 3.5,
        "reasons": ["motion_peak"],
        # NOTE: no "components" key — this is the pre-#8 shape.
    }
    pe = PlanEntry.from_dict(legacy)
    assert pe.components == {}
    assert pe.score == 3.5
    assert pe.reasons == ["motion_peak"]


def test_plan_entry_from_dict_coerces_component_values_to_float():
    """JSON numbers sometimes come back as int (e.g. 10 not 10.0). from_dict
    must coerce to float so downstream consumers can rely on the types."""
    entry = {
        "source": "/c.mp4",
        "start_s": 0.0,
        "end_s": 2.0,
        "out_duration_s": 2.0,
        "speed": 1.0,
        "beat_time_s": 0.0,
        "score": 10,
        "reasons": ["hilight_tag"],
        "components": {"hilight_tag": 10, "motion": 1},
    }
    pe = PlanEntry.from_dict(entry)
    assert pe.components == {"hilight_tag": 10.0, "motion": 1.0}
    for v in pe.components.values():
        assert isinstance(v, float)
