"""Scoring + planning tests with synthetic catalogs (no ffmpeg)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aftermovie.score import scorer as scorer_mod
from aftermovie.score.scorer import (
    allocate_candidates,
    build_candidates,
    build_plan,
    cmd_score,
    score_window,
)
from aftermovie.types import Candidate


def _clip(path: str, duration: float = 8.0, **overrides):
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


def test_hilight_dominates_scoring():
    quiet = _clip("/q.mp4", motion_energy=[0.1] * 8, audio_energy=[0.0] * 8)
    tagged = _clip("/t.mp4", motion_energy=[0.1] * 8, audio_energy=[0.0] * 8,
                   hilight_tags_ms=[2500])
    q_score, q_reasons, _ = score_window(quiet, 2, 4)
    t_score, t_reasons, _ = score_window(tagged, 2, 4)
    assert t_score >= q_score + 10
    assert "hilight_tag" in t_reasons
    assert "hilight_tag" not in q_reasons


def test_audio_peak_rewards_spike_not_steady_loudness():
    """A window that spikes above the clip's own audio baseline (cheer,
    impact) gets the audio_peak bonus; a uniformly loud clip (motor) with
    the same window loudness does not."""
    spike = _clip("/cheer.mp4", audio_energy=[0.2] * 6 + [0.9, 0.9])
    steady = _clip("/motor.mp4", audio_energy=[0.9] * 8)
    s_score, s_reasons, _ = score_window(spike, 6, 8)
    m_score, m_reasons, _ = score_window(steady, 6, 8)
    assert "audio_peak" in s_reasons
    assert "audio_peak" not in m_reasons


def test_gyro_spin_adds_bonus():
    """A fast-rotation gyro peak in the window adds the gyro_spin bonus."""
    spin = _clip("/spin.mp4", gyro_peaks=[0.5, 0.5, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    calm = _clip("/calm.mp4", gyro_peaks=[0.5] * 8)
    sp_score, sp_reasons, _ = score_window(spin, 2, 4)
    ca_score, ca_reasons, _ = score_window(calm, 2, 4)
    assert "gyro_spin" in sp_reasons
    assert "gyro_spin" not in ca_reasons
    assert sp_score == ca_score + 1.5


def test_accel_jump_adds_bonus():
    clip = _clip("/a.mp4", accl_peaks=[10.0, 10.0, 16.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    quiet = _clip("/b.mp4", accl_peaks=[10.0] * 8)
    a_score, a_reasons, _ = score_window(clip, 2, 4)
    q_score, _, _ = score_window(quiet, 2, 4)
    assert "high_accel_jump" in a_reasons
    assert a_score == q_score + 3.0


def test_short_clip_yields_single_candidate():
    catalog = {"clips": [_clip("/short.mp4", duration_s=3.0)]}
    candidates = build_candidates(catalog)
    assert len(candidates) == 1
    assert candidates[0].is_short is False  # 3.0 is short but flag is from is_short_form input


def test_repetition_cap_applied():
    """One high-scoring clip cannot fill more than 3 cuts."""
    catalog = {"clips": [
        _clip("/loud.mp4", duration_s=20.0, hilight_tags_ms=[1000, 5000, 10000, 15000]),
        _clip("/quiet1.mp4", duration_s=20.0),
        _clip("/quiet2.mp4", duration_s=20.0),
        _clip("/quiet3.mp4", duration_s=20.0),
        _clip("/quiet4.mp4", duration_s=20.0),
    ]}
    song = {
        "duration_s": 30.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(60)],
        "downbeats": [i * 2.0 for i in range(15)],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=20.0, no_speed_ramp=False)
    counts: dict[str, int] = {}
    for entry in plan:
        counts[entry["source"]] = counts.get(entry["source"], 0) + 1
    assert counts.get("/loud.mp4", 0) <= 3, "loud clip exceeded repetition cap"


def test_underfilled_plan_distributes_duration_evenly():
    """When there are fewer picks than beat slots, don't park all slack on the last image."""
    catalog = {"clips": [
        _clip("/still1.mp4", duration=2.5, fps=30.0),
        _clip("/still2.mp4", duration=2.5, fps=30.0),
        _clip("/still3.mp4", duration=2.5, fps=30.0),
    ]}
    song = {
        "duration_s": 12.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(24)],
        "downbeats": [i * 2.0 for i in range(6)],
        "intro_end_s": 0.0,
    }

    plan = build_plan(
        catalog, song, target_len=12.0, no_speed_ramp=True,
        pace="fast", source_cap=1,
    )

    assert len(plan) == 3
    assert [round(e["out_duration_s"], 3) for e in plan] == [4.0, 4.0, 4.0]
    assert [round(e["beat_time_s"], 3) for e in plan] == [0.0, 4.0, 8.0]


def test_speed_ramp_fires_when_all_three_conditions_hold():
    """High-fps source + action reason + cut landing on a downbeat → speed=0.5."""
    # Single high-fps source so the picker fills downbeats with it.
    catalog = {"clips": [
        _clip("/fast.mp4", duration_s=20.0, fps=240.0, hilight_tags_ms=[500]),
    ]}
    # Every "beat" IS a downbeat so the very first pick lands on one.
    song = {
        "duration_s": 8.0,
        "tempo_bpm": 60,
        "beats": [0.0, 2.0, 4.0],
        "downbeats": [0.0, 2.0, 4.0],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=6.0, no_speed_ramp=False)
    assert plan, "plan should have at least one entry"
    assert any(e["speed_start"] == 0.4 for e in plan), \
        f"expected at least one slow-mo cut, got speeds {[e['speed'] for e in plan]}"


def test_speed_ramp_suppressed_when_flag_set():
    catalog = {"clips": [
        _clip("/fast.mp4", duration_s=20.0, fps=240.0, hilight_tags_ms=[500]),
    ]}
    song = {
        "duration_s": 8.0,
        "tempo_bpm": 60,
        "beats": [0.0, 2.0, 4.0],
        "downbeats": [0.0, 2.0, 4.0],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=6.0, no_speed_ramp=True)
    assert all(e["speed"] == 1.0 for e in plan), \
        f"--no-speed-ramp should force speed=1.0, got {[e['speed'] for e in plan]}"


def _dup_test_fixtures():
    # Two visually-identical clips (identical phash), one with a HiLight tag
    # so it out-scores its twin by a wide margin. Plus an unrelated clip
    # whose phash is 64 bits away.
    catalog = {"clips": [
        _clip("/twin_lo.mp4", duration=8.0, phash="0" * 16),
        _clip("/twin_hi.mp4", duration=8.0, phash="0" * 16,
              hilight_tags_ms=[2500]),
        _clip("/other.mp4", duration=8.0, phash="f" * 16),
    ]}
    song = {
        "duration_s": 30.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(60)],
        "downbeats": [i * 2.0 for i in range(15)],
        "intro_end_s": 0.0,
    }
    return catalog, song


def test_visual_duplicates_collapse_to_highest_scoring():
    """Two sources whose phashes cluster must yield exactly ONE entry in the
    plan — the higher-scoring one wins, the other is dropped before candidate
    allocation. Sources outside the cluster are untouched. Clustering happens
    at plan time from the catalog phashes (no analyze-time duplicate_group)."""
    catalog, song = _dup_test_fixtures()
    plan = build_plan(catalog, song, target_len=20.0, no_speed_ramp=True,
                      source_cap=5)
    sources = {entry["source"] for entry in plan}
    # The low-scoring twin must be evicted; the high-scoring twin survives.
    assert "/twin_lo.mp4" not in sources
    assert "/twin_hi.mp4" in sources
    # The unrelated clip isn't part of the cluster, so it stays available.
    assert "/other.mp4" in sources


def test_visual_dup_threshold_zero_disables_filter():
    """`visual_dup_threshold=0` must keep both twins in the pool."""
    catalog, song = _dup_test_fixtures()
    plan = build_plan(catalog, song, target_len=20.0, no_speed_ramp=True,
                      source_cap=5, visual_dup_threshold=0)
    sources = {entry["source"] for entry in plan}
    assert "/twin_lo.mp4" in sources
    assert "/twin_hi.mp4" in sources


def test_semantic_duplicates_collapse_and_respect_off_switch():
    """Sources whose embeddings are near-parallel (same scene, different
    angle — phashes far apart) collapse to the best one; setting
    visual_dup_threshold=0 keeps them all. Clips without embeddings pass."""
    import math
    a = [1.0, 0.0, 0.0]
    b = [math.cos(0.2), math.sin(0.2), 0.0]   # cosine ≈ 0.98 with a
    c = [0.0, 0.0, 1.0]                        # orthogonal
    catalog = {"clips": [
        _clip("/scene1_lo.mp4", duration=8.0, embedding=a,
              phash="0" * 16),
        _clip("/scene1_hi.mp4", duration=8.0, embedding=b,
              phash="f" * 16, hilight_tags_ms=[2500]),
        _clip("/other.mp4", duration=8.0, embedding=c,
              phash="00ff00ff00ff00ff"),
    ]}
    song = {
        "duration_s": 30.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(60)],
        "downbeats": [i * 2.0 for i in range(15)],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=20.0, no_speed_ramp=True,
                      source_cap=1)
    sources = {e["source"] for e in plan}
    assert "/scene1_lo.mp4" not in sources
    assert "/scene1_hi.mp4" in sources
    assert "/other.mp4" in sources

    off = build_plan(catalog, song, target_len=20.0, no_speed_ramp=True,
                     source_cap=1, visual_dup_threshold=0)
    off_sources = {e["source"] for e in off}
    assert {"/scene1_lo.mp4", "/scene1_hi.mp4"} <= off_sources


def test_cosine_dot_product_and_mismatch():
    from aftermovie.analyze.embedding import cosine
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0], [1.0, 0.0]) == 0.0  # length mismatch → dissimilar


def test_stabilize_flags_shaky_windows_but_not_spins():
    """With stabilize=True, sustained-high gyro flags the entry; a
    deliberate spin (gyro_spin reason) and calm footage do not. Default off."""
    catalog = {"clips": [
        _clip("/shaky.mp4", duration=8.0, gyro_peaks=[1.5] * 8),
        _clip("/spin.mp4", duration=8.0,
              gyro_peaks=[0.5, 0.5, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
        _clip("/calm.mp4", duration=8.0, gyro_peaks=[0.3] * 8),
    ]}
    song = {
        "duration_s": 30.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(60)],
        "downbeats": [i * 2.0 for i in range(15)],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=20.0, no_speed_ramp=True,
                      source_cap=1, stabilize=True)
    by_src = {e["source"]: e for e in plan}
    assert by_src["/shaky.mp4"]["stabilize"] is True
    assert by_src["/calm.mp4"]["stabilize"] is False
    assert by_src["/spin.mp4"]["stabilize"] is False  # gyro_spin reason wins

    default_plan = build_plan(catalog, song, target_len=20.0,
                              no_speed_ramp=True, source_cap=1)
    assert all(e["stabilize"] is False for e in default_plan)


def test_build_section_gets_speed_lift_and_last_entry_fades_out():
    """Cuts inside a `build` section run 1.0→1.15x; the final plan entry
    carries a fade_out_s stamp scaled to its slot."""
    catalog = {"clips": [
        _clip(f"/c{i}.mp4", duration=8.0) for i in range(4)
    ]}
    song = {
        "duration_s": 16.0,
        "tempo_bpm": 120,
        "beats": [i * 2.0 for i in range(8)],
        "downbeats": [0.0, 8.0],
        "intro_end_s": 0.0,
        "sections": [
            {"kind": "verse", "start_s": 0.0, "end_s": 4.0},
            {"kind": "build", "start_s": 4.0, "end_s": 16.0},
        ],
    }
    plan = build_plan(catalog, song, target_len=8.0, no_speed_ramp=False,
                      source_cap=1, stretch_stills=False)
    in_build = [e for e in plan if e["beat_time_s"] >= 4.0]
    assert in_build, plan
    assert all(e["speed_end"] == 1.15 for e in in_build)
    assert plan[-1]["fade_out_s"] > 0.05
    assert all("fade_out_s" not in e for e in plan[:-1])


def test_hilight_tag_recentered_at_40pct_of_slot():
    """A pick with a HiLight tag re-anchors its source window so the tag
    sits ~40% into the slot instead of wherever the integer-second window
    happened to put it."""
    catalog = {"clips": [
        _clip("/tagged.mp4", duration=20.0, hilight_tags_ms=[10_000]),
    ]}
    song = {
        "duration_s": 10.0,
        "tempo_bpm": 120,
        "beats": [0.0, 2.0, 4.0, 6.0, 8.0],
        "downbeats": [0.0, 4.0, 8.0],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=4.0, no_speed_ramp=True,
                      source_cap=1, stretch_stills=False)
    entry = next(e for e in plan if "hilight_tag" in e["reasons"])
    slot = entry["end_s"] - entry["start_s"]
    tag_pos = (10.0 - entry["start_s"]) / slot
    assert 0.3 < tag_pos < 0.5, \
        f"tag at {tag_pos:.2f} of slot, expected ~0.4 (entry={entry})"


def test_luma_offset_nudges_outliers_toward_catalog_median():
    """A clip much darker than the catalog median gets a positive brightness
    offset (clamped ±0.08); a clip at the median gets ~0."""
    catalog = {"clips": [
        _clip("/dark.mp4", duration=8.0, exposure_per_s=[0.2] * 8),
        _clip("/mid1.mp4", duration=8.0, exposure_per_s=[0.5] * 8),
        _clip("/mid2.mp4", duration=8.0, exposure_per_s=[0.5] * 8),
    ]}
    song = {
        "duration_s": 30.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(60)],
        "downbeats": [i * 2.0 for i in range(15)],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=20.0, no_speed_ramp=True,
                      source_cap=1)
    by_src = {e["source"]: e for e in plan}
    # dark: (0.5 - 0.2) * 0.5 = 0.15 → clamped to 0.08.
    assert by_src["/dark.mp4"]["luma_offset"] == 0.08
    assert by_src["/mid1.mp4"]["luma_offset"] == 0.0


def test_strict_chronological_disables_trailer_arc():
    """With hook/climax off, entries follow capture time exactly; with the
    default trailer arc on, the best pick is hoisted to the front."""
    clips = []
    for i in range(10):
        kw = {"captured_at": 1000.0 + i}
        if i == 6:  # chronologically mid-late clip out-scores everything
            kw["hilight_tags_ms"] = [1000]
        clips.append(_clip(f"/c{i}.mp4", duration=8.0, **kw))
    catalog = {"clips": clips}
    song = {
        "duration_s": 40.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(80)],
        "downbeats": [i * 2.0 for i in range(20)],
        "intro_end_s": 0.0,
    }
    arc = build_plan(catalog, song, target_len=30.0, no_speed_ramp=True,
                     source_cap=1)
    assert arc[0]["source"] == "/c6.mp4"  # hook hoists the best shot

    strict = build_plan(catalog, song, target_len=30.0, no_speed_ramp=True,
                        source_cap=1, hook=False, climax=False)
    order = [e["source"] for e in strict]
    assert order == sorted(order, key=lambda s: int(s[2:-4])), \
        f"strict plan not in capture order: {order}"


def test_low_fps_never_gets_speed_ramp():
    """A 30fps source must never be slowed even on downbeat + action reasons."""
    catalog = {"clips": [
        _clip("/slow.mp4", duration_s=20.0, fps=30.0, hilight_tags_ms=[500]),
    ]}
    song = {
        "duration_s": 8.0,
        "tempo_bpm": 60,
        "beats": [0.0, 2.0, 4.0],
        "downbeats": [0.0, 2.0, 4.0],
        "intro_end_s": 0.0,
    }
    plan = build_plan(catalog, song, target_len=6.0, no_speed_ramp=False)
    assert all(e["speed"] == 1.0 for e in plan)


def test_blurry_window_is_penalized_and_tagged():
    """Bottom-third sharpness in the window must subtract score + emit 'blurry'."""
    # Mostly sharp, but seconds 2-3 are clearly the softest in the clip.
    clip = _clip(
        "/c.mp4",
        sharpness_per_s=[0.9, 0.8, 0.1, 0.1, 0.85, 0.95, 0.9, 0.88],
    )
    baseline = _clip("/c.mp4")  # no sharpness_per_s → no penalty
    blurry_score, blurry_reasons, _ = score_window(clip, 2, 4)
    base_score, base_reasons, _ = score_window(baseline, 2, 4)
    assert "blurry" in blurry_reasons
    assert "blurry" not in base_reasons
    assert blurry_score < base_score


def test_over_bright_window_is_penalized_and_tagged():
    """Mean exposure > 0.85 over the window adds 'poor_exposure' and drops score."""
    clip = _clip(
        "/c.mp4",
        exposure_per_s=[0.5, 0.5, 0.95, 0.92, 0.5, 0.5, 0.5, 0.5],
    )
    baseline = _clip("/c.mp4")
    bright_score, bright_reasons, _ = score_window(clip, 2, 4)
    base_score, _, _ = score_window(baseline, 2, 4)
    assert "poor_exposure" in bright_reasons
    assert bright_score < base_score


def test_under_exposed_window_is_penalized():
    """Mean exposure < 0.25 over the window adds 'poor_exposure' and drops score."""
    clip = _clip(
        "/c.mp4",
        exposure_per_s=[0.5, 0.5, 0.10, 0.08, 0.5, 0.5, 0.5, 0.5],
    )
    baseline = _clip("/c.mp4")
    dark_score, dark_reasons, _ = score_window(clip, 2, 4)
    base_score, _, _ = score_window(baseline, 2, 4)
    assert "poor_exposure" in dark_reasons
    assert dark_score < base_score


def test_well_exposed_midtones_not_penalized():
    """Mean exposure inside [0.25, 0.85] must NOT trigger 'poor_exposure'."""
    clip = _clip("/c.mp4", exposure_per_s=[0.4, 0.45, 0.5, 0.55, 0.5, 0.5, 0.5, 0.5])
    _, reasons, _ = score_window(clip, 2, 4)
    assert "poor_exposure" not in reasons


def test_missing_quality_lists_skip_penalties():
    """Clips analyzed before cv2 was installed have empty quality lists —
    they must not be penalized."""
    clip = _clip("/c.mp4")  # no sharpness_per_s / exposure_per_s overrides
    _, reasons, _ = score_window(clip, 2, 4)
    assert "blurry" not in reasons
    assert "poor_exposure" not in reasons


def test_banned_source_is_dropped_from_candidates():
    """A path listed in `preferences['banned']` produces zero Candidates."""
    catalog = {"clips": [
        _clip("/kept.mp4", duration_s=8.0),
        _clip("/banned.mp4", duration_s=8.0),
    ]}
    candidates = build_candidates(
        catalog,
        preferences={"banned": ["/banned.mp4"]},
    )
    sources = {c.source for c in candidates}
    assert "/banned.mp4" not in sources, "banned source leaked into candidate pool"
    assert "/kept.mp4" in sources


def test_favorited_source_gets_boost_and_reason():
    """Favorited sources gain a flat +2.0 score and a 'user_favorite' reason."""
    catalog = {"clips": [_clip("/fav.mp4", duration_s=8.0)]}
    baseline = build_candidates(catalog)
    boosted = build_candidates(catalog, preferences={"favorited": ["/fav.mp4"]})
    # Same shape, same source — only the score and reasons differ.
    assert len(baseline) == len(boosted)
    by_window = {(c.start_s, c.end_s): c for c in baseline}
    for c in boosted:
        match = by_window[(c.start_s, c.end_s)]
        assert c.score == match.score + 2.0, (
            f"expected +2.0 boost, got {c.score} vs baseline {match.score}"
        )
        assert "user_favorite" in c.reasons


def test_build_plan_drops_banned_and_boosts_favorited():
    """End-to-end: banned source absent from plan; favorited carries its tag."""
    catalog = {"clips": [
        _clip("/fav.mp4", duration_s=8.0),
        _clip("/banned.mp4", duration_s=8.0, hilight_tags_ms=[1000]),
        _clip("/neutral.mp4", duration_s=8.0),
    ]}
    song = {
        "duration_s": 20.0,
        "tempo_bpm": 120,
        "beats": [i * 0.5 for i in range(40)],
        "downbeats": [i * 2.0 for i in range(10)],
        "intro_end_s": 0.0,
    }
    plan = build_plan(
        catalog, song, target_len=12.0, no_speed_ramp=True,
        source_cap=3,
        preferences={"favorited": ["/fav.mp4"], "banned": ["/banned.mp4"]},
    )
    sources = {e["source"] for e in plan}
    # The banned clip is gone even though it had the highest objective score.
    assert "/banned.mp4" not in sources
    # The favorited clip is present and its entries carry the user_favorite tag.
    fav_entries = [e for e in plan if e["source"] == "/fav.mp4"]
    assert fav_entries, "favorited clip should appear in the plan"
    assert all("user_favorite" in e["reasons"] for e in fav_entries)


def test_components_break_down_mixed_signals():
    """A window with motion + audio + face must populate at least those three
    component keys, all positive. Zero-valued signals must be absent."""
    clip = _clip(
        "/m.mp4",
        motion_energy=[0.4] * 8,
        audio_energy=[0.5] * 8,
        # Face boxes per second; non-None entries within the window light up
        # the "face" contribution.
        face_bboxes=[None, None, {"x": 0, "y": 0, "w": 10, "h": 10},
                     {"x": 0, "y": 0, "w": 10, "h": 10}, None, None, None, None],
        # No accel/GPS/HiLight tags so those component keys must be absent.
        accl_peaks=[9.0] * 8,
        gps_speed=[0.0] * 8,
    )
    _, _, components = score_window(clip, 2, 4)
    assert components["motion"] > 0
    assert components["audio"] > 0
    assert components["face"] > 0
    # Signals that didn't fire must not appear as zero entries.
    assert "accl_jump" not in components
    assert "gps_speed" not in components
    assert "hilight_tag" not in components
    assert "blurry" not in components
    assert "poor_exposure" not in components


def test_components_sum_matches_score():
    """sum(components.values()) must equal the returned score within float
    epsilon for every combination of signals — this is the invariant the
    debug-gated assert in score_window guards."""
    clip = _clip(
        "/all.mp4",
        motion_energy=[0.8] * 8,
        audio_energy=[0.9] * 8,
        accl_peaks=[10.0, 10.0, 18.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        gps_speed=[5.0, 5.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        hilight_tags_ms=[2500],
        face_bboxes=[None, None, {"x": 0, "y": 0, "w": 10, "h": 10},
                     {"x": 0, "y": 0, "w": 10, "h": 10}, None, None, None, None],
    )
    score, _, components = score_window(clip, 2, 4)
    assert abs(sum(components.values()) - score) < 1e-9, \
        f"sum({components}) != score={score}"


def test_purely_blurry_window_has_only_negative_blurry_component():
    """A window with no positive signals but bottom-third sharpness must
    yield ONLY the negative `blurry` component — no positive contributions,
    no other penalties."""
    clip = _clip(
        "/blur.mp4",
        # No motion, no audio, no accel above gravity, no GPS, no faces.
        motion_energy=[0.0] * 8,
        audio_energy=[0.0] * 8,
        accl_peaks=[9.8] * 8,
        gps_speed=[0.0] * 8,
        face_bboxes=[None] * 8,
        # Seconds 2-3 are clearly the softest — bottom 30th percentile.
        sharpness_per_s=[0.9, 0.8, 0.1, 0.1, 0.85, 0.95, 0.9, 0.88],
    )
    score, reasons, components = score_window(clip, 2, 4)
    assert "blurry" in reasons
    # The ONLY component should be the negative blurry penalty.
    assert components == {"blurry": -1.5}
    assert score == -1.5


# ---- F1: tightened auto-bump cap + slight-underfill acceptance -------------


def test_auto_bump_cap_ceiling_is_three(capsys: pytest.CaptureFixture[str]):
    """The auto-bump in `allocate_candidates` must never raise the cap above
    3 — same-clip 4× is jarring enough that viewers prefer a slightly
    shorter edit. The math used to allow a bump to 5 when slots / sources
    demanded it; F1 caps that at 3 and logs the new ceiling."""
    # 5 sources, 30 slots → math wants 30/5=6, F1 caps to 3.
    candidates = [
        Candidate(source=f"/src{s}.mp4", start_s=float(i), end_s=float(i) + 2.0,
                  score=100.0 - i - s, reasons=[], src_fps=60.0)
        for s in range(5) for i in range(10)
    ]
    cut_points = [float(i) for i in range(31)]  # 30 slots
    picks = allocate_candidates(candidates, cut_points, source_cap=1)
    counts: dict[str, int] = {}
    for _bt, p in picks:
        counts[p.source] = counts.get(p.source, 0) + 1
    # No source can appear more than 3 times under the new cap.
    assert max(counts.values()) <= 3, \
        f"auto-bump exceeded F1 ceiling of 3: {counts}"
    err = capsys.readouterr().err
    # New log phrasing: "bumped source_cap 1 → 3"; the old "auto-bumped … → 5"
    # is no longer allowed.
    assert "bumped source_cap 1 → 3" in err, \
        f"expected 'bumped source_cap 1 → 3' in stderr, got: {err!r}"
    assert "→ 5" not in err, \
        f"auto-bump leaked past F1 ceiling of 3: {err!r}"


def test_slight_underfill_under_20pct_skips_bump(
        capsys: pytest.CaptureFixture[str]):
    """When the no-bump plan would be only slightly short of the slot
    count, prefer the shorter plan to a clip-repeat plan. The threshold
    is 20% — at 18% short we accept, at 25% short we bump."""
    # 9 unique sources, 10 slots — without a bump, 1 slot stays unfilled
    # (10% underfill, well under 20%). With a bump the 10th slot would
    # repeat one of the sources. F1 says: accept the 9-cut plan.
    candidates = [
        Candidate(source=f"/src{s}.mp4", start_s=0.0, end_s=2.0,
                  score=100.0 - s, reasons=[], src_fps=60.0)
        for s in range(9)
    ]
    cut_points = [float(i) for i in range(11)]  # 10 slots
    picks = allocate_candidates(candidates, cut_points, source_cap=1)
    counts: dict[str, int] = {}
    for _bt, p in picks:
        counts[p.source] = counts.get(p.source, 0) + 1
    # No source repeated — bump was suppressed.
    assert max(counts.values()) == 1, \
        f"expected no repeats (bump suppressed), got {counts}"
    # And exactly 9 picks landed (1 slot intentionally empty).
    assert len(picks) == 9
    err = capsys.readouterr().err
    assert "accepted slight underfill" in err, \
        f"expected 'accepted slight underfill' log, got: {err!r}"
    # And the cap-bump line MUST be absent.
    assert "bumped source_cap" not in err, \
        f"bump fired when it shouldn't have: {err!r}"


def test_large_underfill_over_20pct_triggers_bump(
        capsys: pytest.CaptureFixture[str]):
    """A 60%-short plan is still a bump candidate — F1 only suppresses
    SLIGHT underfill. Mirrors the real-session log line we want for big
    deficits."""
    # 4 unique sources, 10 slots → 60% underfill → must bump to cap=3
    # (capped per F1), landing 4*3=12 ≥ 10 slots so the plan fills.
    candidates = [
        Candidate(source=f"/src{s}.mp4", start_s=float(i), end_s=float(i) + 2.0,
                  score=100.0 - i - s, reasons=[], src_fps=60.0)
        for s in range(4) for i in range(5)
    ]
    cut_points = [float(i) for i in range(11)]
    picks = allocate_candidates(candidates, cut_points, source_cap=1)
    err = capsys.readouterr().err
    assert "bumped source_cap 1 → 3" in err, \
        f"expected bump-to-3 log, got: {err!r}"
    assert "accepted slight underfill" not in err, \
        f"underfill suppression fired when it shouldn't have: {err!r}"
    # Cap=3 actually fills the 10 slots.
    counts: dict[str, int] = {}
    for _bt, p in picks:
        counts[p.source] = counts.get(p.source, 0) + 1
    assert max(counts.values()) <= 3
    assert len(picks) == 10


def test_diversity_log_line_emitted(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]):
    """`cmd_score` must emit a one-line diversity readout after the 'Built N
    cuts' summary so users (and the user-facing CLI) can see at a glance
    how varied the plan is."""
    catalog = {"clips": [_clip(f"/c{i}.mp4", duration=8.0) for i in range(10)]}
    beat_dt = 0.5  # 120 BPM
    n_beats = 40
    song = {
        "duration_s": 20.0,
        "tempo_bpm": 120.0,
        "beats": [i * beat_dt for i in range(n_beats)],
        "downbeats": [i * 2.0 for i in range(10)],
        "intro_end_s": 0.0,
        "energy_per_s": [0.5] * 20,
    }
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(catalog))
    plan_path = tmp_path / "plan.json"
    fake_song = tmp_path / "song.mp3"
    fake_song.write_bytes(b"")

    monkeypatch.setattr(scorer_mod, "analyze_song", lambda _p: song)
    args = argparse.Namespace(
        catalog=str(cat_path),
        song=str(fake_song),
        out=str(plan_path),
        max_length=None,
        aspect="16:9",
        res="1920x1080",
        fps=30,
        lut=None,
        music_db=-8.0,
        clip_db=-18.0,
        no_speed_ramp=True,
        audio_mix="ducked",
        pace="medium",
        transitions="cut",
        titles=None,
        title_text=None,
        no_reframe=False,
        source_cap=3,
        chronological=False,
        burst_window_s=0.0,
    )
    cmd_score(args)
    err = capsys.readouterr().err
    assert "diversity:" in err, \
        f"expected diversity log line, got stderr: {err!r}"
    # The line must contain BOTH counts AND the avg/max repeat readout.
    import re
    match = re.search(
        r"diversity: (\d+) cuts from (\d+) unique sources "
        r"\(avg ([\d.]+) repeats, max (\d+)\)",
        err,
    )
    assert match, f"diversity log line format unexpected: {err!r}"
    n_cuts, n_unique, avg_repeats, max_repeats = match.groups()
    assert int(n_cuts) > 0
    assert int(n_unique) > 0
    # Math sanity: avg = cuts / unique.
    assert abs(float(avg_repeats) - int(n_cuts) / int(n_unique)) < 0.05
    assert int(max_repeats) >= 1


