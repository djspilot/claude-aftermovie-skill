"""Scoring + planning tests with synthetic catalogs (no ffmpeg)."""
from __future__ import annotations

from aftermovie.score.scorer import build_candidates, build_plan, score_window


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
    q_score, q_reasons = score_window(quiet, 2, 4)
    t_score, t_reasons = score_window(tagged, 2, 4)
    assert t_score >= q_score + 10
    assert "hilight_tag" in t_reasons
    assert "hilight_tag" not in q_reasons


def test_accel_jump_adds_bonus():
    clip = _clip("/a.mp4", accl_peaks=[10.0, 10.0, 16.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    quiet = _clip("/b.mp4", accl_peaks=[10.0] * 8)
    a_score, a_reasons = score_window(clip, 2, 4)
    q_score, _ = score_window(quiet, 2, 4)
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
    blurry_score, blurry_reasons = score_window(clip, 2, 4)
    base_score, base_reasons = score_window(baseline, 2, 4)
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
    bright_score, bright_reasons = score_window(clip, 2, 4)
    base_score, _ = score_window(baseline, 2, 4)
    assert "poor_exposure" in bright_reasons
    assert bright_score < base_score


def test_under_exposed_window_is_penalized():
    """Mean exposure < 0.25 over the window adds 'poor_exposure' and drops score."""
    clip = _clip(
        "/c.mp4",
        exposure_per_s=[0.5, 0.5, 0.10, 0.08, 0.5, 0.5, 0.5, 0.5],
    )
    baseline = _clip("/c.mp4")
    dark_score, dark_reasons = score_window(clip, 2, 4)
    base_score, _ = score_window(baseline, 2, 4)
    assert "poor_exposure" in dark_reasons
    assert dark_score < base_score


def test_well_exposed_midtones_not_penalized():
    """Mean exposure inside [0.25, 0.85] must NOT trigger 'poor_exposure'."""
    clip = _clip("/c.mp4", exposure_per_s=[0.4, 0.45, 0.5, 0.55, 0.5, 0.5, 0.5, 0.5])
    _, reasons = score_window(clip, 2, 4)
    assert "poor_exposure" not in reasons


def test_missing_quality_lists_skip_penalties():
    """Clips analyzed before cv2 was installed have empty quality lists —
    they must not be penalized."""
    clip = _clip("/c.mp4")  # no sharpness_per_s / exposure_per_s overrides
    _, reasons = score_window(clip, 2, 4)
    assert "blurry" not in reasons
    assert "poor_exposure" not in reasons
