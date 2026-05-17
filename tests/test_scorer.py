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
