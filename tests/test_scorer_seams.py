"""Unit tests for the build_plan seams: select_cut_points / allocate_candidates / decide_speed."""
from __future__ import annotations

from aftermovie.types import Candidate
from aftermovie.score.scorer import (
    allocate_candidates,
    decide_speed,
    select_cut_points,
)


def test_select_cut_points_auto_respects_min_factor():
    """At high BPM with loud energy, pace=auto must never pack cuts tighter
    than 2 beats apart (Quik's anti-strobe floor)."""
    # 180 BPM → beats every 0.333s. With loud energy throughout, _auto_cut_points
    # uses factor=2, so cuts should land on every 2nd beat (~0.667s apart).
    beat_dt = 60.0 / 180.0
    n_beats = 90
    beats = [i * beat_dt for i in range(n_beats)]
    song = {
        "beats": beats,
        "downbeats": beats[::4],
        "intro_end_s": 0.0,
        "energy_per_s": [1.0] * 30,  # loud everywhere
    }
    cuts = select_cut_points(song, target_len=20.0, pace="auto")
    # Strip the appended sentinel (target_len) before checking gaps.
    assert cuts[-1] == 20.0
    real_cuts = cuts[:-1]
    assert len(real_cuts) >= 2
    for a, b in zip(real_cuts, real_cuts[1:]):
        # Two beats apart at 180 BPM = 0.667s. Allow tiny FP slack.
        assert (b - a) >= (2 * beat_dt) - 1e-9, \
            f"cuts {a:.3f} → {b:.3f} are tighter than 2 beats ({2*beat_dt:.3f}s)"


def test_select_cut_points_appends_target_len_sentinel():
    """The terminating sentinel must always be `target_len` so the gap math works."""
    song = {
        "beats": [0.0, 1.0, 2.0, 3.0],
        "downbeats": [0.0, 2.0],
        "intro_end_s": 0.0,
        "energy_per_s": [0.5, 0.5, 0.5, 0.5],
    }
    for pace in ("fast", "medium", "slow", "auto"):
        cuts = select_cut_points(song, target_len=4.0, pace=pace)
        assert cuts[-1] == 4.0, f"pace={pace} missing sentinel"


def test_allocate_candidates_respects_source_cap():
    """The same source path cannot appear more than `source_cap` times in the
    picks, even if every candidate from that source out-scores everything else."""
    # 10 candidates all from /loud.mp4, each at a different start_s so they're
    # distinct entries. Plus a couple from other sources as fallback.
    candidates = [
        Candidate(source="/loud.mp4", start_s=float(i), end_s=float(i) + 2.0,
                  score=100.0 - i, reasons=["motion_peak"], src_fps=60.0)
        for i in range(10)
    ]
    candidates += [
        Candidate(source=f"/quiet{j}.mp4", start_s=0.0, end_s=2.0,
                  score=1.0, reasons=[], src_fps=60.0)
        for j in range(10)
    ]
    # 10 slots each 1s apart, all wide enough to fill.
    cut_points = [float(i) for i in range(11)]
    picks = allocate_candidates(candidates, cut_points, auto_bump_cap=False, source_cap=3)
    counts: dict[str, int] = {}
    for _beat_t, pick in picks:
        counts[pick.source] = counts.get(pick.source, 0) + 1
    assert counts.get("/loud.mp4", 0) <= 3, \
        f"/loud.mp4 picked {counts.get('/loud.mp4', 0)} times — exceeds cap=3"


def test_allocate_candidates_custom_cap():
    """source_cap is honoured as a parameter, not hard-coded."""
    candidates = [
        Candidate(source="/only.mp4", start_s=float(i), end_s=float(i) + 2.0,
                  score=100.0 - i, reasons=[], src_fps=60.0)
        for i in range(10)
    ]
    cut_points = [float(i) for i in range(11)]
    picks_cap1 = allocate_candidates(candidates, cut_points, auto_bump_cap=False, source_cap=1)
    assert len(picks_cap1) == 1
    picks_cap5 = allocate_candidates(
        [Candidate(source="/only.mp4", start_s=float(i), end_s=float(i) + 2.0,
                   score=100.0 - i, reasons=[], src_fps=60.0)
         for i in range(10)],
        cut_points, auto_bump_cap=False, source_cap=5,
    )
    assert len(picks_cap5) == 5


def test_decide_speed_requires_high_fps():
    """Speed ramp only fires when src_fps >= 90, regardless of downbeat + reasons."""
    song = {"downbeats": [0.0, 2.0, 4.0]}
    # 30fps with every other condition satisfied → must stay 1.0.
    low = Candidate(source="/a.mp4", start_s=0.0, end_s=2.0,
                    src_fps=30.0, reasons=["high_accel_jump"])
    assert decide_speed(low, 0.0, song, no_speed_ramp=False) == (1.0, 1.0)
    # Just below threshold.
    sub = Candidate(source="/b.mp4", start_s=0.0, end_s=2.0,
                    src_fps=89.0, reasons=["motion_peak"])
    assert decide_speed(sub, 2.0, song, no_speed_ramp=False) == (1.0, 1.0)
    # Exactly 90fps with all conditions satisfied → slowmo fires.
    high = Candidate(source="/c.mp4", start_s=0.0, end_s=2.0,
                     src_fps=90.0, reasons=["hilight_tag"])
    assert decide_speed(high, 0.0, song, no_speed_ramp=False) == (0.4, 1.0)


def test_decide_speed_requires_downbeat():
    """Even with high fps + action reason, off-downbeat picks stay at 1.0."""
    song = {"downbeats": [0.0, 4.0]}
    high = Candidate(source="/c.mp4", start_s=0.0, end_s=2.0,
                     src_fps=240.0, reasons=["high_accel_jump"])
    # 2.0 is not within 0.05s of any downbeat (0.0 or 4.0).
    assert decide_speed(high, 2.0, song, no_speed_ramp=False) == (1.0, 1.0)


def test_decide_speed_requires_action_reason():
    """High fps + downbeat without an action reason → stays at 1.0."""
    song = {"downbeats": [0.0]}
    boring = Candidate(source="/c.mp4", start_s=0.0, end_s=2.0,
                       src_fps=240.0, reasons=["face_present", "loud_audio"])
    assert decide_speed(boring, 0.0, song, no_speed_ramp=False) == (1.0, 1.0)


def test_decide_speed_no_speed_ramp_flag_forces_one():
    """no_speed_ramp=True short-circuits even when all other conditions hold."""
    song = {"downbeats": [0.0]}
    high = Candidate(source="/c.mp4", start_s=0.0, end_s=2.0,
                     src_fps=240.0, reasons=["high_accel_jump"])
    assert decide_speed(high, 0.0, song, no_speed_ramp=True) == (1.0, 1.0)
