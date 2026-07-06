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


def test_select_cut_points_snaps_to_nearby_onsets():
    """Cuts land on the actual audio onset when one sits within ±80ms of the
    beat; far-away onsets don't pull, and order stays ascending."""
    song = {
        "beats": [0.0, 1.0, 2.0, 3.0],
        "intro_end_s": 0.0,
        # 1.05 pulls the beat at 1.0; 2.5 is too far from any beat.
        "onset_peaks": [1.05, 2.5],
    }
    cuts = select_cut_points(song, target_len=4.0, pace="fast")
    assert 1.05 in cuts
    assert 1.0 not in cuts
    assert 2.0 in cuts  # unpulled
    assert cuts == sorted(cuts)


def test_snap_rejected_when_it_crowds_previous_cut():
    """A snap that would push two cuts closer than the anti-strobe gap keeps
    the original beat time instead."""
    from aftermovie.score.scorer import _snap_to_onsets
    # Cuts 0.30s apart; onset pulls the second one backwards toward the first.
    snapped = _snap_to_onsets([1.0, 1.30], [1.24], tol=0.08, min_gap=0.25)
    assert snapped == [1.0, 1.30]


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


def test_diversity_penalty_prefers_visually_fresh_candidate():
    """When two candidates tie-ish on score but one looks like an already
    picked source (phash within DIVERSITY_SIMILAR_BITS, above the hard-dedup
    threshold), the visually fresh one wins the slot."""
    from aftermovie.score.scorer import allocate_candidates

    def cand(src: str, score: float) -> Candidate:
        return Candidate(source=src, start_s=0.0, end_s=2.0,
                         score=score, reasons=[], src_fps=30.0)

    # `similar` is 12 bits from `first` (past dedup's 8, inside diversity's
    # 16); `fresh` is ~32 bits away. `similar` out-scores `fresh` by 1.0,
    # less than the 2.0 penalty → fresh must win slot 2.
    sigs = {
        "/first.mp4":   "0000000000000000",
        "/similar.mp4": "0fff000000000000",  # 12 bits
        "/fresh.mp4":   "00000000ffffffff",  # 32 bits
    }
    picks = allocate_candidates(
        [cand("/first.mp4", 10.0), cand("/similar.mp4", 5.0),
         cand("/fresh.mp4", 4.0)],
        [0.0, 2.0, 4.0], source_cap=1, auto_bump_cap=False,
        source_phash=sigs,
    )
    assert [p.source for _, p in picks] == ["/first.mp4", "/fresh.mp4"]

    # Without signatures the higher base score wins as before.
    picks = allocate_candidates(
        [cand("/first.mp4", 10.0), cand("/similar.mp4", 5.0),
         cand("/fresh.mp4", 4.0)],
        [0.0, 2.0, 4.0], source_cap=1, auto_bump_cap=False,
    )
    assert [p.source for _, p in picks] == ["/first.mp4", "/similar.mp4"]


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


def test_decide_speed_drop_slam_ignores_reasons():
    """The first cut of a `drop` section ramps even without action reasons
    or a downbeat — the drop itself is the event. High-fps gate still holds."""
    song = {
        "downbeats": [0.0],
        "sections": [
            {"kind": "verse", "start_s": 0.0, "end_s": 8.0},
            {"kind": "drop", "start_s": 8.0, "end_s": 16.0},
        ],
    }
    boring = Candidate(source="/c.mp4", start_s=0.0, end_s=2.0,
                       src_fps=240.0, reasons=["face_present"])
    # First beat inside the drop → slam.
    assert decide_speed(boring, 8.2, song, no_speed_ramp=False) == (0.4, 1.0)
    # Later beat in the same drop → no slam (and no downbeat/reason → flat).
    assert decide_speed(boring, 12.0, song, no_speed_ramp=False) == (1.0, 1.0)
    # Low fps never slams.
    low = Candidate(source="/d.mp4", start_s=0.0, end_s=2.0,
                    src_fps=30.0, reasons=["face_present"])
    assert decide_speed(low, 8.2, song, no_speed_ramp=False) == (1.0, 1.0)


def test_decide_speed_no_speed_ramp_flag_forces_one():
    """no_speed_ramp=True short-circuits even when all other conditions hold."""
    song = {"downbeats": [0.0]}
    high = Candidate(source="/c.mp4", start_s=0.0, end_s=2.0,
                     src_fps=240.0, reasons=["high_accel_jump"])
    assert decide_speed(high, 0.0, song, no_speed_ramp=True) == (1.0, 1.0)
