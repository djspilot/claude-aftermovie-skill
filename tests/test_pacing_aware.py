"""Tests for Phase C5 pacing-aware allocation.

Two contracts are under test here:

1. `select_cut_points(pace="auto")` consults `song["sections"]` and emits
   visibly higher cut density inside `drop` spans than inside `verse`
   spans (per `SECTION_TO_FACTOR`: drop=1 beat, verse=2 beats).
2. `allocate_candidates(..., sections=...)` biases candidate selection so
   drop sections pull motion- / action-heavy candidates and verses pull
   face-bearing ones. The candidate's *base* `score` is unchanged — only
   the local pick order shifts.

Both tests use a synthetic Song with three hand-placed sections so we can
count cuts-per-second per kind without depending on librosa heuristics.
"""
from __future__ import annotations

from aftermovie.score.scorer import (
    SECTION_BIAS,
    SECTION_TO_FACTOR,
    allocate_candidates,
    build_plan,
    select_cut_points,
)
from aftermovie.types import Candidate


def _clip(path: str, duration: float = 8.0, **overrides):
    """A neutral catalog clip — mirrors `tests/test_scorer.py::_clip`."""
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


def _song_with_sections(duration: float, tempo_bpm: float,
                       sections: list[dict]) -> dict:
    """Build a synthetic song where every beat lands on a 0.5s grid.

    `tempo_bpm=120` → beats every 0.5s, so a 30s song has 60 beats and we
    can hand-allocate dense per-section beat counts without depending on a
    real onset detector."""
    beat_dt = 60.0 / tempo_bpm
    n_beats = int(duration / beat_dt)
    return {
        "duration_s": duration,
        "tempo_bpm": tempo_bpm,
        "beats": [i * beat_dt for i in range(n_beats)],
        "downbeats": [i * (beat_dt * 4) for i in range(n_beats // 4)],
        "intro_end_s": 0.0,
        "energy_per_s": [0.5] * int(duration),
        "sections": sections,
    }


# ---- cut-density tests -----------------------------------------------------

def test_auto_pace_packs_drops_tighter_than_verses():
    """A song with verse → drop → verse must have more cuts/second inside
    the drop span than inside either verse span. The exact ratio depends
    on `SECTION_TO_FACTOR`; we only assert "strictly higher density"."""
    # 30s song, 120 BPM (beats every 0.5s):
    # - verse_a: 0-10s
    # - drop:    10-20s
    # - verse_b: 20-30s
    sections = [
        {"kind": "verse", "start_s": 0.0,  "end_s": 10.0, "intensity": 0.5},
        {"kind": "drop",  "start_s": 10.0, "end_s": 20.0, "intensity": 0.9},
        {"kind": "verse", "start_s": 20.0, "end_s": 30.0, "intensity": 0.5},
    ]
    song = _song_with_sections(30.0, 120.0, sections)
    cuts = select_cut_points(song, target_len=30.0, pace="auto")
    # Strip the appended sentinel.
    cut_times = cuts[:-1]

    def density(start: float, end: float) -> float:
        n = sum(1 for t in cut_times if start <= t < end)
        return n / max(end - start, 1e-6)

    d_verse_a = density(0.0, 10.0)
    d_drop = density(10.0, 20.0)
    d_verse_b = density(20.0, 30.0)
    assert d_drop > d_verse_a, (
        f"drop density ({d_drop:.2f}/s) <= verse_a density "
        f"({d_verse_a:.2f}/s); cut_times={cut_times}"
    )
    assert d_drop > d_verse_b, (
        f"drop density ({d_drop:.2f}/s) <= verse_b density "
        f"({d_verse_b:.2f}/s); cut_times={cut_times}"
    )


def test_section_to_factor_matches_spec():
    """Lock in the documented numbers — changing them in `SECTION_TO_FACTOR`
    forces an update here too, which surfaces the user-visible knob."""
    assert SECTION_TO_FACTOR["intro"] == 4   # every 4 beats
    assert SECTION_TO_FACTOR["verse"] == 2   # every 2 beats
    assert SECTION_TO_FACTOR["build"] == 1   # every beat
    assert SECTION_TO_FACTOR["drop"] == 1    # every beat
    assert SECTION_TO_FACTOR["outro"] == 2   # every 2 beats


def test_intro_section_breathes_compared_to_verse():
    """Intro should be the sparsest body — 4 beats apart vs verse's 2."""
    sections = [
        {"kind": "intro", "start_s": 0.0,  "end_s": 12.0, "intensity": 0.2},
        {"kind": "verse", "start_s": 12.0, "end_s": 24.0, "intensity": 0.5},
    ]
    song = _song_with_sections(24.0, 120.0, sections)
    cuts = select_cut_points(song, target_len=24.0, pace="auto")[:-1]

    def density(start: float, end: float) -> float:
        n = sum(1 for t in cuts if start <= t < end)
        return n / max(end - start, 1e-6)

    assert density(12.0, 24.0) > density(0.0, 12.0), (
        f"intro denser than verse: cuts={cuts}"
    )


# ---- bias tests ------------------------------------------------------------

def _candidate(src: str, **comps) -> Candidate:
    """A Candidate whose `components` dict + `score` are set explicitly so
    we can test the bias logic in isolation."""
    base_score = sum(comps.values())
    return Candidate(
        source=src,
        start_s=0.0,
        end_s=2.0,
        score=base_score,
        reasons=list(comps.keys()),
        src_fps=60.0,
        components=dict(comps),
    )


def test_drop_section_pulls_motion_heavy_candidate_over_face_heavy():
    """Given two candidates with the same base score, the drop bias must
    push the motion-heavy one ahead of the face-heavy one."""
    # Same base score; only the component breakdown differs.
    motion_cand = _candidate("/motion.mp4", motion=2.0, accl_jump=2.0)
    face_cand = _candidate("/face.mp4", face=4.0)
    assert motion_cand.score == face_cand.score, "test setup: same base score"

    sections = [
        {"kind": "drop", "start_s": 0.0, "end_s": 2.0, "intensity": 1.0},
    ]
    cut_points = [0.0, 2.0]  # one slot, one pick
    picks = allocate_candidates(
        [face_cand, motion_cand], cut_points,
        source_cap=1, sections=sections,
    )
    assert len(picks) == 1
    assert picks[0][1].source == "/motion.mp4", (
        f"drop should prefer motion-heavy, got {picks[0][1].source}; "
        f"bias={SECTION_BIAS['drop']}"
    )


def test_verse_section_pulls_face_heavy_candidate_over_motion_heavy():
    """Same setup, opposite preference: verses want faces."""
    motion_cand = _candidate("/motion.mp4", motion=2.0, accl_jump=2.0)
    face_cand = _candidate("/face.mp4", face=4.0)
    assert motion_cand.score == face_cand.score

    sections = [
        {"kind": "verse", "start_s": 0.0, "end_s": 2.0, "intensity": 0.5},
    ]
    cut_points = [0.0, 2.0]
    picks = allocate_candidates(
        [motion_cand, face_cand], cut_points,
        source_cap=1, sections=sections,
    )
    assert len(picks) == 1
    assert picks[0][1].source == "/face.mp4", (
        f"verse should prefer face-heavy, got {picks[0][1].source}; "
        f"bias={SECTION_BIAS['verse']}"
    )


def test_picker_does_not_mutate_candidate_score():
    """The bias is local to the picker — `Candidate.score` and
    `Candidate.components` must be unchanged after `allocate_candidates`."""
    cand = _candidate("/c.mp4", motion=2.0, face=1.0)
    base_score = cand.score
    base_components = dict(cand.components)
    sections = [
        {"kind": "drop", "start_s": 0.0, "end_s": 2.0, "intensity": 1.0},
    ]
    allocate_candidates([cand], [0.0, 2.0],
                        source_cap=1, sections=sections)
    assert cand.score == base_score
    assert cand.components == base_components


# ---- end-to-end build_plan -------------------------------------------------

def test_build_plan_drop_has_higher_motion_than_verse():
    """End-to-end: a catalog with two clearly-tagged groups (motion-heavy
    and face-heavy) plus a 3-section song (verse / drop / verse) must
    produce a plan whose drop entries have higher mean `motion`
    component than its verse entries.

    Setup: 15 motion sources + 5 face sources × 20s. With F3 budget=2
    per source clamped to source_cap=1 → 15 motion + 5 face candidates.
    Total slots = verse_a (5) + drop (10) + verse_b (5) = 20. Motion
    supply (15) covers verse_a + drop; faces fill verse_b. Drop ends up
    100% motion while verses average ~50% motion, so drop > verse.
    """
    catalog = {"clips": []}
    for i in range(15):
        catalog["clips"].append(_clip(
            f"/motion_{i}.mp4", duration=20.0,
            motion_energy=[1.0] * 20,
            accl_peaks=[16.0] * 20,  # triggers high_accel_jump
            audio_energy=[0.0] * 20,
        ))
    for i in range(5):
        catalog["clips"].append(_clip(
            f"/face_{i}.mp4", duration=20.0,
            motion_energy=[0.0] * 20,
            accl_peaks=[9.8] * 20,
            audio_energy=[0.0] * 20,
            face_bboxes=[{"x": 0, "y": 0, "w": 10, "h": 10}] * 20,
        ))

    sections = [
        {"kind": "verse", "start_s": 0.0,  "end_s": 10.0, "intensity": 0.5},
        {"kind": "drop",  "start_s": 10.0, "end_s": 20.0, "intensity": 0.95},
        {"kind": "verse", "start_s": 20.0, "end_s": 30.0, "intensity": 0.5},
    ]
    song = _song_with_sections(30.0, 120.0, sections)

    # `source_cap=5` (user-explicit, not auto-bump — F1's auto-bump ceiling of
    # 3 only applies to `allocate_candidates`' bump-from-1 path). The higher
    # cap gives the motion budget enough headroom that the chronological
    # walker doesn't exhaust motion in verse-1 before the drop section
    # arrives; SECTION_BIAS then has both `motion` and `face` candidates
    # available per slot and can actually steer the pick. The pre-F1 version
    # of this test ran at `cap=2` and relied on subset-mode's old
    # "top-N-by-score" trim to deprive the allocator of face candidates
    # altogether, so verse picks degraded to motion-only by exhaustion — a
    # property the F1 diversity-aware trim deliberately removes.
    plan = build_plan(
        catalog, song, target_len=30.0, no_speed_ramp=True,
        pace="auto", source_cap=1,
        chronological=False,   # keep section ordering preserved
        hook=False, climax=False,
    )
    assert plan, "plan should not be empty"

    def in_span(entry: dict, start: float, end: float) -> bool:
        bt = entry["beat_time_s"]
        return start <= bt < end

    drop_entries = [e for e in plan if in_span(e, 10.0, 20.0)]
    verse_entries = ([e for e in plan if in_span(e, 0.0, 10.0)]
                     + [e for e in plan if in_span(e, 20.0, 30.0)])

    assert drop_entries, f"no entries landed in drop span; plan={plan}"
    assert verse_entries, f"no entries landed in verse spans; plan={plan}"

    def mean_motion(entries):
        if not entries:
            return 0.0
        return sum(e["components"].get("motion", 0.0) for e in entries) / len(entries)

    drop_motion = mean_motion(drop_entries)
    verse_motion = mean_motion(verse_entries)
    assert drop_motion > verse_motion, (
        f"drop mean motion ({drop_motion:.2f}) <= verse mean motion "
        f"({verse_motion:.2f}); drop={drop_entries}, verse={verse_entries}"
    )


def test_build_plan_verse_only_vs_with_drop_cut_count():
    """Two synthetic songs of the same length:
       - verse_only:  one long verse
       - with_drop:   verse → drop → verse
    The with_drop plan should emit MORE cuts (the drop's 1-beat stride
    packs more anchors than the verse's 2-beat stride).
    """
    # 20s, 120 BPM → 40 beats. Verse-only: every 2 beats = 20 cuts max.
    # With-drop: 10s verse (10 cuts max) + 10s drop (20 cuts max) =
    # 30 cuts max — visibly more than the verse-only baseline.
    duration = 20.0
    bpm = 120.0
    catalog = {"clips": [
        _clip(f"/c_{i}.mp4", duration=2.0,
              motion_energy=[0.5, 0.5], audio_energy=[0.3, 0.3])
        for i in range(60)  # plenty of unique sources
    ]}

    verse_only = _song_with_sections(duration, bpm, [
        {"kind": "verse", "start_s": 0.0, "end_s": duration, "intensity": 0.5},
    ])
    with_drop = _song_with_sections(duration, bpm, [
        {"kind": "verse", "start_s": 0.0,  "end_s": 10.0, "intensity": 0.5},
        {"kind": "drop",  "start_s": 10.0, "end_s": 20.0, "intensity": 0.9},
    ])

    plan_v = build_plan(catalog, verse_only, target_len=duration,
                        no_speed_ramp=True, pace="auto", source_cap=1,
                        chronological=False, hook=False, climax=False)
    plan_d = build_plan(catalog, with_drop, target_len=duration,
                        no_speed_ramp=True, pace="auto", source_cap=1,
                        chronological=False, hook=False, climax=False)
    # The plan with a drop must produce strictly more cuts because the
    # drop's stride is half the verse's.
    assert len(plan_d) > len(plan_v), (
        f"with_drop emitted {len(plan_d)} cuts, verse_only emitted "
        f"{len(plan_v)} — expected drop to pack tighter"
    )


def test_manual_pace_modes_ignore_sections():
    """`pace=medium` (and friends) must NOT consult sections — those modes
    are the user's "predictable behavior" escape hatch per the plan."""
    sections = [
        {"kind": "drop", "start_s": 0.0,  "end_s": 10.0, "intensity": 0.95},
        {"kind": "drop", "start_s": 10.0, "end_s": 20.0, "intensity": 0.95},
    ]
    song = _song_with_sections(20.0, 120.0, sections)

    # `medium` = every 2nd beat. With 40 beats over 20s that's exactly 20
    # cuts. If sections leaked in, drop's factor=1 would produce ~40.
    cuts_medium = select_cut_points(song, target_len=20.0, pace="medium")[:-1]
    assert 18 <= len(cuts_medium) <= 22, (
        f"pace=medium emitted {len(cuts_medium)} cuts — section bleed?"
    )
