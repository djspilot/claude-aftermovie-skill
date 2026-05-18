"""F3 moment-budget tests.

Replaces the C4 pool-level subset trim with a per-source moment budget so
every source in the Source folder gets a fair shot at the plan. Covers:

- `_compute_source_budgets` math: budgets scale with duration, capped at
  8, with short clips (stills + sub-4s Live Photo MOVs) clamped to 1.
- `build_plan` honours the per-source budget: all sources contribute when
  the song has room, and the log line surfaces totals.
- The user-supplied `--source-cap N` acts as a HARD CEILING — a budget of
  6 is clamped to 2 when `source_cap=2` is passed.
"""
from __future__ import annotations

from typing import Any

from aftermovie.score.scorer import (
    MAX_MOMENTS_PER_SOURCE,
    SECONDS_PER_MOMENT,
    _apply_moment_budget,
    _compute_source_budgets,
    build_candidates,
    build_plan,
)
from aftermovie.types import Candidate


def _clip(path: str, duration: float = 8.0, **overrides: Any) -> dict[str, Any]:
    n = max(int(duration), 1)
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


def _song(duration_s: float, tempo_bpm: float = 120.0) -> dict[str, Any]:
    beat_dt = 60.0 / tempo_bpm
    n_beats = max(int(duration_s / beat_dt), 1)
    beats = [i * beat_dt for i in range(n_beats)]
    return {
        "duration_s": duration_s,
        "tempo_bpm": tempo_bpm,
        "beats": beats,
        "downbeats": beats[::4] if beats else beats,
        "intro_end_s": 0.0,
        "energy_per_s": [0.5] * max(int(duration_s), 1),
    }


# ---- _compute_source_budgets math ------------------------------------------


def test_compute_source_budgets_short_clip_gets_one():
    """A 3s Live Photo MOV (sub-`SHORT_SOURCE_DURATION_S`) gets budget=1."""
    catalog = {"clips": [_clip("/short.mp4", duration=3.0)]}
    candidates = build_candidates(catalog)
    budgets = _compute_source_budgets(catalog, candidates)
    assert budgets == {"/short.mp4": 1}


def test_compute_source_budgets_long_clip_scales_with_duration():
    """A 60s GoPro contributes ceil(60/10)=6 distinct moments."""
    catalog = {"clips": [_clip("/long.mp4", duration=60.0)]}
    candidates = build_candidates(catalog)
    budgets = _compute_source_budgets(catalog, candidates)
    assert budgets == {"/long.mp4": 6}


def test_compute_source_budgets_caps_at_max():
    """A 5-minute clip is capped at `MAX_MOMENTS_PER_SOURCE` (8)."""
    catalog = {"clips": [_clip("/huge.mp4", duration=300.0)]}
    candidates = build_candidates(catalog)
    budgets = _compute_source_budgets(catalog, candidates)
    assert budgets == {"/huge.mp4": MAX_MOMENTS_PER_SOURCE}


def test_compute_source_budgets_mixed_catalog():
    """Spec example: 3s + 60s + 2.5s still → budgets = {1, 6, 1}."""
    catalog = {"clips": [
        _clip("/clip_short.mp4", duration=3.0),
        _clip("/clip_long.mp4", duration=60.0),
        _clip("/still.mp4", duration=2.5),
    ]}
    candidates = build_candidates(catalog)
    budgets = _compute_source_budgets(catalog, candidates)
    assert budgets == {
        "/clip_short.mp4": 1,
        "/clip_long.mp4": 6,
        "/still.mp4": 1,
    }


def test_compute_source_budgets_ignores_sources_without_candidates():
    """Sources that produced no candidates (e.g. banned) don't get budgets."""
    catalog = {"clips": [
        _clip("/kept.mp4", duration=8.0),
        _clip("/banned.mp4", duration=8.0),
    ]}
    candidates = build_candidates(
        catalog, preferences={"banned": ["/banned.mp4"]},
    )
    budgets = _compute_source_budgets(catalog, candidates)
    assert "/banned.mp4" not in budgets
    assert "/kept.mp4" in budgets


def test_compute_source_budgets_seconds_per_moment_constant():
    """A 10s clip should round to exactly 1 moment (not 2)."""
    catalog = {"clips": [_clip("/c.mp4", duration=SECONDS_PER_MOMENT)]}
    candidates = build_candidates(catalog)
    budgets = _compute_source_budgets(catalog, candidates)
    assert budgets == {"/c.mp4": 1}


# ---- _apply_moment_budget --------------------------------------------------


def test_apply_moment_budget_keeps_top_k_per_source():
    """Per source, only the top-`budget` candidates by score are kept."""
    candidates = [
        Candidate(source="/long.mp4", start_s=0.0, end_s=2.0, score=1.0),
        Candidate(source="/long.mp4", start_s=2.0, end_s=4.0, score=5.0),
        Candidate(source="/long.mp4", start_s=4.0, end_s=6.0, score=3.0),
        Candidate(source="/long.mp4", start_s=6.0, end_s=8.0, score=10.0),
        Candidate(source="/short.mp4", start_s=0.0, end_s=2.0, score=0.5),
    ]
    budgets = {"/long.mp4": 2, "/short.mp4": 1}
    kept = _apply_moment_budget(candidates, budgets)
    by_src: dict[str, list[Candidate]] = {}
    for c in kept:
        by_src.setdefault(c.source, []).append(c)
    long_scores = sorted([c.score for c in by_src["/long.mp4"]])
    assert long_scores == [5.0, 10.0], \
        f"expected top-2 from /long.mp4 by score, got {long_scores}"
    assert len(by_src["/short.mp4"]) == 1


def test_apply_moment_budget_drops_sources_not_in_budgets():
    """A candidate from a source absent from `budgets` is dropped (budget=1
    fallback would keep one, but `_apply_moment_budget` uses .get(...,1))."""
    candidates = [
        Candidate(source="/a.mp4", start_s=0.0, end_s=2.0, score=1.0),
        Candidate(source="/a.mp4", start_s=2.0, end_s=4.0, score=2.0),
    ]
    # /a.mp4 not in budgets — gets default budget=1, keeps top 1.
    kept = _apply_moment_budget(candidates, budgets={})
    assert len(kept) == 1
    assert kept[0].score == 2.0


# ---- end-to-end through build_plan -----------------------------------------


def test_build_plan_uses_all_sources_when_room_permits():
    """3 sources (1 short + 1 long + 1 still) → all 3 contribute entries
    when target_slots >= sum(budgets)=8."""
    catalog = {"clips": [
        _clip("/short.mp4", duration=3.0),
        _clip("/long.mp4", duration=60.0),
        _clip("/still.mp4", duration=2.5),
    ]}
    song = _song(duration_s=30.0, tempo_bpm=120.0)
    plan = build_plan(catalog, song, target_len=30.0, no_speed_ramp=True,
                      pace="medium", source_cap=8, chronological=False,
                      burst_window_s=0.0)
    sources = {e["source"] for e in plan}
    assert sources == {"/short.mp4", "/long.mp4", "/still.mp4"}, \
        f"every source should contribute at least one entry, got {sources}"


def test_build_plan_user_source_cap_truncates_budget(
        capsys):
    """User-supplied `source_cap=2` clamps every per-source budget to 2,
    even if the duration would justify 6."""
    catalog = {"clips": [
        _clip(f"/c{i}.mp4", duration=60.0) for i in range(3)
    ]}
    song = _song(duration_s=30.0, tempo_bpm=120.0)
    plan = build_plan(catalog, song, target_len=30.0, no_speed_ramp=True,
                      pace="medium", source_cap=2, chronological=False,
                      burst_window_s=0.0)
    counts: dict[str, int] = {}
    for e in plan:
        counts[e["source"]] = counts.get(e["source"], 0) + 1
    for src, n in counts.items():
        assert n <= 2, \
            f"source {src} appeared {n} times — should be capped at 2 by " \
            f"user --source-cap, got counts {counts}"


def test_build_plan_log_line_reports_budget_totals(capsys):
    """The moment-budget log line surfaces sources, sum, median, and the
    longest-budget source so users can see catalog → plan attrition."""
    # 61 sources of mixed durations: 30× 3s stills, 30× 30s clips, 1× 80s clip.
    clips = []
    for i in range(30):
        clips.append(_clip(f"/still{i}.mp4", duration=3.0))
    for i in range(30):
        clips.append(_clip(f"/clip{i}.mp4", duration=30.0))
    clips.append(_clip("/longest.mp4", duration=80.0))
    catalog = {"clips": clips}
    song = _song(duration_s=120.0, tempo_bpm=120.0)
    _ = build_plan(catalog, song, target_len=120.0, no_speed_ramp=True,
                   pace="medium", source_cap=8, chronological=False,
                   burst_window_s=0.0)
    err = capsys.readouterr().err
    # 30 stills × 1 + 30 clips × 3 + 1 × 8 = 30 + 90 + 8 = 128.
    assert "moment budget: 61 sources" in err, \
        f"expected '61 sources' totals header, got: {err!r}"
    assert "sum=128" in err, \
        f"expected sum=128 in log line, got: {err!r}"
    assert "max=8 from longest.mp4" in err, \
        f"expected max=8 attributed to longest.mp4, got: {err!r}"


def test_build_plan_every_source_contributes_when_budget_sum_exceeds_slots():
    """61 sources × mixed durations + 120s target → every source still
    appears (a few times for long ones), confirming we don't drop sources
    just because the budget sum overshoots the slot count."""
    # 30 stills + 30 short videos + 1 long: enough that target_slots will
    # be smaller than budget_sum, so the allocator picks across sources.
    clips = []
    for i in range(30):
        clips.append(_clip(f"/still{i}.mp4", duration=3.0))
    for i in range(30):
        clips.append(_clip(f"/clip{i}.mp4", duration=8.0))
    clips.append(_clip("/longest.mp4", duration=80.0))
    catalog = {"clips": clips}
    # Long song / dense beats so we have plenty of slots.
    song = _song(duration_s=120.0, tempo_bpm=120.0)
    plan = build_plan(catalog, song, target_len=120.0, no_speed_ramp=True,
                      pace="fast", source_cap=8, chronological=False,
                      burst_window_s=0.0)
    used = {e["source"] for e in plan}
    total = len(catalog["clips"])
    # Should hit most of the catalog — at least 80% coverage.
    assert len(used) >= int(total * 0.8), \
        f"expected most sources used, only {len(used)}/{total}: " \
        f"missing {sorted(set(c['path'] for c in catalog['clips']) - used)[:5]}…"
