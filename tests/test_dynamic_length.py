"""Phase C dynamic-length planner tests.

Covers:
- C1: `max_length=None` → plan duration tracks the Song's duration, not 90s.
- C1: `max_length=60` → plan duration ~ 60s regardless of song length.
- C3 stretch mode: small Source folder + long Song → planner fills the
  target via cap-bump / still-stretch / tail-stretch and logs each lever.
- C4 subset mode: huge Source folder + short Song → planner keeps only the
  top-scoring candidates and logs the trim.

These tests stub the analyze step (synthetic Catalogs) so they're hermetic
and run in milliseconds — no ffmpeg, no librosa.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from aftermovie.score import scorer as scorer_mod
from aftermovie.score.scorer import build_plan, cmd_score


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
    """Synthetic Song dict with a steady beat grid spanning `duration_s`."""
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


# ---- C1: target_len falls out of `max_length=None` -------------------------


def _run_cmd_score(tmp_path: Path,
                   catalog: dict[str, Any],
                   song_dict: dict[str, Any],
                   max_length: float | None,
                   monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Drive `cmd_score` end-to-end against synthetic inputs.

    Avoids librosa by monkeypatching `analyze_song` to return the canned
    Song dict. Returns the persisted Plan JSON.
    """
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(catalog))
    plan_path = tmp_path / "plan.json"
    fake_song_file = tmp_path / "song.mp3"
    fake_song_file.write_bytes(b"")  # cmd_score only stats the path

    monkeypatch.setattr(scorer_mod, "analyze_song",
                        lambda _path: song_dict)
    args = argparse.Namespace(
        catalog=str(cat_path),
        song=str(fake_song_file),
        out=str(plan_path),
        max_length=max_length,
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
    return json.loads(plan_path.read_text())


def test_max_length_none_uses_full_song_duration(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
    """`max_length=None` (no user override) → target_length_s == song duration.

    Previously the planner capped at 90s; post-C1 the Song's `duration_s`
    is the canonical default, so a 156s Song renders a 156s aftermovie.
    """
    catalog = {"clips": [_clip(f"/c{i}.mp4", duration=8.0)
                         for i in range(20)]}
    song = _song(duration_s=156.0)
    plan = _run_cmd_score(tmp_path, catalog, song,
                          max_length=None, monkeypatch=monkeypatch)
    target_len = float(plan["target_length_s"])
    assert abs(target_len - 156.0) < 0.001, \
        f"target_length_s should equal song duration (156), got {target_len}"
    # And the entry timeline should span ~156s (±5s tolerance for end-of-song
    # beat slot rounding).
    plan_dur = sum(float(e["out_duration_s"]) for e in plan["entries"])
    assert abs(plan_dur - 156.0) <= 5.0, \
        f"plan duration {plan_dur:.1f}s drifted too far from target 156.0s"


def test_max_length_override_caps_below_song(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    """User-supplied `max_length=60` clamps regardless of a longer Song."""
    catalog = {"clips": [_clip(f"/c{i}.mp4", duration=8.0)
                         for i in range(20)]}
    song = _song(duration_s=156.0)
    plan = _run_cmd_score(tmp_path, catalog, song,
                          max_length=60.0, monkeypatch=monkeypatch)
    target_len = float(plan["target_length_s"])
    assert abs(target_len - 60.0) < 0.001, \
        f"max_length=60 should cap target_length_s at 60, got {target_len}"


# ---- C3: stretch mode ------------------------------------------------------


def test_stretch_mode_fills_target_with_small_source_folder(
        capsys: pytest.CaptureFixture[str]):
    """5 unique 4s clips + 60s target → planner stretches to fill ~60s.

    Under F3 moment-budget mode each 4s clip's budget caps at 1 so this
    test exercises the legacy `stretch_stills` even-distribution path
    (lever 2 / lever 3 are downstream of allocate). The total must still
    fill the target window and the moment-budget log line must surface."""
    catalog = {"clips": [_clip(f"/c{i}.mp4", duration=4.0) for i in range(5)]}
    song = _song(duration_s=60.0, tempo_bpm=120.0)

    plan = build_plan(catalog, song, target_len=60.0, no_speed_ramp=True,
                      pace="medium", source_cap=1, chronological=False,
                      burst_window_s=0.0)

    assert plan, "stretch mode should still produce a populated plan"
    total = sum(float(e["out_duration_s"]) for e in plan)
    assert total >= 60.0 * 0.85, \
        f"stretch mode failed to fill 60s target (got {total:.1f}s)"

    err = capsys.readouterr().err
    # F3: the moment-budget log line replaces the old subset/auto-bump
    # surfacing. Stretch levers may still fire (when even-distribution
    # falls short of target) so accept any of them too.
    assert ("moment budget" in err
            or "stretch-mode" in err
            or "stretched" in err), \
        f"expected a moment-budget / stretch log line, got stderr: {err!r}"


def test_moment_budget_protects_winners_from_huge_pool(
        capsys: pytest.CaptureFixture[str]):
    """F3: 200 short clips + 30s target → planner picks the top winners by
    score from the moment-budgeted pool. Replaces the legacy C4 subset-mode
    behavior, which globally trimmed the pool's tail; the new behavior
    keeps every source's best window AND still surfaces the top scorers.

    Winners must dominate because their per-source budget=1 candidate
    out-scores every quiet source's per-source budget=1 candidate."""
    # 200 candidates, all roughly the same baseline score except 15 with
    # HiLight tags so they out-score everything else.
    winners = {f"/w{i}.mp4" for i in range(15)}
    clips = []
    for i in range(200):
        path = f"/w{i}.mp4" if i < 15 else f"/q{i}.mp4"
        kw: dict[str, Any] = {}
        if path in winners:
            kw["hilight_tags_ms"] = [1000]
        clips.append(_clip(path, duration=4.0, **kw))
    catalog = {"clips": clips}
    song = _song(duration_s=30.0, tempo_bpm=120.0)

    # source_cap=3 is the user's hard ceiling; the moment budget for each
    # 4s clip is 1 so the effective per-source cap clamps to 1 anyway.
    # The planner should fill ~30 slots from 30 unique sources, leading
    # with all 15 winners.
    plan = build_plan(catalog, song, target_len=30.0, no_speed_ramp=True,
                      pace="medium", source_cap=3, chronological=False,
                      burst_window_s=0.0)

    err = capsys.readouterr().err
    assert "moment budget" in err, \
        f"expected moment-budget log, got stderr: {err!r}"

    assert plan, "planner should still produce a populated plan"
    chosen = {e["source"] for e in plan}
    # All 15 winners must make it into the plan — moment budget is here to
    # protect the top scorers from being crowded out by the noise pool.
    winners_used = chosen & winners
    assert winners_used == winners, \
        f"moment-budget mode dropped {len(winners) - len(winners_used)} " \
        f"winner(s) from the plan: missing {winners - winners_used}"
    # And every entry must be a unique source (budget=1 per short clip):
    # no source repeats, even though source_cap=3 would have allowed up to 3.
    assert len(chosen) == len(plan), \
        f"expected one unique source per entry, got {len(chosen)} sources " \
        f"for {len(plan)} entries"
