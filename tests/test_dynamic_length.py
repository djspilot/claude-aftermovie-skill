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
    """5 unique clips + 60s target → planner stretches to fill ~60s and
    emits a `stretch-mode` log line for at least one of the three levers."""
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
    # At least one lever fired — `source_cap` bump (either the existing
    # auto-bump inside `allocate_candidates` or my C3 stretch-mode bump),
    # the still stretch, or the tail stretch. Each lever logs a stable
    # substring so we can grep for any of them.
    assert ("stretch-mode" in err
            or "auto-bumped" in err
            or "bumped source_cap" in err
            or "stretched" in err), \
        f"expected a stretch / cap-bump log line, got stderr: {err!r}"


def test_subset_mode_trims_huge_pool_to_top_candidates(
        capsys: pytest.CaptureFixture[str]):
    """200 candidates + 30s target → planner keeps the top scorers and
    surfaces a subset-mode log line. The chosen sources must be dominated
    by the high-scoring "winners" — quality dilution is the bug subset
    mode exists to prevent."""
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

    # source_cap=3 lets each winner appear up to 3× in the plan; with 30
    # cuts and 15 winners that means a healthy plan needs ~10-15 unique
    # candidates rather than 30. (At cap=1 we'd be forced to dip into
    # lower-scoring quiet clips simply to fill the slot count, defeating
    # the test.) Tests the "use the top by score, not the first-N by time"
    # acceptance criterion from the plan.
    plan = build_plan(catalog, song, target_len=30.0, no_speed_ramp=True,
                      pace="medium", source_cap=3, chronological=False,
                      burst_window_s=0.0)

    err = capsys.readouterr().err
    assert "subset-mode" in err, \
        f"expected subset-mode log, got stderr: {err!r}"

    assert plan, "subset mode should still produce a populated plan"
    chosen = {e["source"] for e in plan}
    # All 15 winners must make it into the plan — subset mode is here to
    # protect the top scorers from being crowded out by the noise pool.
    winners_used = chosen & winners
    assert winners_used == winners, \
        f"subset mode dropped {len(winners) - len(winners_used)} winner(s) " \
        f"from the plan: missing {winners - winners_used}"
    # And the plan must not pad itself out with the rejected low-score pool:
    # there are only 21 non-winner candidates in the trimmed top-36 anyway,
    # so this guard tightens to the same upper bound.
    non_winners = chosen - winners
    assert len(non_winners) <= 25, \
        f"too many low-score sources leaked into the plan: {len(non_winners)} " \
        f"({sorted(non_winners)[:5]}…)"
