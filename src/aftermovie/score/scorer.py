"""Candidate scoring + greedy plan builder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aftermovie.config import DEFAULT_TARGET_LEN_S, MIN_CLIP_S
from aftermovie.ffmpeg_cmd import log
from aftermovie.render.transitions import decide_transitions
from aftermovie.score.song import analyze_song
from aftermovie.types import Candidate


def score_window(clip: dict[str, Any], start: int, end: int) -> tuple[float, list[str]]:
    """
    Composite score for a window into a clip.
    Returns (score, list of reasons explaining the score).
    """
    reasons: list[str] = []
    score = 0.0

    motion = clip.get("motion_energy", [])
    motion_avg = (
        sum(motion[start:end]) / (end - start)
        if motion[start:end] else 0.0
    )
    if motion_avg > 0:
        score += motion_avg * 1.5
        if motion_avg > (max(motion) * 0.7 if motion else 0):
            reasons.append("motion_peak")

    audio = clip.get("audio_energy", [])
    audio_avg = (
        sum(audio[start:end]) / (end - start)
        if audio[start:end] else 0.0
    )
    score += audio_avg * 1.0
    if audio_avg > 0.7:
        reasons.append("loud_audio")

    accl = clip.get("accl_peaks", [])
    if accl[start:end]:
        accl_max = max(accl[start:end])
        if accl_max > 15:
            score += 3.0
            reasons.append("high_accel_jump")
        elif accl_max > 12:
            score += 1.5
            reasons.append("moderate_accel")

    speeds = clip.get("gps_speed", [])
    if speeds[start:end]:
        sp_max = max(speeds[start:end])
        sp_overall_max = max(speeds) if speeds else 0
        if sp_overall_max > 0 and sp_max > sp_overall_max * 0.8:
            score += 2.0
            reasons.append("speed_peak")

    win_ms_start = start * 1000
    win_ms_end = end * 1000
    for tag_ms in clip.get("hilight_tags_ms", []):
        if win_ms_start <= tag_ms <= win_ms_end:
            score += 10.0
            reasons.append("hilight_tag")
            break

    faces = clip.get("face_bboxes") or []
    in_window = [f for f in faces[start:end] if f]
    if in_window:
        score += 0.5
        reasons.append("face_present")

    return score, reasons


def build_candidates(catalog: dict[str, Any]) -> list[Candidate]:
    """Turn a catalog of clips into per-second candidate sub-clips with scores."""
    candidates: list[Candidate] = []
    for clip in catalog["clips"]:
        path = clip["path"]
        duration = clip["duration_s"]
        fps = clip.get("fps", 30.0)
        is_short = clip.get("is_short_form", False)
        if duration <= 4.0:
            score, reasons = score_window(clip, 0, int(duration))
            candidates.append(Candidate(
                source=path,
                start_s=0.0,
                end_s=duration,
                score=score,
                reasons=reasons,
                src_fps=fps,
                is_short=is_short,
            ))
            continue
        n_sec = int(duration)
        step = 1
        win = 2
        for start in range(0, max(1, n_sec - win + 1), step):
            end = min(n_sec, start + win)
            score, reasons = score_window(clip, start, end)
            candidates.append(Candidate(
                source=path,
                start_s=float(start),
                end_s=float(end),
                score=score,
                reasons=reasons,
                src_fps=fps,
                is_short=False,
            ))
    return candidates


PACE_TO_FACTOR = {"fast": 1, "medium": 2, "slow": 4}


def _auto_cut_points(beats: list[float], intro_end_s: float, target_len: float,
                     energy_per_s: list[float]) -> list[float]:
    """Energy-aware beat selection: pack cuts tighter during loud sections,
    space them out during quiet ones. Targets ~1.5–2.5s per cut, never tighter
    than 2 beats (so high-BPM songs don't strobe). Mirrors Quik's pacing arc.
    """
    out: list[float] = []
    last_kept = -10**9
    for i, t in enumerate(beats):
        if t < intro_end_s or t >= target_len:
            continue
        idx = min(int(t), len(energy_per_s) - 1) if energy_per_s else 0
        e = energy_per_s[idx] if energy_per_s else 0.5
        # Higher numbers = sparser cuts. Floor at 2 so cuts always last
        # at least ~1s on a typical 100-130 BPM song.
        if e >= 0.8:
            factor = 2
        elif e >= 0.55:
            factor = 3
        elif e >= 0.3:
            factor = 4
        else:
            factor = 6
        if i - last_kept >= factor:
            out.append(t)
            last_kept = i
    return out


def select_cut_points(song: dict[str, Any], target_len: float, pace: str) -> list[float]:
    """Pick beat times that anchor cuts, respecting `pace`.

    pace: "fast" (every beat) / "medium" (every 2nd beat) /
          "slow" (every 4th beat — downbeats only) /
          "auto" (energy-aware — fast on loud sections, slow on quiet ones).

    Returns the list of beat times in [intro_end_s, target_len), with
    `target_len` appended as the terminating sentinel.
    """
    if pace == "auto":
        cut_points = _auto_cut_points(
            song["beats"], song["intro_end_s"], target_len,
            song.get("energy_per_s", []),
        )
        if not cut_points:
            cut_points = [b for b in song["beats"]
                          if song["intro_end_s"] <= b < target_len][::2]
    else:
        factor = PACE_TO_FACTOR.get(pace, 2)
        cut_points = [b for b in song["beats"] if b >= song["intro_end_s"]]
        if not cut_points:
            cut_points = song["beats"]
        if factor > 1:
            cut_points = cut_points[::factor]
        cut_points = [t for t in cut_points if t < target_len]
    cut_points.append(target_len)
    return cut_points


def allocate_candidates(candidates: list[Candidate],
                        cut_points: list[float],
                        source_cap: int = 3) -> list[tuple[float, Candidate]]:
    """Greedy fill: walk cut points, pick the highest-scoring unused candidate
    that hasn't already hit `source_cap` reuses of the same source file.

    Returns `[(beat_t, picked_candidate), ...]` in cut order. The final
    sentinel cut point (used only as the gap terminator) is not yielded.
    """
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    used_sources: dict[str, int] = {}
    picks: list[tuple[float, Candidate]] = []

    for i in range(len(cut_points) - 1):
        beat_t = cut_points[i]
        next_t = cut_points[i + 1]
        gap = next_t - beat_t
        if gap < MIN_CLIP_S:
            continue
        pick: Candidate | None = None
        for c in candidates:
            clip_len = c.end_s - c.start_s
            if clip_len < MIN_CLIP_S:
                continue
            if used_sources.get(c.source, 0) >= source_cap:
                continue
            pick = c
            break
        if not pick:
            continue
        candidates.remove(pick)
        used_sources[pick.source] = used_sources.get(pick.source, 0) + 1
        picks.append((beat_t, pick))
    return picks


def decide_speed(pick: Candidate, beat_t: float,
                 song: dict[str, Any], no_speed_ramp: bool) -> float:
    """Return 0.5 for a slow-mo ramp, 1.0 otherwise.

    A speed ramp fires only when ALL hold:
    - `no_speed_ramp` is False
    - the source is shot at >= 90 fps (so 0.5x stays smooth)
    - the cut lands within 50ms of a downbeat
    - the candidate scored on an action reason
      (`high_accel_jump`, `motion_peak`, or `hilight_tag`)
    """
    if no_speed_ramp:
        return 1.0
    if pick.src_fps < 90:
        return 1.0
    on_downbeat = any(abs(beat_t - db) < 0.05 for db in song["downbeats"])
    if not on_downbeat:
        return 1.0
    if not any(r in ("high_accel_jump", "motion_peak", "hilight_tag") for r in pick.reasons):
        return 1.0
    return 0.5


def build_plan(catalog: dict[str, Any], song: dict[str, Any],
               target_len: float, no_speed_ramp: bool,
               pace: str = "medium",
               source_cap: int = 3,
               chronological: bool = True) -> list[dict[str, Any]]:
    """Greedy-fill plan entries against the song's beat structure.

    Orchestrates three seams:
    - `select_cut_points` chooses the beat anchors
    - `allocate_candidates` picks the best clip per anchor
    - `decide_speed` decides 1.0 vs 0.5 per pick

    Then for each pick we expand the source window up to the source's
    duration, stamp voice-band `audio_interest`, and slice face boxes.

    `source_cap` is the max times a single source file may appear in the
    plan. Pass 1 to forbid duplicates.

    When `chronological` is True (default), the picks are re-ordered by
    capture time (EXIF / ffprobe creation_time / mtime fallback) before
    being bound to beat anchors. Scoring still controls *which* clips win
    a slot; this only controls their order in the timeline.
    """
    # Map source path → original clip dict so we can attach face data + dims.
    by_source = {c["path"]: c for c in catalog["clips"]}
    candidates = build_candidates(catalog)
    cut_points = select_cut_points(song, target_len, pace)
    picks = allocate_candidates(candidates, cut_points, source_cap=source_cap)

    if chronological and picks:
        def _captured(pick) -> float:
            src = by_source.get(pick.source, {}) or {}
            ts = src.get("captured_at")
            return float(ts) if ts is not None else float("inf")

        # Re-bind picks to beat anchors in capture-time order. Anchors stay
        # at their original times; only the (anchor → pick) pairing changes.
        beats = [b for b, _ in picks]
        sorted_picks = sorted([p for _, p in picks], key=_captured)
        picks = list(zip(beats, sorted_picks))

    # Map beat_t → next cut so we can recover the slot duration per pick.
    next_cut: dict[float, float] = {}
    for i in range(len(cut_points) - 1):
        next_cut[cut_points[i]] = cut_points[i + 1]

    plan_entries: list[dict[str, Any]] = []
    for beat_t, pick in picks:
        gap = next_cut[beat_t] - beat_t
        speed = decide_speed(pick, beat_t, song, no_speed_ramp)
        src_time_needed = gap * speed
        src_clip = by_source.get(pick.source, {})
        # Extend up to the source's full duration when filling the slot — the
        # candidate window is only the scoring centroid, not a usage cap.
        src_dur = float(src_clip.get("duration_s", pick.end_s))
        actual_end = min(src_dur, pick.start_s + src_time_needed)
        start_i = int(pick.start_s)
        end_i = max(start_i + 1, int(actual_end + 0.999))
        src_faces = src_clip.get("face_bboxes") or []
        # Mean voice-band energy over the cut's source window. The renderer
        # gates clip audio against this so wind / silence / mumble don't pollute
        # the duck mix; voices and impacts surface through.
        voice = src_clip.get("voice_energy") or []
        if voice:
            window = voice[start_i:end_i] or voice[:1]
            audio_interest = float(sum(window) / max(len(window), 1))
        else:
            audio_interest = 0.0
        plan_entries.append({
            "source": pick.source,
            "start_s": pick.start_s,
            "end_s": actual_end,
            "out_duration_s": gap,
            "speed": speed,
            "beat_time_s": beat_t,
            "score": pick.score,
            "reasons": pick.reasons,
            "audio_interest": audio_interest,
            "source_width": int(src_clip.get("width", 1920)),
            "source_height": int(src_clip.get("height", 1080)),
            "face_bboxes": src_faces[start_i:end_i] if src_faces else [],
        })
    return plan_entries


def cmd_score(args: argparse.Namespace) -> None:
    catalog = json.loads(Path(args.catalog).expanduser().read_text())
    song = analyze_song(Path(args.song).expanduser().resolve())
    log(f"Song: {song['tempo_bpm']:.0f} BPM, "
        f"{len(song['beats'])} beats, intro ends ~{song['intro_end_s']:.1f}s")

    target_len = min(
        song["duration_s"],
        float(args.max_length) if args.max_length else DEFAULT_TARGET_LEN_S,
    )

    entries = build_plan(
        catalog, song, target_len, args.no_speed_ramp,
        pace=getattr(args, "pace", "medium"),
        source_cap=int(getattr(args, "source_cap", 1) or 1),
        chronological=bool(getattr(args, "chronological", True)),
    )
    tmode = getattr(args, "transitions", "cut")
    if tmode in ("auto", "soft"):
        decide_transitions(entries, song, mode=tmode)
    log(f"Built {len(entries)} cuts over {target_len:.1f}s")

    titles: list[dict] = []
    title_flag = getattr(args, "titles", None)
    if title_flag:
        for kind in (k.strip() for k in title_flag.split(",")):
            if kind in ("intro", "outro"):
                titles.append({"kind": kind, "text": getattr(args, "title_text", "") or "",
                               "duration_s": 2.0})

    plan = {
        "song": str(Path(args.song).expanduser().resolve()),
        "song_meta": song,
        "target_length_s": target_len,
        "song_start_s": float(song["intro_end_s"]),
        "aspect": args.aspect,
        "resolution": args.res,
        "fps": args.fps,
        "lut": args.lut,
        "music_db": args.music_db,
        "clip_db": args.clip_db,
        "audio_mix": getattr(args, "audio_mix", "music_only"),
        "transitions": getattr(args, "transitions", "cut"),
        "titles": titles,
        "reframe": not getattr(args, "no_reframe", False),
        "entries": entries,
    }
    out = Path(args.out).expanduser().resolve()
    out.write_text(json.dumps(plan, indent=2))
    log(f"Plan → {out}")
