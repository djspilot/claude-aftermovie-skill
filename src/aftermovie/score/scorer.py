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


def build_plan(catalog: dict[str, Any], song: dict[str, Any],
               target_len: float, no_speed_ramp: bool) -> list[dict[str, Any]]:
    """Greedy-fill plan entries against the song's beat structure."""
    # Map source path → original clip dict so we can attach face data + dims.
    by_source = {c["path"]: c for c in catalog["clips"]}
    candidates = build_candidates(catalog)

    cut_points = [b for b in song["beats"] if b >= song["intro_end_s"]]
    if not cut_points:
        cut_points = song["beats"]
    cut_points = [t for t in cut_points if t < target_len]
    cut_points.append(target_len)

    candidates.sort(key=lambda c: c.score, reverse=True)
    used_sources: dict[str, int] = {}
    plan_entries: list[dict[str, Any]] = []

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
            if used_sources.get(c.source, 0) >= 3:
                continue
            pick = c
            break
        if not pick:
            continue
        candidates.remove(pick)
        used_sources[pick.source] = used_sources.get(pick.source, 0) + 1

        on_downbeat = any(abs(beat_t - db) < 0.05 for db in song["downbeats"])
        is_high_fps = pick.src_fps >= 90
        wants_slowmo = (
            is_high_fps
            and on_downbeat
            and any(r in ("high_accel_jump", "motion_peak", "hilight_tag") for r in pick.reasons)
            and not no_speed_ramp
        )
        speed = 0.5 if wants_slowmo else 1.0
        src_time_needed = gap * speed
        actual_end = min(pick.end_s, pick.start_s + src_time_needed)

        src_clip = by_source.get(pick.source, {})
        start_i = int(pick.start_s)
        end_i = max(start_i + 1, int(actual_end + 0.999))
        src_faces = src_clip.get("face_bboxes") or []
        plan_entries.append({
            "source": pick.source,
            "start_s": pick.start_s,
            "end_s": actual_end,
            "out_duration_s": gap,
            "speed": speed,
            "beat_time_s": beat_t,
            "score": pick.score,
            "reasons": pick.reasons,
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

    entries = build_plan(catalog, song, target_len, args.no_speed_ramp)
    if getattr(args, "transitions", "cut") == "auto":
        decide_transitions(entries, song)
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
