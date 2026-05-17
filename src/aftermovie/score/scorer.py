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

DEFAULT_BURST_WINDOW_S = 3.0


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


def _source_captured_at(src_clip: dict[str, Any]) -> float | None:
    """Best-effort capture timestamp for a source clip. Returns None when
    missing — caller leaves the source in place."""
    ts = src_clip.get("captured_at")
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def _suppress_bursts(candidates: list[Candidate],
                     by_source: dict[str, dict[str, Any]],
                     window_s: float = DEFAULT_BURST_WINDOW_S) -> list[Candidate]:
    """Drop candidates from sources whose capture time is within `window_s`
    of a higher-scored sibling. Clustering is at the SOURCE level; only the
    cluster's best-scored source keeps its candidates. `window_s <= 0` disables."""
    if window_s <= 0 or not candidates:
        return candidates

    max_score: dict[str, float] = {}
    for c in candidates:
        if c.score > max_score.get(c.source, float("-inf")):
            max_score[c.source] = c.score

    timed: list[tuple[float, str]] = []
    for src in max_score:
        ts = _source_captured_at(by_source.get(src, {"path": src}))
        if ts is not None:
            timed.append((ts, src))
    timed.sort()

    suppressed: set[str] = set()
    cluster: list[str] = []
    last_ts: float | None = None
    for ts, src in timed:
        if last_ts is None or (ts - last_ts) <= window_s:
            cluster.append(src)
        else:
            if len(cluster) > 1:
                best = max(cluster, key=lambda s: max_score[s])
                suppressed.update(s for s in cluster if s != best)
            cluster = [src]
        last_ts = ts
    if len(cluster) > 1:
        best = max(cluster, key=lambda s: max_score[s])
        suppressed.update(s for s in cluster if s != best)

    if not suppressed:
        return candidates
    # Safety: if we'd drop > 70% of timed sources, the timestamps are almost
    # certainly bogus (e.g. all-from-mtime after `cp -r` collapsed them into
    # the same second). Skip suppression rather than gut the catalog.
    if len(timed) >= 3 and len(suppressed) / max(len(timed), 1) > 0.50:
        return candidates
    log(f"burst-suppress: dropped {len(suppressed)} source(s) within "
        f"{window_s:.1f}s of a higher-scored sibling")
    return [c for c in candidates if c.source not in suppressed]


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
                        source_cap: int = 3,
                        auto_bump_cap: bool = True,
                        max_auto_cap: int = 5) -> list[tuple[float, Candidate]]:
    """Greedy fill: walk cut points, pick the highest-scoring unused candidate
    that hasn't already hit `source_cap` reuses of the same source file.

    When `auto_bump_cap=True` (default) and the available unique sources
    can't fill every cut point at the given `source_cap`, the cap is bumped
    automatically (up to `max_auto_cap`). This lets users hit a target
    length (e.g. `--max-length 120`) even when their source folder is small:
    some sources will appear more than once, but the alternative is a
    truncated edit. A single log line surfaces the bump.

    Returns `[(beat_t, picked_candidate), ...]` in cut order.
    """
    n_unique = len({c.source for c in candidates})
    n_slots = max(0, len(cut_points) - 1)
    effective_cap = source_cap
    # Only auto-bump from the strict-no-duplicates default. When the caller
    # explicitly passed source_cap > 1 they already weighed reuse vs. variety.
    if (auto_bump_cap and source_cap == 1
            and n_unique > 0 and n_unique * source_cap < n_slots):
        # Round up: e.g. 64 sources, 75 slots, cap=1 → need cap=2.
        needed = -(-n_slots // n_unique)
        effective_cap = min(max(source_cap, needed), max_auto_cap)
        if effective_cap > source_cap:
            log(f"  ! source_cap auto-bumped {source_cap} → {effective_cap} to fit "
                f"{n_slots} cuts from {n_unique} unique sources")

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
            if used_sources.get(c.source, 0) >= effective_cap:
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
                 song: dict[str, Any], no_speed_ramp: bool) -> tuple[float, float]:
    """Return (start_speed, end_speed) for a candidate cut.

    Beat-anchored ramp: when a high-fps source lands on a downbeat with an
    action-peak reason, we ramp 0.4x → 1.0x across the slot so the cut
    "lands" on the beat at full speed. Otherwise flat (1.0, 1.0).
    """
    if no_speed_ramp:
        return (1.0, 1.0)
    if pick.src_fps < 90:
        return (1.0, 1.0)
    on_downbeat = any(abs(beat_t - db) < 0.05 for db in song["downbeats"])
    if not on_downbeat:
        return (1.0, 1.0)
    if not any(r in ("high_accel_jump", "motion_peak", "hilight_tag") for r in pick.reasons):
        return (1.0, 1.0)
    return (0.4, 1.0)


def build_plan(catalog: dict[str, Any], song: dict[str, Any],
               target_len: float, no_speed_ramp: bool,
               pace: str = "medium",
               source_cap: int = 3,
               chronological: bool = True,
               burst_window_s: float = DEFAULT_BURST_WINDOW_S,
               hook: bool = True,
               climax: bool = True,
               stretch_stills: bool = True) -> list[dict[str, Any]]:
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
    candidates = _suppress_bursts(candidates, by_source, window_s=burst_window_s)
    cut_points = select_cut_points(song, target_len, pace)
    picks = allocate_candidates(candidates, cut_points, source_cap=source_cap)

    if chronological and picks:
        def _captured(pick) -> float:
            src = by_source.get(pick.source, {}) or {}
            ts = src.get("captured_at")
            return float(ts) if ts is not None else float("inf")

        beats = [b for b, _ in picks]
        sorted_picks = sorted([p for _, p in picks], key=_captured)
        n = len(sorted_picks)
        # Skip the trailer arc on very short plans (<8 cuts) — the curve has
        # no room to breathe and ends up dropping entries.
        if n < 8 or (not hook and not climax):
            picks = list(zip(beats, sorted_picks))
        else:
            by_score = sorted([p for _, p in picks],
                              key=lambda p: p.score, reverse=True)
            climax_n = max(1, n // 4) if climax else 0
            hook_pick = by_score[0] if hook else None
            # Hook is removed from climax_set FIRST, then climax_tail is
            # padded from the next-best non-hook picks so we never shrink.
            climax_pool = [p for p in by_score
                           if hook_pick is None or id(p) is not id(hook_pick)]
            climax_tail = climax_pool[:climax_n]
            climax_ids = {id(p) for p in climax_tail}
            if hook_pick is not None:
                climax_ids.add(id(hook_pick))
            body = [p for p in sorted_picks if id(p) not in climax_ids]
            new_order = ([hook_pick] if hook_pick else []) + body + climax_tail
            # Guarantee we fill every beat anchor — if hook+climax math
            # under-fills (rare edge), pad from sorted_picks tail.
            if len(new_order) < n:
                used = {id(p) for p in new_order}
                new_order += [p for p in sorted_picks if id(p) not in used]
            picks = list(zip(beats, new_order[:n]))

    # Map beat_t → next cut so we can recover the slot duration per pick.
    next_cut: dict[float, float] = {}
    for i in range(len(cut_points) - 1):
        next_cut[cut_points[i]] = cut_points[i + 1]

    plan_entries: list[dict[str, Any]] = []
    for beat_t, pick in picks:
        gap = next_cut[beat_t] - beat_t
        speed_start, speed_end = decide_speed(pick, beat_t, song, no_speed_ramp)
        # Use start-speed for slot-fill math (matches the legacy single-speed
        # behaviour); the renderer applies the ramp internally.
        speed = speed_start
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
            "speed_start": speed_start,
            "speed_end": speed_end,
            "beat_time_s": beat_t,
            "score": pick.score,
            "reasons": pick.reasons,
            "audio_interest": audio_interest,
            "source_width": int(src_clip.get("width", 1920)),
            "source_height": int(src_clip.get("height", 1080)),
            "face_bboxes": src_faces[start_i:end_i] if src_faces else [],
        })

    # Stretch-stills: if there are unfilled beat slots after this entry (the
    # allocator ran out of unique sources), extend this entry's slot to cover
    # the gap. The renderer's tpad/apad logic already holds the last frame
    # for still-derived clips, so a stretched still looks like a Ken Burns
    # that settles instead of a duplicated cut.
    if stretch_stills and plan_entries:
        # If the LAST plan_entry doesn't reach target_len, extend it.
        target_end = float(target_len)
        # Walk in pairs and absorb skipped slots into the predecessor.
        for i, e in enumerate(plan_entries):
            this_beat = float(e["beat_time_s"])
            if i + 1 < len(plan_entries):
                next_beat = float(plan_entries[i + 1]["beat_time_s"])
            else:
                next_beat = target_end
            true_gap = next_beat - this_beat
            if true_gap > e["out_duration_s"] + 1e-6:
                e["out_duration_s"] = true_gap

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
        burst_window_s=float(getattr(args, "burst_window_s",
                                      DEFAULT_BURST_WINDOW_S)
                              or DEFAULT_BURST_WINDOW_S),
        hook=bool(getattr(args, "hook", True)),
        climax=bool(getattr(args, "climax", True)),
        stretch_stills=bool(getattr(args, "stretch_stills", True)),
    )
    tmode = getattr(args, "transitions", "cut")
    if tmode in ("auto", "soft"):
        decide_transitions(entries, song, mode=tmode)
    log(f"Built {len(entries)} cuts over {target_len:.1f}s")

    # Heuristic: pace=auto produces beat anchors based on tempo + energy.
    # If the planner emitted noticeably fewer cuts than the target window's
    # beat density, we ran out of unique sources under the current
    # source_cap — tell the user so they can raise it or add clips.
    expected_cuts_per_s = float(song.get("tempo_bpm", 100)) / 60 / 3
    expected_cuts = int(target_len * expected_cuts_per_s)
    cap = int(getattr(args, "source_cap", 1) or 1)
    if len(entries) < expected_cuts * 0.8 and cap < 4:
        unique = len({e["source"] for e in entries})
        log(f"  ! only {len(entries)} cuts fit (wanted ~{expected_cuts}) — "
            f"{unique} unique sources at source_cap={cap}. "
            f"Add more clips or raise --source-cap (e.g. {cap+1}).")

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
