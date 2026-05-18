"""Candidate scoring + greedy plan builder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from aftermovie.config import DEFAULT_TARGET_LEN_S, MIN_CLIP_S
from aftermovie.ffmpeg_cmd import log
from aftermovie.render.transitions import decide_transitions
from aftermovie.score import components as sc
from aftermovie.score.components import ScoreComponent
from aftermovie.score.song import analyze_song
from aftermovie.types import Candidate

DEFAULT_BURST_WINDOW_S = 3.0

# Float tolerance for `sum(components.values()) == score`. Generous because
# the legacy total is also accumulated by repeated `+=` in arbitrary order,
# so we only need to guard against logic drift, not full IEEE strictness.
_SCORE_DEBUG_EPS = 1e-6


def score_window(
    clip: dict[str, Any], start: int, end: int,
) -> tuple[float, list[str], dict[str, float]]:
    """
    Composite score for a window into a clip.

    Returns ``(score, reasons, components)`` where ``components`` is a
    breakdown of every named signal that contributed to ``score``. Zero
    contributions are omitted so consumers can iterate the dict without
    filtering. Invariant (debug-gated): ``sum(components.values()) == score``.
    """
    reasons: list[str] = []
    components: dict[str, float] = {}
    score = 0.0

    def _add(component: ScoreComponent, delta: float) -> None:
        """Record a contribution from a registered ScoreComponent.

        Accepting a `ScoreComponent` rather than a bare string means typos
        like `_add("hilght_tag", ...)` fail at import time, and the set of
        keys a window can emit is exactly `iter_components()`.
        """
        nonlocal score
        if delta == 0.0:
            return
        # Belt-and-braces: only `ScoreComponent` instances from the registry
        # may write into `components`. Catches in-tree drift if anyone
        # bypasses the typed signature with a raw string.
        if not isinstance(component, ScoreComponent) or not sc.is_known(component.name):
            raise ValueError(
                f"score_window: unknown ScoreComponent {component!r} — "
                f"add it to aftermovie.score.components first"
            )
        score += delta
        # Accumulate in case the same signal contributes more than once
        # (none do today, but keeps the invariant honest).
        components[component.name] = components.get(component.name, 0.0) + delta

    motion = clip.get("motion_energy", [])
    motion_avg = (
        sum(motion[start:end]) / (end - start)
        if motion[start:end] else 0.0
    )
    if motion_avg > 0:
        _add(sc.MOTION, motion_avg * 1.5)
        if motion_avg > (max(motion) * 0.7 if motion else 0):
            reasons.append("motion_peak")

    audio = clip.get("audio_energy", [])
    audio_avg = (
        sum(audio[start:end]) / (end - start)
        if audio[start:end] else 0.0
    )
    _add(sc.AUDIO, audio_avg * 1.0)
    if audio_avg > 0.7:
        reasons.append("loud_audio")

    accl = clip.get("accl_peaks", [])
    if accl[start:end]:
        accl_max = max(accl[start:end])
        if accl_max > 15:
            _add(sc.ACCL_JUMP, 3.0)
            reasons.append("high_accel_jump")
        elif accl_max > 12:
            _add(sc.ACCL_JUMP, 1.5)
            reasons.append("moderate_accel")

    speeds = clip.get("gps_speed", [])
    if speeds[start:end]:
        sp_max = max(speeds[start:end])
        sp_overall_max = max(speeds) if speeds else 0
        if sp_overall_max > 0 and sp_max > sp_overall_max * 0.8:
            _add(sc.GPS_SPEED, 2.0)
            reasons.append("speed_peak")

    win_ms_start = start * 1000
    win_ms_end = end * 1000
    for tag_ms in clip.get("hilight_tags_ms", []):
        if win_ms_start <= tag_ms <= win_ms_end:
            _add(sc.HILIGHT_TAG, 10.0)
            reasons.append("hilight_tag")
            break

    faces = clip.get("face_bboxes") or []
    in_window = [f for f in faces[start:end] if f]
    if in_window:
        _add(sc.FACE, 0.5)
        reasons.append("face_present")

    # Quality penalties. Both lists are absent when cv2 wasn't installed at
    # analyze time — in that case we simply skip the penalty rather than
    # punishing every window in the catalog.
    sharp = clip.get("sharpness_per_s") or []
    sharp_win = sharp[start:end]
    if sharp_win and len(sharp) >= 2:
        sharp_avg = sum(sharp_win) / len(sharp_win)
        # 30th-percentile cutoff within this clip — penalize the softest third.
        # Using <= so a window whose mean lands on the cutoff still trips it
        # (otherwise clips with many tied values slip through).
        sorted_sharp = sorted(sharp)
        p30_idx = max(0, min(len(sorted_sharp) - 1, int(len(sorted_sharp) * 0.3)))
        p30 = sorted_sharp[p30_idx]
        if sharp_avg <= p30:
            _add(sc.BLURRY, -1.5)
            reasons.append("blurry")

    expo = clip.get("exposure_per_s") or []
    expo_win = expo[start:end]
    if expo_win:
        expo_avg = sum(expo_win) / len(expo_win)
        if expo_avg < 0.25 or expo_avg > 0.85:
            _add(sc.POOR_EXPOSURE, -1.5)
            reasons.append("poor_exposure")

    if os.environ.get("AFTERMOVIE_SCORE_DEBUG") == "1":
        total = sum(components.values())
        assert abs(total - score) < _SCORE_DEBUG_EPS, (
            f"score_window invariant broken: sum(components)={total!r} != "
            f"score={score!r} for clip={clip.get('path')} [{start}:{end}] "
            f"components={components!r}"
        )

    return score, reasons, components


def build_candidates(
    catalog: dict[str, Any],
    preferences: dict[str, Any] | None = None,
) -> list[Candidate]:
    """Turn a catalog of clips into per-second candidate sub-clips with scores.

    When `preferences` is provided (from the per-folder
    `.aftermovie-preferences.json` sidecar), the user's bans and favorites
    influence the candidate pool:

    - sources listed in `preferences["banned"]` are skipped entirely so they
      never appear as a Candidate (and therefore never reach the plan);
    - sources listed in `preferences["favorited"]` get a flat `+2.0` boost
      on every Candidate from that source plus the `"user_favorite"` reason
      tag — small enough that a strong objective signal (e.g. a HiLight tag
      worth +10) still dominates, but big enough to break ties between
      similarly-scored clips in the user's favor.
    """
    banned: set[str] = set()
    favorited: set[str] = set()
    if isinstance(preferences, dict):
        raw_banned = preferences.get("banned")
        if isinstance(raw_banned, (list, tuple, set, frozenset)):
            banned = {str(p) for p in raw_banned}
        raw_fav = preferences.get("favorited")
        if isinstance(raw_fav, (list, tuple, set, frozenset)):
            favorited = {str(p) for p in raw_fav}

    candidates: list[Candidate] = []
    for clip in catalog["clips"]:
        path = clip["path"]
        if path in banned:
            continue
        duration = clip["duration_s"]
        fps = clip.get("fps", 30.0)
        is_short = clip.get("is_short_form", False)
        is_favorite = path in favorited
        if duration <= 4.0:
            score, reasons, components = score_window(clip, 0, int(duration))
            if is_favorite:
                score += sc.USER_FAVORITE.weight_hint or 0.0
                reasons = list(reasons) + [sc.USER_FAVORITE.name]
                components = {**components,
                              sc.USER_FAVORITE.name: sc.USER_FAVORITE.weight_hint or 0.0}
            candidates.append(Candidate(
                source=path,
                start_s=0.0,
                end_s=duration,
                score=score,
                reasons=reasons,
                src_fps=fps,
                is_short=is_short,
                components=components,
            ))
            continue
        n_sec = int(duration)
        step = 1
        win = 2
        for start in range(0, max(1, n_sec - win + 1), step):
            end = min(n_sec, start + win)
            score, reasons, components = score_window(clip, start, end)
            if is_favorite:
                score += sc.USER_FAVORITE.weight_hint or 0.0
                reasons = list(reasons) + [sc.USER_FAVORITE.name]
                components = {**components,
                              sc.USER_FAVORITE.name: sc.USER_FAVORITE.weight_hint or 0.0}
            candidates.append(Candidate(
                source=path,
                start_s=float(start),
                end_s=float(end),
                score=score,
                reasons=reasons,
                src_fps=fps,
                is_short=False,
                components=components,
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


def _suppress_visual_duplicates(candidates: list[Candidate],
                                by_source: dict[str, dict[str, Any]]) -> list[Candidate]:
    """Collapse candidates whose source shares a `duplicate_group` (set at
    analyze time by `analyze/duplicates.group_duplicates`). For each cluster,
    keep ONLY the highest-scoring candidate; drop the rest of the group.

    Sources without a duplicate_group (singletons / clips that had no phash)
    are passed through untouched.

    Composes with `_suppress_bursts`: bursts fire first to drop same-moment
    same-camera spam, then this filter catches same-look-from-anywhere.
    Doing it in this order means a burst cluster's best candidate has
    already been chosen by the time we ask "is this also a visual twin of
    something else?", so we never overcount the work."""
    if not candidates:
        return candidates

    # Per-group, find the source whose best-scoring candidate wins.
    group_best_source: dict[str, tuple[float, str]] = {}
    for c in candidates:
        src = by_source.get(c.source) or {}
        gid = src.get("duplicate_group")
        if not gid:
            continue
        prev = group_best_source.get(gid)
        if prev is None or c.score > prev[0]:
            group_best_source[gid] = (c.score, c.source)

    if not group_best_source:
        return candidates

    # Every source in a cluster that ISN'T the winner gets dropped.
    suppressed_sources: set[str] = set()
    cluster_members: dict[str, list[str]] = {}
    for src, info in by_source.items():
        gid = info.get("duplicate_group")
        if gid:
            cluster_members.setdefault(gid, []).append(src)
    for gid, members in cluster_members.items():
        keep = group_best_source.get(gid, (0.0, None))[1]
        for src in members:
            if src != keep:
                suppressed_sources.add(src)

    if not suppressed_sources:
        return candidates
    log(f"visual-dup suppress: dropped {len(suppressed_sources)} source(s) "
        f"sharing a duplicate_group with a higher-scored sibling")
    return [c for c in candidates if c.source not in suppressed_sources]


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
               stretch_stills: bool = True,
               preferences: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
    candidates = build_candidates(catalog, preferences=preferences)
    # Filter order matters: bursts first (catches same-moment, same-camera
    # repetition where the only signal is timestamp clustering), then
    # visual-duplicate grouping (catches same-look-from-anywhere using the
    # phashes attached at analyze time). Reversing the order would let
    # burst-suppression mask a visual-twin survivor we'd actually want to
    # drop. See analyze/duplicates.py for the dHash details.
    candidates = _suppress_bursts(candidates, by_source, window_s=burst_window_s)
    candidates = _suppress_visual_duplicates(candidates, by_source)
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
            "components": dict(pick.components),
            "audio_interest": audio_interest,
            "source_width": int(src_clip.get("width", 1920)),
            "source_height": int(src_clip.get("height", 1080)),
            "face_bboxes": src_faces[start_i:end_i] if src_faces else [],
        })

    # When the allocator emits fewer entries than beat slots (often because a
    # still/live-photo heavy folder has many one-window sources), don't dump
    # all skipped time into the predecessor or final entry. That makes the last
    # image hang for ages. Instead, distribute the target length evenly across
    # the chosen entries so the edit stays balanced.
    if stretch_stills and plan_entries:
        n_slots = max(0, len(cut_points) - 1)
        if len(plan_entries) < n_slots:
            even_gap = float(target_len) / len(plan_entries)
            beat_t = 0.0
            for e in plan_entries:
                e["beat_time_s"] = beat_t
                e["out_duration_s"] = even_gap
                beat_t += even_gap

    return plan_entries


def _infer_clips_root(catalog: dict[str, Any]) -> Path | None:
    """Best-effort guess at the clips folder a catalog was built from.

    The Score stage (and the MCP `score_clips` tool) doesn't take a
    `--clips` argument today — the catalog records absolute source paths
    instead. To find the per-folder `.aftermovie-preferences.json` sidecar
    we walk the longest common prefix of those source paths and treat the
    deepest existing directory as the clips root. Returns None when the
    catalog has no clips or the prefix doesn't resolve to a directory.
    """
    sources = [c.get("path") for c in catalog.get("clips", [])
               if isinstance(c.get("path"), str)]
    if not sources:
        return None
    import os
    common = os.path.commonpath(sources)
    p = Path(common)
    # If commonpath gave us a file (e.g. a single source), walk up to its parent.
    if p.is_file():
        p = p.parent
    if p.is_dir():
        return p
    return None


def cmd_score(args: argparse.Namespace) -> None:
    from aftermovie.analyze.preferences import load_preferences

    catalog = json.loads(Path(args.catalog).expanduser().read_text())
    song = analyze_song(Path(args.song).expanduser().resolve())
    log(f"Song: {song['tempo_bpm']:.0f} BPM, "
        f"{len(song['beats'])} beats, intro ends ~{song['intro_end_s']:.1f}s")

    target_len = min(
        song["duration_s"],
        float(args.max_length) if args.max_length else DEFAULT_TARGET_LEN_S,
    )

    # Preferences live in a sidecar next to the user's clips. `cmd_score`
    # only sees the catalog (not the original clips folder) so we derive the
    # folder from the longest common prefix of source paths and load the
    # sidecar from there. Missing sidecar → empty prefs → scorer behavior
    # unchanged from before this issue landed.
    preferences: dict[str, Any] | None = None
    clips_root = _infer_clips_root(catalog)
    if clips_root is not None:
        preferences = load_preferences(clips_root)

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
        preferences=preferences,
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
