"""Candidate scoring + greedy plan builder."""
from __future__ import annotations

import argparse
import json
import math
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

# Phase C5 — per-section beat-stride factors for pace=auto.
# Numbers are "how many beats between consecutive cuts" inside each Section.
# Numbers come from docs/IMPROVEMENT_PLAN.md Phase 4: intro breathes (4 beats),
# verse is medium (2 beats), build tightens (down to 1 beat on the way to the
# drop), drop is action-tight (1 beat — never tighter, anti-strobe), outro is
# medium and the climax-tail in build_plan dilates the last few entries.
SECTION_TO_FACTOR = {
    "intro": 4,
    "verse": 2,
    "build": 1,
    "drop": 1,
    "outro": 2,
}


def _section_for_time(t: float, sections: list[dict[str, Any]]) -> str:
    """Return the `kind` of the section containing `t`, or `"verse"` when
    `sections` is empty / the time falls in a gap. Linear scan is fine —
    a typical song has <8 sections."""
    if not sections:
        return "verse"
    for s in sections:
        if float(s.get("start_s", 0.0)) <= t < float(s.get("end_s", 0.0)):
            k = s.get("kind")
            if isinstance(k, str):
                return k
    # Past the last section's end → treat as outro for late-tail picking.
    return "verse"


def _auto_cut_points(beats: list[float], intro_end_s: float, target_len: float,
                     energy_per_s: list[float],
                     sections: list[dict[str, Any]] | None = None,
                     ) -> list[float]:
    """Energy-aware beat selection: pack cuts tighter during loud sections,
    space them out during quiet ones. Targets ~1.5–2.5s per cut, never tighter
    than 2 beats (so high-BPM songs don't strobe). Mirrors Quik's pacing arc.

    When `sections` is provided (Phase C5), the per-beat stride factor is
    decided by the section kind rather than by raw energy: intro=4, verse=2,
    build=1, drop=1, outro=2. Drops still get a half-beat micro-pack guard
    via the next-beat distance, but never closer than 2 beats apart in
    real-time (the anti-strobe floor). When sections are absent we fall back
    to the legacy energy-banded behaviour so old `song.json` files still work.
    """
    out: list[float] = []
    last_kept = -10**9
    last_kept_t = -1.0e9
    # Absolute anti-strobe floor in real-time. 0.33s ≈ 12fps at the
    # video-cut level, which is the perceptual upper bound where the
    # human eye still parses a cut as "a cut" rather than a flicker. The
    # legacy energy-banded path was its own implicit anti-strobe via the
    # `factor=2` floor; in section-aware mode the drop's `factor=1` is
    # allowed to fire at the song's beat tempo (which on a 180 BPM track
    # is 0.333s — exactly the floor below).
    min_gap_s = 0.33
    for i, t in enumerate(beats):
        if t < intro_end_s or t >= target_len:
            continue
        if sections:
            kind = _section_for_time(t, sections)
            factor = SECTION_TO_FACTOR.get(kind, 2)
        else:
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
        if i - last_kept >= factor and (t - last_kept_t) >= min_gap_s - 1e-9:
            out.append(t)
            last_kept = i
            last_kept_t = t
    return out


def select_cut_points(song: dict[str, Any], target_len: float, pace: str) -> list[float]:
    """Pick beat times that anchor cuts, respecting `pace`.

    pace: "fast" (every beat) / "medium" (every 2nd beat) /
          "slow" (every 4th beat — downbeats only) /
          "auto" (energy-aware — fast on loud sections, slow on quiet ones,
                  section-aware when `song["sections"]` is populated).

    Returns the list of beat times in [intro_end_s, target_len), with
    `target_len` appended as the terminating sentinel.
    """
    if pace == "auto":
        cut_points = _auto_cut_points(
            song["beats"], song["intro_end_s"], target_len,
            song.get("energy_per_s", []),
            sections=song.get("sections") or None,
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


# Phase C5 — per-section candidate-rank bias.
# When a cut anchor falls inside a section of a given kind, the allocator
# adds the matching bias dict's contributions (multiplied by the candidate's
# matching `components` value) to the candidate's score for the purpose of
# picking. The candidate's persisted `.score` and `.components` are NOT
# mutated — this is a local re-rank for greedy selection only, so the plan
# json continues to record the original score the user can reason about.
#
# Numbers are deliberately modest: a HiLight tag is +10 in the base score,
# so a +2 nudge from "this is the drop" is enough to flip a tie between two
# similarly-scored picks without overriding genuine signal differences.
SECTION_BIAS: dict[str, dict[str, float]] = {
    # Drops want energy: motion, hard impacts, and any user-tagged hilight.
    "drop": {
        "motion": 2.0,
        "accl_jump": 1.5,
        "hilight_tag": 1.0,
        "gps_speed": 1.0,
        "face": -0.5,  # gently de-prioritise talking-head moments on drops
    },
    "build": {
        # Build mirrors the drop but at half-strength — the tension is rising
        # so we want movement, but the punch lands inside the drop proper.
        "motion": 1.0,
        "accl_jump": 0.75,
    },
    # Verses are the story beats: people, faces, moderate motion.
    "verse": {
        "face": 1.5,
        "motion": 0.5,
    },
    # Intro / outro stay close to neutral — the climax-tail logic in
    # build_plan handles the outro's dilation, and the intro is short.
    "intro": {
        "face": 0.5,
    },
    "outro": {
        "face": 0.5,
    },
}


def _section_picker_score(c: Candidate, kind: str) -> float:
    """Score a candidate for the *picker* under a given section kind.

    Applies `SECTION_BIAS[kind]` on top of the candidate's base `score` by
    looking up the components it actually carries. A candidate with no
    `motion` component contributes nothing for the `motion` bias, which is
    exactly what we want (we're amplifying the signals the scorer already
    surfaced, not inventing new ones)."""
    bias = SECTION_BIAS.get(kind) or {}
    if not bias or not c.components:
        return c.score
    bonus = 0.0
    for key, weight in bias.items():
        # `c.components[key]` is the absolute contribution that component
        # made to the base score. We treat its presence-and-magnitude as a
        # proxy for "this candidate has that signal" and add `weight` per
        # unit of it, capped softly to avoid double-counting massive signals
        # like a +10 hilight_tag.
        v = float(c.components.get(key, 0.0))
        if v == 0.0:
            continue
        # Saturate: positive components past 5.0 stop adding extra bias so a
        # single dominant signal doesn't crowd out a balanced candidate.
        if v > 0:
            v = min(v, 5.0)
        else:
            v = max(v, -5.0)
        bonus += weight * (v / 1.0)
    return c.score + bonus


def allocate_candidates(candidates: list[Candidate],
                        cut_points: list[float],
                        source_cap: int = 3,
                        auto_bump_cap: bool = True,
                        max_auto_cap: int = 3,
                        sections: list[dict[str, Any]] | None = None,
                        source_budgets: dict[str, int] | None = None,
                        ) -> list[tuple[float, Candidate]]:
    """Greedy fill: walk cut points, pick the highest-scoring unused candidate
    that hasn't already hit its per-source reuse cap.

    Per-source cap resolution (F3): when `source_budgets` is provided, each
    source's effective cap is `min(source_budgets[source], source_cap)`. The
    user-supplied `source_cap` becomes a HARD CEILING — a budget of 6 with
    `source_cap=2` clamps to 2 — but otherwise the budget controls the cap
    so a 70s GoPro can contribute up to 7 distinct moments while a 4s Live
    Photo can only contribute 1.

    When `auto_bump_cap=True` and `source_budgets is None` (legacy path) and
    the unique sources can't fill every cut point at the given `source_cap`,
    the cap is bumped automatically (up to `max_auto_cap`). This path is
    near-unreachable under F3 moment-budget mode because each source's
    budget already covers more than one moment for long footage, but it
    remains intact for callers that bypass `build_plan` (e.g. unit tests
    that pass a hand-rolled Candidate list).

    F1 acceptance (legacy path): if the bump-free plan would only be
    *slightly* short (within `AUTO_BUMP_UNDERFILL_TOLERANCE` of the slot
    count), we accept the shorter plan and log `accepted slight underfill`
    instead of bumping. A 3.8%-short edit beats a 4×-repeat edit.

    When `sections` is provided (Phase C5), the per-anchor candidate ranking
    is re-scored via `SECTION_BIAS` so drops/builds tend to pull motion- and
    impact-heavy Candidates and verses prefer face-bearing ones. The base
    `Candidate.score` is unchanged — the plan still records the original
    score the user can reason about; only the local pick order shifts.

    Returns `[(beat_t, picked_candidate), ...]` in cut order.
    """
    n_unique = len({c.source for c in candidates})
    n_slots = max(0, len(cut_points) - 1)
    effective_cap = source_cap
    # Build a per-source cap map. Default is `effective_cap` for any source
    # not present in `source_budgets`; when a budget IS present it's clamped
    # by `source_cap` so the user's hard ceiling always wins.
    per_source_cap: dict[str, int] = {}
    if source_budgets:
        for src, budget in source_budgets.items():
            per_source_cap[src] = max(1, min(int(budget), source_cap))

    # Legacy auto-bump path: only fires when no per-source budget is supplied
    # AND the caller is still at the strict-no-duplicates default. With F3
    # moment-budgets the planner sums the per-source budgets BEFORE allocate
    # is called, so this branch is effectively a no-op in the new pipeline —
    # we leave it in place for the small handful of tests/callers that drive
    # `allocate_candidates` directly with a flat `source_cap`.
    if (auto_bump_cap and source_cap == 1 and not source_budgets
            and n_unique > 0 and n_unique * source_cap < n_slots):
        # How many slots would stay unfilled if we DON'T bump? If the gap is
        # within tolerance, prefer the slightly-shorter plan to a repeated-clip
        # plan — F1 trades length for variety on the margin.
        deficit_slots = n_slots - n_unique * source_cap
        underfill_ratio = deficit_slots / n_slots
        if underfill_ratio <= AUTO_BUMP_UNDERFILL_TOLERANCE:
            log(f"  ! accepted slight underfill: {n_unique * source_cap} cuts "
                f"(would-be {n_slots}) — {underfill_ratio * 100:.1f}% short, "
                f"within {AUTO_BUMP_UNDERFILL_TOLERANCE * 100:.0f}% tolerance")
        else:
            # Round up: e.g. 64 sources, 75 slots, cap=1 → need cap=2.
            needed = -(-n_slots // n_unique)
            effective_cap = min(max(source_cap, needed), max_auto_cap)
            if effective_cap > source_cap:
                log(f"  ! bumped source_cap {source_cap} → {effective_cap} to fit "
                    f"{n_slots} cuts from {n_unique} unique sources")

    def _cap_for(src: str) -> int:
        return per_source_cap.get(src, effective_cap)

    # We keep a pool sorted by base score and re-rank per-anchor when
    # sections are in play. When sections are absent the inner loop falls
    # through to the original "highest base score wins" path.
    pool = sorted(candidates, key=lambda c: c.score, reverse=True)
    used_sources: dict[str, int] = {}
    picks: list[tuple[float, Candidate]] = []

    for i in range(len(cut_points) - 1):
        beat_t = cut_points[i]
        next_t = cut_points[i + 1]
        gap = next_t - beat_t
        if gap < MIN_CLIP_S:
            continue
        pick: Candidate | None = None
        if sections:
            kind = _section_for_time(beat_t, sections)
            # Local re-rank: walk the pool once, scoring with the bias. We
            # don't mutate `pool` so the next anchor sees the same order.
            best: tuple[float, Candidate] | None = None
            for c in pool:
                clip_len = c.end_s - c.start_s
                if clip_len < MIN_CLIP_S:
                    continue
                if used_sources.get(c.source, 0) >= _cap_for(c.source):
                    continue
                s = _section_picker_score(c, kind)
                if best is None or s > best[0]:
                    best = (s, c)
            if best is not None:
                pick = best[1]
        else:
            for c in pool:
                clip_len = c.end_s - c.start_s
                if clip_len < MIN_CLIP_S:
                    continue
                if used_sources.get(c.source, 0) >= _cap_for(c.source):
                    continue
                pick = c
                break
        if not pick:
            continue
        pool.remove(pick)
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


# Stretch-mode tunables (C3). When the planner produces a plan visibly
# shorter than `target_len`, we walk three levers in order — bump source_cap,
# stretch stills, then stretch the trailing beat slots — until the plan
# fills (or we've exhausted them). All three log a single line on use so the
# user can see why an edit ended up tighter or looser than expected.
STRETCH_FILL_RATIO = 0.85           # below this we engage stretch mode
STRETCH_MAX_SOURCE_CAP = 3          # ceiling for the +1 cap bump retry
                                    # (F1: 5 → 3; same-clip 4× is jarring)
STRETCH_MAX_STILL_DURATION_S = 4.0  # ceiling for per-entry still stretch
STRETCH_DEFAULT_STILL_DURATION_S = 2.5
STRETCH_TAIL_FACTOR = 1.5           # last-entries beat-slot stretch ceiling
STRETCH_TAIL_ENTRIES = 4            # how many trailing entries the lever 3
                                    # stretch is allowed to dilate

# Moment-budget tunables (F3). Replaces the legacy C4 "subset mode" pool-level
# top-N trim, which concentrated cuts on a handful of high-scoring sources and
# dropped the rest of the catalog entirely. The C4 approach was correct about
# "don't show the bottom of the score pile" but wrong about HOW to trim — a
# global top-N is blind to source provenance, so a 200-clip pool with one
# loud GoPro got reduced to that GoPro's 36 best windows and 0 from the
# other 60 cameras.
#
# Moment-budget mode instead asks: "for each source, what's the maximum
# number of *distinct* high-score moments it could plausibly contribute?"
# A 30s clip can hold ~3 separate beats of interest; a 3s Live Photo MOV
# holds exactly 1. We compute that per-source ceiling, then take the top-K
# candidates from each source by score. The result: every source gets a
# fair shot at the plan, but no single source can flood it.
SECONDS_PER_MOMENT = 10.0      # one distinct "moment" per 10s of footage
# Cap on distinct sub-clips a single source contributes. Default is 1: the
# observed UX problem is users seeing the same source repeat 4-5× in a single
# aftermovie. With one moment per source and slot-stretch (`C3`), a 60-source
# folder fills a 156s song at ~2.6s per cut, no repetition. Override via
# `AFTERMOVIE_MOMENTS_PER_SOURCE` env or `--moments-per-source N` CLI flag
# (passed through `cmd_score` / `cmd_auto`) when you want longer renders that
# tolerate inter-source repetition.
def _default_moments_per_source() -> int:
    raw = os.environ.get("AFTERMOVIE_MOMENTS_PER_SOURCE", "1").strip()
    try:
        n = int(raw)
    except ValueError:
        return 1
    return max(1, min(n, 8))
MAX_MOMENTS_PER_SOURCE = _default_moments_per_source()
SHORT_SOURCE_DURATION_S = 4.0  # stills + sub-4s Live Photo MOVs → budget=1

# F1 auto-bump underfill tolerance. When `allocate_candidates` would have to
# bump `source_cap` only to cover a few trailing beats, the lever instead
# accepts the slightly-shorter plan (logged). The threshold is "at most 20%
# of slots unfilled" — viewers will notice a missing 4 seconds less than they
# notice the same clip showing 4 times.
AUTO_BUMP_UNDERFILL_TOLERANCE = 0.20


def _compute_source_budgets(catalog: dict[str, Any],
                            candidates: list[Candidate],
                            *,
                            max_moments_per_source: int | None = None,
                            ) -> dict[str, int]:
    """Per-source moment budget — how many distinct windows from each source
    are eligible for the plan.

    For each source path present in `candidates`, looks up its
    `duration_s` in the catalog and computes:

        budget = max(1, ceil(duration_s / SECONDS_PER_MOMENT))

    capped at `MAX_MOMENTS_PER_SOURCE`. Sources shorter than
    `SHORT_SOURCE_DURATION_S` (stills + sub-4s Live Photo MOVs) always
    get exactly budget=1 — they hold one moment by construction.

    Sources that appear in the catalog but produced no Candidates (e.g.
    banned, suppressed by burst/visual-dup) are absent from the returned
    dict — there is no budget to spend on a source that has nothing left
    to offer.
    """
    by_source: dict[str, dict[str, Any]] = {
        c["path"]: c for c in catalog.get("clips", [])
    }
    cap = MAX_MOMENTS_PER_SOURCE if max_moments_per_source is None else max(1, max_moments_per_source)
    present_sources = {c.source for c in candidates}
    budgets: dict[str, int] = {}
    for src in present_sources:
        clip = by_source.get(src, {})
        duration = float(clip.get("duration_s", 0.0) or 0.0)
        if duration <= SHORT_SOURCE_DURATION_S:
            budgets[src] = 1
            continue
        # When the cap is 1 (default), every source contributes its single
        # best moment regardless of duration — no per-source ceiling math
        # needed.
        if cap <= 1:
            budgets[src] = 1
            continue
        n = int(math.ceil(duration / SECONDS_PER_MOMENT))
        budgets[src] = max(1, min(n, cap))
    return budgets


def _apply_moment_budget(candidates: list[Candidate],
                         budgets: dict[str, int]) -> list[Candidate]:
    """Take top-`budgets[source]` candidates from each source by score; drop
    the rest. Replaces the C4 pool-level subset trim.

    Order of the returned list is unspecified — the allocator re-sorts by
    score anyway. We preserve the relative order *within* a source so that
    when two candidates from the same source tie on score, the earlier one
    (lower `start_s`) wins, matching the legacy `sorted(..., reverse=True)`
    stable-sort tiebreak.
    """
    if not candidates:
        return candidates
    by_src: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_src.setdefault(c.source, []).append(c)
    kept: list[Candidate] = []
    for src, group in by_src.items():
        budget = budgets.get(src, 1)
        # Stable: ties broken by original index (which is `start_s` order
        # because `build_candidates` walks windows left-to-right).
        ranked = sorted(group, key=lambda c: c.score, reverse=True)
        kept.extend(ranked[:budget])
    return kept


def _format_top_source_for_log(budgets: dict[str, int],
                               catalog: dict[str, Any]) -> str:
    """Find the source contributing the largest budget and format
    `<filename> <duration>s` for the log line. Best-effort — returns a
    fallback string when the catalog can't surface a duration."""
    if not budgets:
        return "n/a"
    top_src = max(budgets, key=lambda s: budgets[s])
    by_source: dict[str, dict[str, Any]] = {
        c["path"]: c for c in catalog.get("clips", [])
    }
    duration = float(by_source.get(top_src, {}).get("duration_s", 0.0) or 0.0)
    name = os.path.basename(top_src) or top_src
    return f"{name} {duration:.1f}s"


def _plan_total_duration_s(plan_entries: list[dict[str, Any]]) -> float:
    return sum(float(e.get("out_duration_s", 0.0)) for e in plan_entries)


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

    Stretch mode (C3): when the resulting plan fills less than
    `STRETCH_FILL_RATIO × target_len`, three levers fire in order — bump
    `source_cap` (retry the allocator), stretch per-entry duration up to
    `STRETCH_MAX_STILL_DURATION_S`, and dilate the trailing entries' beat
    slots up to `STRETCH_TAIL_FACTOR×`. Each lever logs a single line.

    Moment-budget mode (F3, replaces C4 subset mode): every source in the
    catalog gets a per-source "moment budget" — `ceil(duration / 10s)`
    capped at 8 — and the candidate pool is trimmed to the top-K windows
    per source by score. This guarantees every source contributes to the
    plan (no more "25 out of 61 sources" pool concentration) while still
    keeping the highest-scoring windows from each. The allocator then uses
    each source's budget as its per-source repetition cap, with the
    user-supplied `source_cap` enforced as a hard ceiling.
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

    # F3 moment-budget pass — replaces the C4 pool-level top-N subset trim.
    # The old subset mode sorted ALL candidates globally by score and kept
    # the top N. That dropped low-scoring sources entirely (a "loud" GoPro
    # crowded out 60 other cameras' best moments). The new pass instead
    # asks "how many distinct moments can each source plausibly contribute?"
    # and keeps the top-K per source. The burst-suppress + visual-duplicate
    # passes above remain — those are PER-CLIP cleanups (same-camera burst,
    # same-look-everywhere), not pool reductions, so they compose cleanly
    # with the per-source budget.
    source_budgets = _compute_source_budgets(catalog, candidates)
    if source_budgets:
        budget_values = sorted(source_budgets.values())
        n_sources = len(budget_values)
        budget_sum = sum(budget_values)
        median = budget_values[n_sources // 2]
        max_budget = budget_values[-1]
        log(f"moment budget: {n_sources} sources, sum={budget_sum} "
            f"(median={median}, max={max_budget} from "
            f"{_format_top_source_for_log(source_budgets, catalog)})")
    candidates = _apply_moment_budget(candidates, source_budgets)

    # Section-aware allocation only fires for pace=auto. fast/medium/slow
    # are kept deliberately predictable per the IMPROVEMENT_PLAN.md spec:
    # "Preserve manual fast, medium, and slow modes for predictable behavior."
    alloc_sections = song.get("sections") if pace == "auto" else None
    picks = allocate_candidates(candidates, cut_points,
                                source_cap=source_cap,
                                sections=alloc_sections,
                                source_budgets=source_budgets)

    # Stretch mode lever 1 (C3): if the greedy allocator left obvious holes
    # (`< STRETCH_FILL_RATIO × n_slots` picked) AND the caller is still at
    # the strict-no-duplicates default (`source_cap == 1`), bump the cap one
    # notch and retry. Mirrors the existing `allocate_candidates` auto-bump
    # contract: callers who explicitly passed `source_cap >= 2` have already
    # decided "I want this much variety at most", so we don't over-rule
    # them — levers 2 (still stretch) and 3 (tail stretch) take over for
    # those callers and absorb the remaining deficit.
    n_slots = max(0, len(cut_points) - 1)
    if (n_slots > 0 and len(picks) < int(n_slots * STRETCH_FILL_RATIO)
            and source_cap == 1):
        new_cap = min(source_cap + 1, STRETCH_MAX_SOURCE_CAP)
        # Re-run the greedy walker with a slightly looser cap. We use the
        # same Candidate list — bursts/visual-dups/moment-budget already
        # pruned it. The per-source budgets still apply (clamped by
        # `new_cap`) so the retry can't suddenly let one source dominate.
        retry_picks = allocate_candidates(candidates, cut_points,
                                          source_cap=new_cap,
                                          sections=alloc_sections,
                                          source_budgets=source_budgets)
        if len(retry_picks) > len(picks):
            log(f"  ! stretch-mode: bumped source_cap {source_cap} → {new_cap} "
                f"to fill {target_len:.1f}s ({len(picks)} → {len(retry_picks)} "
                f"picks)")
            picks = retry_picks
            source_cap = new_cap

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

    # Stretch mode lever 2 (C3): the plan is *populated* but still falls
    # short of `target_len` (e.g. the allocator filled every slot but
    # cumulative `out_duration_s` < target_len because the beat anchors only
    # span the first part of the song). Grow per-entry duration uniformly up
    # to `STRETCH_MAX_STILL_DURATION_S` to make up the deficit.
    plan_total = _plan_total_duration_s(plan_entries)
    if (plan_entries
            and plan_total > 0
            and plan_total < target_len * STRETCH_FILL_RATIO):
        old_dur = plan_total / len(plan_entries)
        # Compute the per-entry duration we *want* to hit target; cap it at
        # the stretched-still ceiling so we don't park a single image for 8s.
        want_dur = target_len / len(plan_entries)
        new_dur = min(STRETCH_MAX_STILL_DURATION_S, want_dur)
        if new_dur > old_dur + 0.05:
            scale = new_dur / old_dur
            beat_t = float(plan_entries[0].get("beat_time_s", 0.0))
            for e in plan_entries:
                e["beat_time_s"] = beat_t
                e["out_duration_s"] = float(e["out_duration_s"]) * scale
                beat_t += float(e["out_duration_s"])
            log(f"  ! stretch-mode: stretched stills "
                f"{STRETCH_DEFAULT_STILL_DURATION_S:.1f}s → {new_dur:.1f}s "
                f"to fill {target_len:.1f}s")

    # Stretch mode lever 3 (C3): if we're still short, give the trailing
    # entries a bit more breathing room (up to `STRETCH_TAIL_FACTOR×` their
    # current beat-slot). Keeps the climax frames on screen long enough for
    # the song's outro to land without a hard cut into silence.
    plan_total = _plan_total_duration_s(plan_entries)
    deficit = target_len - plan_total
    if (plan_entries
            and deficit > 0.5
            and len(plan_entries) >= 1):
        tail_n = min(STRETCH_TAIL_ENTRIES, len(plan_entries))
        tail = plan_entries[-tail_n:]
        tail_total = sum(float(e["out_duration_s"]) for e in tail)
        if tail_total > 0:
            # Cap the multiplier at STRETCH_TAIL_FACTOR so we never balloon a
            # single entry past 1.5× its original beat slot.
            mult = min(STRETCH_TAIL_FACTOR,
                       1.0 + deficit / tail_total)
            old_tail_total = tail_total
            new_tail_total = tail_total * mult
            # Recompute beat_time_s monotonically so the renderer's timeline
            # math stays consistent — we touch only the tail's durations and
            # roll their start times forward.
            beat_t = float(plan_entries[-tail_n].get("beat_time_s", 0.0))
            for e in tail:
                e["beat_time_s"] = beat_t
                e["out_duration_s"] = float(e["out_duration_s"]) * mult
                beat_t += float(e["out_duration_s"])
            log(f"  ! stretch-mode: stretched tail {tail_n} entries "
                f"{old_tail_total:.1f}s → {new_tail_total:.1f}s "
                f"(×{mult:.2f}) to fill {target_len:.1f}s")

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

    # `max_length=None` means "no user override" → fill the full Song. The
    # 90s cap from before C1 used to live here; `DEFAULT_TARGET_LEN_S` is now
    # only a fallback ceiling for the case where the song's duration came
    # back unusable (analyze failure → 0/NaN).
    song_dur = float(song.get("duration_s") or 0.0)
    if args.max_length is None:
        target_len = song_dur if song_dur > 0 else float(DEFAULT_TARGET_LEN_S)
    else:
        target_len = min(
            song_dur if song_dur > 0 else float(DEFAULT_TARGET_LEN_S),
            float(args.max_length),
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
    # Coverage: how many source files actually contributed an entry vs. how
    # many were available in the catalog. The "3 had no in-range moments"
    # tail surfaces sources that fell out at burst-suppress / visual-dup /
    # moment-budget time so the user can see catalog → plan attrition.
    used_sources = {e["source"] for e in entries}
    total_sources = len({c.get("path") for c in catalog.get("clips", [])
                         if c.get("path")})
    unused = max(0, total_sources - len(used_sources))
    tail = f" ({unused} had no in-range moments)" if unused else ""
    log(f"built {len(entries)} cuts over {target_len:.1f}s — "
        f"{len(used_sources)}/{total_sources} sources used{tail}")

    # F1 diversity metric: one-line at-a-glance read on how many unique
    # sources back the plan and whether any are repeated. avg_repeats is
    # `cuts / sources`; max_repeats surfaces the worst-case clip reuse.
    if entries:
        per_source: dict[str, int] = {}
        for e in entries:
            per_source[e["source"]] = per_source.get(e["source"], 0) + 1
        n_unique = len(per_source)
        avg_repeats = len(entries) / max(n_unique, 1)
        max_repeats = max(per_source.values())
        log(f"  diversity: {len(entries)} cuts from {n_unique} unique sources "
            f"(avg {avg_repeats:.1f} repeats, max {max_repeats})")

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
