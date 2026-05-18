"""F1: diversity-aware subset trim.

Replaces the C4 "top-N by score" trim with a per-source round-robin so that
every Source contributes its best Candidate to the trimmed pool before any
Source contributes a second. Net effect: when the user's folder has 25
sources and we asked for 175 cuts, the planner doesn't repeat the same 5
high-scoring clips 4× each — it gives every source at least one slot first.

These tests exercise `_subset_filter_candidates` directly so the round-robin
contract is pinned independently of `allocate_candidates`' downstream
behavior.
"""
from __future__ import annotations

from aftermovie.types import Candidate
from aftermovie.score.scorer import _subset_filter_candidates


def _candidate(source: str, score: float, start: float = 0.0) -> Candidate:
    return Candidate(
        source=source,
        start_s=start,
        end_s=start + 2.0,
        score=score,
        reasons=[],
        src_fps=60.0,
    )


def _cut_points(n_slots: int, slot_s: float = 1.0) -> list[float]:
    """Build `n_slots + 1` cut points spaced `slot_s` apart."""
    return [i * slot_s for i in range(n_slots + 1)]


def test_round_robin_keeps_every_source_when_pool_is_large():
    """30 sources × 20 candidates each = 600 candidates. With a target of 60
    slots and SUBSET_KEEP_RATIO=1.2, keep ~72. Round-robin depth 2 across 30
    sources is 60; depth 3 would be 90; so we expect every source to
    contribute at least 2 candidates and the trim to land between 60 and 90.
    The key invariant is ALL 30 sources must survive — no source is dropped
    entirely just because its scores were lower."""
    candidates: list[Candidate] = []
    # Score offsets keep ordering deterministic per source.
    for s in range(30):
        for w in range(20):
            # Window-level score within a source decreases by `w`; between
            # sources, the baseline shifts by `s * 0.01` so source 0's best
            # candidate beats source 29's best candidate.
            candidates.append(
                _candidate(f"/src{s}.mp4",
                           score=100.0 - w + (29 - s) * 0.01,
                           start=float(w))
            )
    cut_points = _cut_points(60)  # n_slots = 60
    trimmed = _subset_filter_candidates(candidates, cut_points, target_len=60.0)

    # Subset mode must trigger (600 cand × 1s/slot = 600s pool capacity ≫
    # 60s × 1.5 trigger ratio).
    assert len(trimmed) < len(candidates), \
        f"trim didn't fire (got {len(trimmed)} from {len(candidates)})"
    # Every source represented — the headline F1 invariant.
    sources_kept = {c.source for c in trimmed}
    assert len(sources_kept) == 30, \
        f"round-robin dropped a source: kept {len(sources_kept)}/30"
    # Each source contributes EXACTLY 2 candidates (depth 2 of round-robin
    # for 30 sources = 60, which already meets the keep target of ~72 once
    # we round; the algorithm stops at depth 2 + a few from depth 3).
    per_source: dict[str, int] = {}
    for c in trimmed:
        per_source[c.source] = per_source.get(c.source, 0) + 1
    # At least 2 candidates per source; some may have a 3rd from the
    # partial depth-3 pass.
    assert all(v >= 2 for v in per_source.values()), \
        f"some sources got fewer than 2 candidates: {per_source}"
    # No source should jump ahead — round-robin enforces fairness.
    assert max(per_source.values()) - min(per_source.values()) <= 1, \
        f"round-robin asymmetric: {per_source}"


def test_round_robin_falls_back_to_best_score_when_sources_few():
    """5 sources × 100 candidates each = 500 candidates. Target 60 slots →
    keep ~72. Round-robin can only contribute 5 × depth, so once every
    source has been picked enough times, the top-up phase pulls the best
    remaining candidates from any source."""
    candidates: list[Candidate] = []
    for s in range(5):
        for w in range(100):
            candidates.append(
                _candidate(f"/src{s}.mp4",
                           score=200.0 - w - s * 0.001,
                           start=float(w))
            )
    cut_points = _cut_points(60)
    trimmed = _subset_filter_candidates(candidates, cut_points, target_len=60.0)

    assert len(trimmed) < len(candidates)
    # All 5 sources represented.
    sources_kept = {c.source for c in trimmed}
    assert sources_kept == {f"/src{s}.mp4" for s in range(5)}, \
        f"missing source(s): {sources_kept}"
    # Each source contributes ≈ keep / n_sources candidates.
    per_source: dict[str, int] = {}
    for c in trimmed:
        per_source[c.source] = per_source.get(c.source, 0) + 1
    # 72 / 5 ≈ 14-15 candidates per source.
    assert all(v >= 10 for v in per_source.values()), \
        f"a source got fewer than 10 candidates: {per_source}"


def test_subset_trim_skipped_when_pool_is_small():
    """Pool capacity within `SUBSET_TRIGGER_RATIO × target_len` is left
    intact — no trim, no log."""
    candidates = [_candidate(f"/src{i}.mp4", score=10.0) for i in range(40)]
    cut_points = _cut_points(60)  # 60s target, 40 candidates × 1s = 40s pool
    trimmed = _subset_filter_candidates(candidates, cut_points, target_len=60.0)
    assert trimmed is candidates or trimmed == candidates, \
        "small pool must pass through unchanged"


def test_round_robin_preserves_best_candidate_per_source():
    """Within each bucket the round-robin takes the source's TOP-scoring
    candidate first. Verify that for every source represented in the trim,
    the highest-scoring candidate from that source survived."""
    candidates: list[Candidate] = []
    for s in range(30):
        # Each source has one obvious winner and 19 also-rans.
        candidates.append(_candidate(f"/src{s}.mp4", score=500.0, start=0.0))
        for w in range(1, 20):
            candidates.append(_candidate(f"/src{s}.mp4", score=1.0,
                                          start=float(w)))
    cut_points = _cut_points(60)
    trimmed = _subset_filter_candidates(candidates, cut_points, target_len=60.0)

    # For every source that survived, the winning (score=500) candidate
    # must be among its trimmed candidates.
    by_source: dict[str, list[Candidate]] = {}
    for c in trimmed:
        by_source.setdefault(c.source, []).append(c)
    for src, cands in by_source.items():
        scores = sorted([c.score for c in cands], reverse=True)
        assert scores[0] == 500.0, \
            f"{src} kept lower-scoring candidates but dropped its winner: {scores}"
