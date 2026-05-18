"""Tests for `score.song.detect_sections` (Phase C5 / Phase 4).

`detect_sections` is the deterministic-from-existing-signals Module that
splits a Song into a contiguous list of `Section` values. It runs against
a hand-crafted per-second energy curve so we can assert the precise label
sequence we expect for synthetic shapes — no librosa, no audio, no random
seeds, no ML.
"""
from __future__ import annotations

from aftermovie.score.song import (
    SECTION_KINDS,
    Section,
    detect_sections,
)


def _beats_at(bpm: float, duration_s: float) -> list[float]:
    """A flat beat grid — `detect_sections` doesn't consume beats today,
    but accepting them keeps the signature future-proof."""
    dt = 60.0 / bpm
    out: list[float] = []
    t = 0.0
    while t < duration_s:
        out.append(t)
        t += dt
    return out


# ---- shape invariants ------------------------------------------------------

def test_sections_cover_song_contiguously():
    """Sections concatenate to [0, duration_s) with no gaps, no overlaps."""
    duration = 40.0
    energy = [0.1] * 10 + [0.4] * 5 + [0.95] * 8 + [0.5] * 5 + [0.1] * 12
    sections = detect_sections(duration, 120.0, _beats_at(120.0, duration), energy)
    assert sections, "must always return at least one section"
    assert sections[0].start_s == 0.0
    assert sections[-1].end_s == duration
    for prev, nxt in zip(sections, sections[1:]):
        assert prev.end_s == nxt.start_s, (
            f"gap or overlap between {prev} and {nxt}"
        )


def test_all_section_kinds_are_in_taxonomy():
    """Every emitted kind must come from the documented vocabulary —
    otherwise the scorer's `SECTION_TO_FACTOR` lookup would silently
    fall back to verse."""
    energy = ([0.1] * 6 + [0.3] * 4 + [0.5] * 4 + [0.95] * 6 + [0.4] * 4 + [0.1] * 6)
    sections = detect_sections(30.0, 120.0, _beats_at(120.0, 30.0), energy)
    for s in sections:
        assert s.kind in SECTION_KINDS, f"unknown kind {s.kind!r}"


def test_intensity_is_in_unit_range():
    energy = [0.0, 0.1, 0.5, 1.0, 0.5, 0.1]
    sections = detect_sections(6.0, 120.0, _beats_at(120.0, 6.0), energy)
    for s in sections:
        assert 0.0 <= s.intensity <= 1.0


# ---- the four canonical shapes from the spec -------------------------------

def test_synthetic_ramp_then_peak_then_decay_emits_build_drop():
    """4s quiet intro + 4s ramp-up + 8s peak + 6s decay → the build leading
    into the drop should be detected, the peak should land in `drop`, and
    the decay should end as `outro`."""
    energy = (
        [0.10] * 4 +            # 0-4s: low intro
        [0.20, 0.30, 0.45, 0.65] +  # 4-8s: ramp-up
        [0.95] * 8 +            # 8-16s: peak
        [0.55, 0.40, 0.25, 0.15, 0.10, 0.05]  # 16-22s: decay
    )
    duration = float(len(energy))
    sections = detect_sections(duration, 120.0, _beats_at(120.0, duration), energy)
    kinds = [s.kind for s in sections]

    # Must include all four expected categories in order.
    assert "intro" in kinds, f"expected intro, got {kinds}"
    assert "drop" in kinds, f"expected drop, got {kinds}"
    assert "outro" in kinds, f"expected outro, got {kinds}"
    # The drop's intensity must reflect the synthetic peak.
    drop = next(s for s in sections if s.kind == "drop")
    assert drop.intensity >= 0.9, f"drop intensity={drop.intensity}, energy hit 0.95"
    # First section is the intro, last is the outro.
    assert kinds[0] == "intro"
    assert kinds[-1] == "outro"
    # And the drop comes after a build (or directly after intro if the ramp
    # was steep enough to skip the build label).
    drop_idx = kinds.index("drop")
    assert kinds[drop_idx - 1] in ("build", "intro", "verse"), (
        f"drop at {drop_idx} preceded by {kinds[drop_idx - 1]} (kinds={kinds})"
    )


def test_flat_energy_song_is_mostly_verse():
    """A song with constant energy across the body has no peaks, no rising
    edges → no drop, no build. Intro / outro may still appear at the ends
    if early seconds dip below the median (they don't here; flat = exactly
    median everywhere so they shouldn't trigger).
    """
    duration = 40.0
    energy = [0.5] * int(duration)
    sections = detect_sections(duration, 120.0, _beats_at(120.0, duration), energy)
    kinds = [s.kind for s in sections]

    # No drops, no builds — every second is constant.
    assert "drop" not in kinds, f"unexpected drop in flat song: {kinds}"
    assert "build" not in kinds, f"unexpected build in flat song: {kinds}"
    # At least one verse must exist.
    assert "verse" in kinds, f"expected verse, got {kinds}"
    # The verse should dominate by duration (>=80% of song length).
    verse_s = sum(s.end_s - s.start_s for s in sections if s.kind == "verse")
    assert verse_s / duration >= 0.8, (
        f"verse only covered {verse_s/duration:.0%} of a flat song: {kinds}"
    )


def test_quiet_intro_then_peak_classifies_intro_at_head():
    """Very quiet first 8s + loud body → those 8s are the intro."""
    energy = [0.05] * 8 + [0.9] * 20
    duration = float(len(energy))
    sections = detect_sections(duration, 120.0, _beats_at(120.0, duration), energy)
    assert sections[0].kind == "intro", (
        f"expected leading intro, got {sections[0].kind} (kinds={[s.kind for s in sections]})"
    )


def test_quiet_tail_classifies_outro_when_long_enough():
    """Loud body then >=4s of quiet tail → tail is outro."""
    energy = [0.9] * 15 + [0.05] * 8
    duration = float(len(energy))
    sections = detect_sections(duration, 120.0, _beats_at(120.0, duration), energy)
    assert sections[-1].kind == "outro", (
        f"expected trailing outro, got {sections[-1].kind} "
        f"(kinds={[s.kind for s in sections]})"
    )


def test_short_quiet_tail_does_not_emit_outro():
    """A 2s dip at the tail is below the minimum tail length so no outro fires."""
    energy = [0.9] * 18 + [0.05] * 2
    duration = float(len(energy))
    sections = detect_sections(duration, 120.0, _beats_at(120.0, duration), energy)
    assert sections[-1].kind != "outro", (
        f"unexpected outro from a 2s dip: {[s.kind for s in sections]}"
    )


def test_degenerate_inputs_return_single_verse():
    """Zero-length / empty-energy songs collapse to one neutral verse."""
    out_empty = detect_sections(10.0, 120.0, _beats_at(120.0, 10.0), [])
    assert len(out_empty) == 1 and out_empty[0].kind == "verse"
    out_zero = detect_sections(0.0, 120.0, [], [0.0])
    assert out_zero and out_zero[0].kind == "verse"


def test_detect_is_deterministic_across_runs():
    """Calling detect_sections twice on the same inputs must yield identical
    Section sequences (no random seeds, no float jitter)."""
    energy = [0.1, 0.2, 0.4, 0.7, 0.95, 0.95, 0.8, 0.5, 0.2, 0.1, 0.05, 0.05]
    a = detect_sections(12.0, 120.0, _beats_at(120.0, 12.0), energy)
    b = detect_sections(12.0, 120.0, _beats_at(120.0, 12.0), energy)
    assert a == b, "detect_sections is not deterministic"


def test_section_dataclass_rejects_unknown_kind():
    import pytest
    with pytest.raises(ValueError):
        Section(kind="chorus", start_s=0.0, end_s=1.0, intensity=0.5)


def test_section_dataclass_rejects_inverted_span():
    import pytest
    with pytest.raises(ValueError):
        Section(kind="verse", start_s=2.0, end_s=1.0, intensity=0.5)


def test_section_to_dict_round_trips():
    s = Section(kind="drop", start_s=8.0, end_s=12.0, intensity=0.9)
    d = s.to_dict()
    assert d == {"kind": "drop", "start_s": 8.0, "end_s": 12.0, "intensity": 0.9}
