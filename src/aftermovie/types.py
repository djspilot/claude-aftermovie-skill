"""Shared dataclasses used by analyze, score, render, and the MCP server."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A single candidate sub-clip (1-5 seconds within a source file)."""
    source: str
    start_s: float
    end_s: float
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    src_fps: float = 30.0
    is_short: bool = False
    # Named contributions that sum to `score`. Each named signal (motion,
    # audio, hilight_tag, blurry penalty, ...) writes its delta here so the
    # GUI / "why this won?" inspector can break down WHY a candidate scored
    # what it did, not just the total. Zero-valued signals are omitted.
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class ClipInfo:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    has_gpmf: bool
    hilight_tags_ms: list[int]
    motion_energy: list[float]
    audio_energy: list[float]
    voice_energy: list[float]
    accl_peaks: list[float]
    gps_speed: list[float]
    is_short_form: bool
    # Per-second peak GPMF gyro magnitude (rad/s); empty when no telemetry.
    gyro_peaks: list[float] = field(default_factory=list)
    captured_at: float | None = None
    face_bboxes: list[dict | None] = field(default_factory=list)
    sharpness_per_s: list[float] = field(default_factory=list)
    exposure_per_s: list[float] = field(default_factory=list)
    # Perceptual hash of a representative frame (8x8 dHash, 16-char hex) or
    # None if hashing failed / deps unavailable. See analyze/duplicates.py.
    phash: str | None = None
    # L2-normalized semantic embedding of a representative frame (MediaPipe
    # ImageEmbedder), or None when the optional dep/model is unavailable.
    embedding: list[float] | None = None
    # Visual-duplicate cluster id assigned at the end of analyze; None means
    # "singleton" (no near-twins in this folder). The scorer uses this to
    # keep only the highest-scoring candidate from each cluster.
    duplicate_group: str | None = None


@dataclass
class PlanEntry:
    source: str
    start_s: float
    end_s: float
    out_duration_s: float
    speed: float
    beat_time_s: float
    score: float
    reasons: list[str] = field(default_factory=list)
    # Per-signal score breakdown forwarded from the Candidate that won this
    # slot. Sum equals `score` within float epsilon. Plan.json files written
    # before this field existed are tolerated by `from_dict`.
    components: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanEntry":
        return cls(
            source=d["source"],
            start_s=float(d["start_s"]),
            end_s=float(d["end_s"]),
            out_duration_s=float(d.get("out_duration_s", d["end_s"] - d["start_s"])),
            speed=float(d.get("speed", 1.0)),
            beat_time_s=float(d.get("beat_time_s", 0.0)),
            score=float(d.get("score", 0.0)),
            reasons=list(d.get("reasons", [])),
            components={k: float(v) for k, v in (d.get("components") or {}).items()},
        )
