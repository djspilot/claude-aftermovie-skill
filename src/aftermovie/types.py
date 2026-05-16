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
    accl_peaks: list[float]
    gps_speed: list[float]
    is_short_form: bool
    face_bboxes: list[dict | None] = field(default_factory=list)


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
        )
