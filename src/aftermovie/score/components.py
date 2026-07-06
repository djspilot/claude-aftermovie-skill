"""Registry of every score component the scorer may emit.

Each named signal that contributes to a Candidate's score is declared here
exactly once. The scorer threads `ScoreComponent` instances through `_add`
instead of bare string literals so the set of known components is a single
in-code source of truth — for the scorer, for tests, and for any future
"why this entry won?" UI that needs the vocabulary.

Interface
---------
The Module exposes:

- `ScoreComponent` — a frozen dataclass with `name`, `description`,
  `polarity` (`"+"` for positive contributions, `"-"` for penalties), and an
  optional `weight_hint` (rough magnitude the scorer applies, useful for
  tuning docs/UIs — NOT consumed by the scorer itself).
- One module-level `ScoreComponent` constant per known signal
  (`MOTION`, `AUDIO`, `HILIGHT_TAG`, `ACCL_JUMP`, `GPS_SPEED`, `FACE`,
  `BLURRY`, `POOR_EXPOSURE`, `USER_FAVORITE`). Importers reference these
  by attribute, not string.
- `is_known(name) -> bool` — predicate over the JSON-layer keys (the
  `Candidate.components` dict still maps `str → float` for back-compat).
- `iter_components() -> Iterable[ScoreComponent]` — every declared component
  in declaration order, for docs, CLI listings, and registry tests.
- `get(name) -> ScoreComponent` — raise `KeyError` for unknown names; the
  scorer's `_add` uses this internally to gate string-keyed dicts.

Invariants
----------
- Every `name` is unique.
- The string a `ScoreComponent.name` exposes IS the JSON key the scorer
  writes into `components` dicts — back-compat with on-disk `plan.json`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ScoreComponent:
    """A single named signal that may contribute to a Candidate's score.

    `name` is also the JSON key the scorer writes into the on-disk
    `Candidate.components` / `PlanEntry.components` dicts — renaming a
    component breaks back-compat with existing catalog.json / plan.json
    files, so don't.
    """
    name: str
    description: str
    polarity: str  # "+" for boosts, "-" for penalties
    weight_hint: float | None = None


# Positive signals — boosts.
MOTION = ScoreComponent(
    name="motion",
    description="Mean per-second motion energy across the window (scaled x1.5).",
    polarity="+",
    weight_hint=1.5,
)
AUDIO = ScoreComponent(
    name="audio",
    description="Mean per-second audio energy across the window (scaled x1.0).",
    polarity="+",
    weight_hint=1.0,
)
HILIGHT_TAG = ScoreComponent(
    name="hilight_tag",
    description="A GoPro HiLight tag (HMMT atom) lands inside the window.",
    polarity="+",
    weight_hint=10.0,
)
ACCL_JUMP = ScoreComponent(
    name="accl_jump",
    description="Peak GPMF accel magnitude in the window exceeds a jump threshold.",
    polarity="+",
    weight_hint=3.0,
)
GPS_SPEED = ScoreComponent(
    name="gps_speed",
    description="Window's GPS speed peaks at >=80% of the clip's overall max.",
    polarity="+",
    weight_hint=2.0,
)
FACE = ScoreComponent(
    name="face",
    description="At least one detected face bbox in the window.",
    polarity="+",
    weight_hint=0.5,
)
AUDIO_PEAK = ScoreComponent(
    name="audio_peak",
    description="Window's audio loudness spikes vs the clip's own baseline "
                "(>=85th percentile and loud in absolute terms) — cheers, "
                "impacts, crowd roars. Steady loudness never trips this.",
    polarity="+",
    weight_hint=1.5,
)
GYRO_SPIN = ScoreComponent(
    name="gyro_spin",
    description="Peak GPMF gyro magnitude in the window exceeds a fast-"
                "rotation threshold (spins, whips, tricks).",
    polarity="+",
    weight_hint=1.5,
)
USER_FAVORITE = ScoreComponent(
    name="user_favorite",
    description="Source is in the user's per-folder favorites preferences list.",
    polarity="+",
    weight_hint=2.0,
)

# Negative signals — penalties.
BLURRY = ScoreComponent(
    name="blurry",
    description="Window's mean sharpness sits in the clip's bottom-30th percentile.",
    polarity="-",
    weight_hint=-1.5,
)
POOR_EXPOSURE = ScoreComponent(
    name="poor_exposure",
    description="Window's mean exposure is crushed (<0.25) or blown (>0.85).",
    polarity="-",
    weight_hint=-1.5,
)


# Declaration order is the order `iter_components` walks and the order the
# CLI prints. Keep positive signals first, then penalties.
_ALL: tuple[ScoreComponent, ...] = (
    MOTION,
    AUDIO,
    HILIGHT_TAG,
    ACCL_JUMP,
    GPS_SPEED,
    AUDIO_PEAK,
    GYRO_SPIN,
    FACE,
    USER_FAVORITE,
    BLURRY,
    POOR_EXPOSURE,
)

_BY_NAME: dict[str, ScoreComponent] = {c.name: c for c in _ALL}
assert len(_BY_NAME) == len(_ALL), "duplicate ScoreComponent name in registry"


def iter_components() -> Iterable[ScoreComponent]:
    """Yield every registered ScoreComponent in declaration order."""
    return iter(_ALL)


def is_known(name: str) -> bool:
    """True iff `name` is a registered component key."""
    return name in _BY_NAME


def get(name: str) -> ScoreComponent:
    """Look up a ScoreComponent by its JSON-layer name. Raises KeyError
    for unknown names — `score_window`'s `_add` uses this to reject typos."""
    return _BY_NAME[name]


__all__ = [
    "ACCL_JUMP",
    "AUDIO",
    "AUDIO_PEAK",
    "BLURRY",
    "FACE",
    "GPS_SPEED",
    "GYRO_SPIN",
    "HILIGHT_TAG",
    "MOTION",
    "POOR_EXPOSURE",
    "ScoreComponent",
    "USER_FAVORITE",
    "get",
    "is_known",
    "iter_components",
]
