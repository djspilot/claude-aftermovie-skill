"""Contract tests for the ScoreComponent registry Module.

Pins three properties of the registry seam:

1. Every name the scorer's `_add` writes into a `Candidate.components` dict
   is registered. Tests this by AST-scanning `_add(...)` call sites and by
   running the scorer through every code path that emits a component.
2. The registry exposes at least the 9 current named signals — guards
   against accidental deletion of an entry that on-disk plan.json files
   already reference.
3. Smuggling an unknown ScoreComponent into `_add` raises — the Interface
   does not silently accept arbitrary strings the way the old `_add(str, ...)`
   helper did.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from aftermovie.score import components as sc
from aftermovie.score import scorer as scorer_mod
from aftermovie.score.components import ScoreComponent
from aftermovie.score.scorer import build_candidates, score_window


# ----- 1. Registry surface --------------------------------------------------


def test_registry_exposes_all_nine_current_components():
    """The 9 signals scattered across the old scorer must all be declared.

    A new signal is fine; deleting one is a back-compat break for on-disk
    plan.json files written before the deletion."""
    expected = {
        "motion", "audio", "hilight_tag", "accl_jump", "gps_speed",
        "face", "user_favorite", "blurry", "poor_exposure",
    }
    names = {c.name for c in sc.iter_components()}
    missing = expected - names
    assert not missing, f"registry is missing required components: {missing}"


def test_iter_components_yields_unique_score_component_instances():
    """`iter_components` must never repeat a name (the registry's primary
    key) and must only yield ScoreComponent instances."""
    seen: set[str] = set()
    for c in sc.iter_components():
        assert isinstance(c, ScoreComponent)
        assert c.name not in seen, f"duplicate component name {c.name!r}"
        seen.add(c.name)
        assert c.polarity in ("+", "-"), \
            f"component {c.name!r} has unexpected polarity {c.polarity!r}"
        assert c.description.strip(), \
            f"component {c.name!r} is missing a description"


def test_is_known_and_get_agree():
    """`is_known` must answer truthfully for every iterated component, and
    `get` must round-trip the exact instance."""
    for c in sc.iter_components():
        assert sc.is_known(c.name)
        assert sc.get(c.name) is c
    assert not sc.is_known("definitely_not_a_real_signal")
    with pytest.raises(KeyError):
        sc.get("definitely_not_a_real_signal")


# ----- 2. Every `_add` call uses a registered ScoreComponent ---------------


def _add_call_argument_names() -> set[str]:
    """Walk `score_window` source AST and collect the attribute name passed
    to every `_add(sc.<NAME>, ...)` call.

    Catches the "someone added a new branch that calls `_add(sc.NEW, ...)`
    without declaring NEW in components.py" regression at test time."""
    src = inspect.getsource(score_window)
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match `_add(<first_arg>, <delta>)` calls only.
        if not (isinstance(node.func, ast.Name) and node.func.id == "_add"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        # We expect `sc.MOTION`, `sc.AUDIO`, ... — an Attribute access on the
        # module alias bound at the top of scorer.py.
        if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
            if arg.value.id == "sc":
                names.add(arg.attr)
    return names


def test_every_add_call_in_score_window_uses_a_registered_component():
    """Every `_add(sc.X, ...)` in `score_window` must reference a registry
    constant whose `.name` is also registered."""
    add_names = _add_call_argument_names()
    assert add_names, "AST scan found no _add(sc.X, ...) calls — check the scanner"
    for attr in add_names:
        component = getattr(sc, attr, None)
        assert isinstance(component, ScoreComponent), (
            f"scorer._add references sc.{attr}, which is not a ScoreComponent "
            f"in aftermovie.score.components"
        )
        assert sc.is_known(component.name), (
            f"scorer._add references sc.{attr} (name={component.name!r}) "
            f"which is not registered via iter_components()"
        )


def test_build_candidates_only_emits_registered_component_keys():
    """End-to-end: every key that lands in a Candidate.components dict —
    including the favorited path that goes through `build_candidates`, not
    `score_window` — must be a registered name."""
    clip = {
        "path": "/all.mp4",
        "duration_s": 8.0,
        "fps": 60.0,
        "width": 1920,
        "height": 1080,
        "has_gpmf": False,
        "hilight_tags_ms": [2500],
        "motion_energy": [0.8] * 8,
        "audio_energy": [0.9] * 8,
        "accl_peaks": [10.0, 10.0, 18.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "gps_speed": [5.0, 5.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "is_short_form": False,
        "face_bboxes": [None, None, {"x": 0, "y": 0, "w": 10, "h": 10},
                        {"x": 0, "y": 0, "w": 10, "h": 10},
                        None, None, None, None],
        "sharpness_per_s": [0.9, 0.8, 0.1, 0.1, 0.85, 0.95, 0.9, 0.88],
        "exposure_per_s": [0.5, 0.5, 0.95, 0.92, 0.5, 0.5, 0.5, 0.5],
    }
    candidates = build_candidates({"clips": [clip]},
                                  preferences={"favorited": ["/all.mp4"]})
    seen: set[str] = set()
    for cand in candidates:
        for key in cand.components:
            assert sc.is_known(key), (
                f"Candidate.components key {key!r} is not registered in "
                f"aftermovie.score.components"
            )
            seen.add(key)
    # Sanity: the synthetic clip exercises most signals plus the favorite
    # boost — make sure the test actually saw them rather than passing
    # trivially on an empty dict.
    assert "user_favorite" in seen
    assert "hilight_tag" in seen
    assert "motion" in seen


# ----- 3. Unknown components are rejected -----------------------------------


def test_score_window_rejects_unknown_score_component():
    """Patching `_add` to be called with a non-registry ScoreComponent must
    raise. Belt-and-braces against in-tree drift where someone constructs
    a `ScoreComponent` ad-hoc instead of declaring it in `components.py`."""
    # The `_add` helper is a closure inside `score_window`, so we can't grab
    # it directly. Build a fake ScoreComponent (instance, but not registered)
    # and patch `sc.is_known` to confirm the guard fires — easier than
    # rewriting score_window for a single test.
    fake = ScoreComponent(name="not_a_real_signal", description="x", polarity="+")
    assert not sc.is_known(fake.name), \
        "test setup invariant: fake component must not be registered"

    # Monkey-patch sc.MOTION temporarily so the very first `_add` call inside
    # `score_window` hits an unregistered ScoreComponent. We restore it in
    # `finally` so other tests aren't affected.
    original = scorer_mod.sc.MOTION
    scorer_mod.sc.MOTION = fake  # type: ignore[misc]
    try:
        clip = {
            "path": "/m.mp4",
            "duration_s": 4.0,
            "motion_energy": [0.5] * 4,
            "audio_energy": [0.0] * 4,
            "accl_peaks": [],
            "gps_speed": [],
            "hilight_tags_ms": [],
            "is_short_form": False,
        }
        with pytest.raises(ValueError, match="unknown ScoreComponent"):
            score_window(clip, 0, 2)
    finally:
        scorer_mod.sc.MOTION = original  # type: ignore[misc]
