"""Tests for the per-folder `.aftermovie-preferences.json` sidecar.

The sidecar lives next to `.aftermovie-selection.json` and captures the
GUI's longer-lived likes/bans/pins. These tests cover the pure helpers in
`analyze/preferences.py` — round-trip, lookups, and missing-file fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

from aftermovie.analyze.preferences import (
    PREFERENCES_FILENAME,
    PREFERENCES_VERSION,
    clear_cache,
    is_banned,
    is_favorited,
    load_preferences,
    preferences_path,
    save_preferences,
)


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    """`save_preferences` writes JSON, `load_preferences` reads it back intact."""
    clear_cache()
    fav = [str(tmp_path / "good.mp4"), str(tmp_path / "great.mov")]
    ban = [str(tmp_path / "bad.mp4")]
    pinned = ["entry-id-1", "entry-id-2"]
    out = save_preferences(tmp_path, {
        "favorited": fav,
        "banned": ban,
        "pinned_entries": pinned,
    })
    # Sidecar lives at the expected path.
    assert out == preferences_path(tmp_path).resolve()
    assert out.is_file()

    # On-disk payload has the documented shape.
    payload = json.loads(out.read_text())
    assert payload["favorited"] == fav
    assert payload["banned"] == ban
    assert payload["pinned_entries"] == pinned
    assert payload["generated_by"] == "aftermovie-select"
    assert payload["version"] == PREFERENCES_VERSION

    # Round-trip through load_preferences yields the same data (sorted).
    loaded = load_preferences(tmp_path)
    assert sorted(loaded["favorited"]) == sorted(fav)
    assert sorted(loaded["banned"]) == sorted(ban)
    assert sorted(loaded["pinned_entries"]) == sorted(pinned)


def test_is_favorited_and_is_banned_honor_lists(tmp_path: Path) -> None:
    """`is_favorited` / `is_banned` reflect the saved lists; other paths are clear."""
    clear_cache()
    fav_path = (tmp_path / "loved.mp4").resolve()
    ban_path = (tmp_path / "hated.mp4").resolve()
    other = (tmp_path / "neutral.mp4").resolve()
    save_preferences(tmp_path, {
        "favorited": [str(fav_path)],
        "banned": [str(ban_path)],
    })

    assert is_favorited(tmp_path, fav_path) is True
    assert is_banned(tmp_path, ban_path) is True
    # Cross-check: favorited is not banned, banned is not favorited.
    assert is_banned(tmp_path, fav_path) is False
    assert is_favorited(tmp_path, ban_path) is False
    # Untracked paths come back clean.
    assert is_favorited(tmp_path, other) is False
    assert is_banned(tmp_path, other) is False


def test_missing_sidecar_returns_empty_defaults(tmp_path: Path) -> None:
    """No sidecar on disk → load_preferences returns empty lists, no exception."""
    clear_cache()
    # Sanity: no sidecar exists.
    assert not (tmp_path / PREFERENCES_FILENAME).exists()
    prefs = load_preferences(tmp_path)
    assert prefs == {"favorited": [], "banned": [], "pinned_entries": []}
    # Lookups stay safe too.
    assert is_favorited(tmp_path, tmp_path / "anything.mp4") is False
    assert is_banned(tmp_path, tmp_path / "anything.mp4") is False


def test_save_dedupes_and_drops_non_strings(tmp_path: Path) -> None:
    """`save_preferences` de-duplicates entries and ignores non-string items."""
    clear_cache()
    save_preferences(tmp_path, {
        "favorited": ["a.mp4", "a.mp4", "b.mp4", 42, None],  # dup + junk
        "banned": ["x.mp4"],
    })
    payload = json.loads((tmp_path / PREFERENCES_FILENAME).read_text())
    assert payload["favorited"] == ["a.mp4", "b.mp4"]
    assert payload["banned"] == ["x.mp4"]
    # pinned_entries defaults to [] when absent from the input.
    assert payload["pinned_entries"] == []


def test_malformed_sidecar_falls_back_to_empty(tmp_path: Path) -> None:
    """Garbage JSON in the sidecar must not raise — caller sees empty defaults."""
    clear_cache()
    (tmp_path / PREFERENCES_FILENAME).write_text("{this is not json")
    prefs = load_preferences(tmp_path)
    assert prefs == {"favorited": [], "banned": [], "pinned_entries": []}
