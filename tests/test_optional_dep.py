"""Tests for the shared optional-dependency Module.

The Module owns the truth-of-availability for every recoverable third-party
dep the analyzers consume. These tests pin its three invariants:

  1. A successful import yields the module + ``available is True``.
  2. A failed import yields ``available is False`` AND triggers exactly one
     warning log on the first ``.require()`` (none on subsequent calls).
  3. The analyzers' own ``*_available()`` predicates reflect the shared
     state — they no longer carry private ``_WARNED`` flags.
"""
from __future__ import annotations

import importlib

import pytest

from aftermovie import optional_dep


@pytest.fixture(autouse=True)
def _clean_registry():
    """Wipe the get-or-create cache so tests don't see each other's state."""
    optional_dep._reset_for_tests()
    yield
    optional_dep._reset_for_tests()


def test_successful_import_yields_module_and_available_true():
    dep = optional_dep.optional_import("json", warning="!")
    assert dep.available is True
    assert bool(dep) is True
    assert dep.module is not None
    # The module returned must be the same one importlib hands out.
    assert dep.module is importlib.import_module("json")


def test_failed_import_is_not_available_and_warns_exactly_once(caplog, capsys):
    dep = optional_dep.optional_import(
        "definitely_not_a_real_module_xyz",
        warning="! synthetic dep missing — feature skipped",
    )
    assert dep.available is False
    assert bool(dep) is False
    assert dep.module is None

    # First .require() logs through aftermovie.ffmpeg_cmd.log → stderr.
    assert dep.require() is None
    err1 = capsys.readouterr().err
    assert "synthetic dep missing" in err1

    # Second + third calls must be silent.
    assert dep.require() is None
    assert dep.require() is None
    err2 = capsys.readouterr().err
    assert err2 == ""


def test_optional_import_is_get_or_create_so_warning_state_is_shared(capsys):
    """Two callers asking for the same dep share one warn-once instance."""
    first = optional_dep.optional_import("not_a_real_pkg_abc", warning="! first message")
    second = optional_dep.optional_import("not_a_real_pkg_abc", warning="! second message")
    assert first is second

    first.require()
    second.require()
    err = capsys.readouterr().err
    # Exactly one log line total — and it's the first registered warning.
    assert err.count("\n") == 1
    assert "first message" in err
    assert "second message" not in err


def test_optional_command_present_reports_available():
    dep = optional_dep.optional_command(
        "sh",
        warning="! sh missing — that would be wild",
    )
    assert dep.available is True
    assert bool(dep) is True
    assert dep.path is not None
    assert dep.path.endswith("/sh") or dep.path == "sh"


def test_optional_command_missing_warns_exactly_once(capsys):
    dep = optional_dep.optional_command(
        "definitely_not_a_real_command_xyz",
        warning="! synthetic CLI missing — feature skipped",
    )
    assert dep.available is False
    assert bool(dep) is False
    assert dep.path is None

    assert dep.require() is None
    err1 = capsys.readouterr().err
    assert "synthetic CLI missing" in err1

    assert dep.require() is None
    assert dep.require() is None
    err2 = capsys.readouterr().err
    assert err2 == ""


# ---------------------------------------------------------------------------
# Analyzer-side: each *_available() predicate reflects the optional_dep state.
# ---------------------------------------------------------------------------


def test_quality_available_reflects_cv2_optional_dep(monkeypatch):
    from aftermovie.analyze import quality

    monkeypatch.setattr(quality._CV2, "module", None)
    assert quality.available() is False
    monkeypatch.setattr(quality._CV2, "module", object())
    assert quality.available() is True


def test_faces_available_reflects_mediapipe_optional_dep(monkeypatch, tmp_path):
    from aftermovie.analyze import faces

    # Stub out the on-disk model check so we're only measuring the dep flip.
    monkeypatch.setattr(
        faces, "models_dir", lambda: tmp_path, raising=True,
    )
    model = tmp_path / "blaze_face_short_range.tflite"
    model.write_bytes(b"stub")

    monkeypatch.setattr(faces._MEDIAPIPE, "module", None)
    assert faces.available() is False
    monkeypatch.setattr(faces._MEDIAPIPE, "module", object())
    assert faces.available() is True


def test_live_photo_exiftool_available_reflects_optional_dep(monkeypatch):
    from aftermovie.analyze import live_photo

    monkeypatch.setattr(live_photo._EXIFTOOL, "path", None)
    assert live_photo.exiftool_available() is False
    monkeypatch.setattr(live_photo._EXIFTOOL, "path", "/usr/local/bin/exiftool")
    assert live_photo.exiftool_available() is True


def test_duplicates_compute_phash_returns_none_when_pil_missing(monkeypatch, tmp_path):
    """The PIL gate routes through the shared OptionalImport handle."""
    from aftermovie.analyze import duplicates

    monkeypatch.setattr(duplicates._PIL, "module", None)
    monkeypatch.setattr(duplicates._PIL, "_warned", False)
    target = tmp_path / "x.jpg"
    target.write_bytes(b"\x00")
    assert duplicates.compute_phash(target) is None
