"""Tests for `SidecarStore` — the shared seam under preferences + selection.

`SidecarStore` owns the mtime-aware cache, atomic write-then-rename, and
malformed-JSON recovery used by both `analyze/selection.py` and
`analyze/preferences.py`. Adapter-level tests (`test_preferences.py`,
`test_select.py`) cover the projections; these tests cover the seam
itself — the Interface a third sidecar would target without having to
re-discover the contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aftermovie.analyze.sidecar import SidecarStore, clear_all_caches

FILENAME = ".aftermovie-fake.json"
DEFAULTS: dict[str, object] = {"items": [], "version": 1}


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    """Each test starts with an empty cache + clean malformed-warning state."""
    clear_all_caches()


def _store(folder: Path) -> SidecarStore:
    return SidecarStore(folder, FILENAME, DEFAULTS)


def test_round_trip_read_write(tmp_path: Path) -> None:
    """`write(payload)` followed by `read()` returns the payload verbatim."""
    store = _store(tmp_path)
    payload = {"items": ["a.mp4", "b.mp4"], "version": 1}

    out = store.write(payload)

    assert out == (tmp_path / FILENAME).resolve()
    assert out.is_file()
    # The on-disk JSON is the payload we handed in.
    assert json.loads(out.read_text()) == payload
    # `read()` returns the same data; `path()` matches.
    assert store.read() == payload
    assert store.path() == tmp_path / FILENAME


def test_read_missing_returns_defaults_copy(tmp_path: Path) -> None:
    """No sidecar on disk → defaults, and the returned dict is a fresh copy."""
    store = _store(tmp_path)

    first = store.read()
    assert first == DEFAULTS

    # Mutating the returned dict must not contaminate the next read.
    first["items"].append("intruder")
    assert store.read() == DEFAULTS


def test_cache_hit_skips_disk_when_mtime_unchanged(tmp_path: Path) -> None:
    """Second `read()` with the same mtime parses zero bytes from disk."""
    store = _store(tmp_path)
    store.write({"items": ["x"], "version": 1})

    # Prime the cache.
    assert store.read() == {"items": ["x"], "version": 1}

    # If the cache is honored, `read_text` should NOT be called again.
    real_read_text = Path.read_text
    calls: list[Path] = []

    def _spy(self: Path, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        calls.append(self)
        return real_read_text(self, *args, **kwargs)

    with patch.object(Path, "read_text", _spy):
        assert store.read() == {"items": ["x"], "version": 1}

    sidecar = tmp_path / FILENAME
    assert sidecar not in calls, (
        "expected cached read to skip disk, but read_text was called for the sidecar"
    )


def test_cache_miss_after_mtime_bump(tmp_path: Path) -> None:
    """Touching the file (new mtime) forces the next `read()` to reparse."""
    store = _store(tmp_path)
    store.write({"items": ["v1"], "version": 1})
    assert store.read() == {"items": ["v1"], "version": 1}

    sidecar = tmp_path / FILENAME
    # Rewrite the file out-of-band so the cache key still exists but the
    # on-disk mtime is newer. Bump mtime forward by a full second to defeat
    # filesystems with coarse mtime granularity.
    sidecar.write_text(json.dumps({"items": ["v2"], "version": 1}))
    st = sidecar.stat()
    future = st.st_mtime + 1.0
    os.utime(sidecar, (future, future))

    assert store.read() == {"items": ["v2"], "version": 1}


def test_malformed_json_returns_defaults_and_warns_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Garbage JSON → defaults, with exactly one stderr line per (folder, file)."""
    sidecar = tmp_path / FILENAME
    sidecar.write_text("{not json at all")

    store = _store(tmp_path)

    # Three reads of the same malformed file…
    for _ in range(3):
        assert store.read() == DEFAULTS

    captured = capsys.readouterr()
    # …yield exactly one warning on stderr.
    warnings = [
        line for line in captured.err.splitlines() if "malformed sidecar" in line
    ]
    assert len(warnings) == 1, (
        f"expected exactly one malformed-sidecar warning, got: {warnings}"
    )
    assert FILENAME in warnings[0]


def test_non_dict_root_falls_back_to_defaults(tmp_path: Path) -> None:
    """A JSON list (valid JSON, wrong shape) is treated as malformed."""
    (tmp_path / FILENAME).write_text(json.dumps(["just", "a", "list"]))
    store = _store(tmp_path)

    assert store.read() == DEFAULTS


def test_atomic_write_does_not_leave_partial_on_failure(
    tmp_path: Path,
) -> None:
    """If `replace()` fails, no `.tmp` clobbers the previous good sidecar."""
    store = _store(tmp_path)
    sidecar = tmp_path / FILENAME
    store.write({"items": ["good"], "version": 1})
    good_bytes = sidecar.read_bytes()

    # Simulate the rename failing partway through. The previous good file
    # must still be on disk and parseable — that's what "atomic" buys us.
    real_replace = Path.replace

    def _boom(self: Path, target: Path) -> Path:  # type: ignore[no-untyped-def]
        raise OSError("simulated replace failure")

    with patch.object(Path, "replace", _boom):
        with pytest.raises(OSError, match="simulated replace failure"):
            store.write({"items": ["bad-partial"], "version": 1})

    # The original sidecar survived intact.
    assert sidecar.read_bytes() == good_bytes
    # And a fresh store still reads the good payload, not the failed write.
    clear_all_caches()
    assert _store(tmp_path).read() == {"items": ["good"], "version": 1}

    # Sanity: real `Path.replace` is restored, so cleanup works.
    assert Path.replace is real_replace


def test_clear_cache_forces_disk_reparse(tmp_path: Path) -> None:
    """After `clear_cache()`, the next read goes back to disk even if mtime is unchanged."""
    store = _store(tmp_path)
    store.write({"items": ["one"], "version": 1})
    assert store.read() == {"items": ["one"], "version": 1}

    # Out-of-band overwrite WITHOUT bumping mtime — emulate a tight test loop
    # where two writes happen inside one mtime tick. Without clear_cache(),
    # we'd see the stale cached value.
    sidecar = tmp_path / FILENAME
    st = sidecar.stat()
    sidecar.write_text(json.dumps({"items": ["two"], "version": 1}))
    os.utime(sidecar, (st.st_atime, st.st_mtime))  # restore mtime

    store.clear_cache()
    assert store.read() == {"items": ["two"], "version": 1}


def test_two_stores_share_no_state(tmp_path: Path) -> None:
    """Independent (folder, filename) pairs cache separately."""
    other = tmp_path / "other"
    other.mkdir()

    a = SidecarStore(tmp_path, FILENAME, DEFAULTS)
    b = SidecarStore(other, FILENAME, DEFAULTS)

    a.write({"items": ["from-a"], "version": 1})
    b.write({"items": ["from-b"], "version": 1})

    assert a.read() == {"items": ["from-a"], "version": 1}
    assert b.read() == {"items": ["from-b"], "version": 1}


def test_write_resets_malformed_warning_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A successful write means the file is well-formed again; future corruption warns afresh."""
    sidecar = tmp_path / FILENAME
    sidecar.write_text("{broken")
    store = _store(tmp_path)

    # First read warns once.
    store.read()
    err1 = capsys.readouterr().err
    assert err1.count("malformed sidecar") == 1

    # Successful write should clear the "already warned" flag.
    store.write({"items": [], "version": 1})

    # Re-corrupt the file. Bump mtime so the cache misses.
    sidecar.write_text("{still broken")
    st = sidecar.stat()
    os.utime(sidecar, (st.st_atime + 1.0, st.st_mtime + 1.0))

    store.read()
    err2 = capsys.readouterr().err
    assert err2.count("malformed sidecar") == 1, (
        "expected a fresh warning after the file was rewritten and re-corrupted"
    )
