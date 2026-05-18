"""Tests for the import_sources Module — Adapter Interface + registry."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aftermovie import optional_dep
from aftermovie.import_sources import (
    CopyResult,
    ImportItem,
    all_sources,
)
from aftermovie.import_sources.base import copy_files
from aftermovie.import_sources.gopro import GoProAdapter
from aftermovie.import_sources.photos import PhotosLibraryAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_optional_dep_registry():
    """Make sure the optional_dep cache doesn't bleed between tests."""
    optional_dep._reset_for_tests()
    yield
    optional_dep._reset_for_tests()


def _seed_gopro_dcim(root: Path) -> Path:
    """Build a small fixture DCIM/ tree under `root` and return the mount."""
    mount = root / "GOPRO"
    dcim = mount / "DCIM" / "100GOPRO"
    dcim.mkdir(parents=True)

    # An MP4 from yesterday, a JPG from today, a Live Photo pair from today,
    # plus an LRV proxy + THM thumb the Adapter MUST skip.
    now = datetime.now()
    yesterday = now - timedelta(days=1)

    mp4 = dcim / "GX010001.MP4"
    mp4.write_bytes(b"\x00" * 64)
    os.utime(mp4, (yesterday.timestamp(), yesterday.timestamp()))

    jpg = dcim / "GOPR0002.JPG"
    jpg.write_bytes(b"\x00" * 64)
    os.utime(jpg, (now.timestamp(), now.timestamp()))

    heic = dcim / "IMG_0003.HEIC"
    heic.write_bytes(b"\x00" * 64)
    os.utime(heic, (now.timestamp(), now.timestamp()))
    mov = dcim / "IMG_0003.MOV"
    mov.write_bytes(b"\x00" * 64)
    os.utime(mov, (now.timestamp(), now.timestamp()))

    # Proxies that MUST be skipped.
    lrv = dcim / "GX010001.LRV"
    lrv.write_bytes(b"\x00" * 64)
    os.utime(lrv, (now.timestamp(), now.timestamp()))
    thm = dcim / "GX010001.THM"
    thm.write_bytes(b"\x00" * 64)
    os.utime(thm, (now.timestamp(), now.timestamp()))

    return mount


# ---------------------------------------------------------------------------
# GoProAdapter
# ---------------------------------------------------------------------------


def test_gopro_list_in_range_filters_by_mtime_and_skips_proxies(tmp_path: Path):
    """Adapter walks DCIM/, filters by mtime, ignores LRV/THM proxies."""
    mount = _seed_gopro_dcim(tmp_path)
    adapter = GoProAdapter(mount)

    assert adapter.available()
    assert adapter.name == "gopro"
    assert "GOPRO" in adapter.label

    now = datetime.now()
    two_days_ago = now - timedelta(days=2)
    items = adapter.list_in_range(two_days_ago, now + timedelta(minutes=1))

    names = sorted(Path(it.src_path).name for it in items)
    # MP4 + JPG + HEIC (HEIC pairs the MOV into extra). MOV is consumed
    # into the HEIC's live_photo entry; LRV + THM are filtered out.
    assert names == ["GOPR0002.JPG", "GX010001.MP4", "IMG_0003.HEIC"]

    by_name = {Path(it.src_path).name: it for it in items}
    assert by_name["GX010001.MP4"].kind == "video"
    assert by_name["GOPR0002.JPG"].kind == "still"
    assert by_name["IMG_0003.HEIC"].kind == "live_photo"
    assert by_name["IMG_0003.HEIC"].extra["live_photo_mov"].endswith("IMG_0003.MOV")


def test_gopro_list_in_range_excludes_files_outside_window(tmp_path: Path):
    """A file with an mtime before `since` must NOT appear."""
    mount = _seed_gopro_dcim(tmp_path)
    adapter = GoProAdapter(mount)

    # Window starts AFTER yesterday → the MP4 from yesterday is excluded.
    just_now = datetime.now() - timedelta(seconds=2)
    items = adapter.list_in_range(just_now, datetime.now() + timedelta(minutes=1))
    names = {Path(it.src_path).name for it in items}
    assert "GX010001.MP4" not in names  # yesterday — out of range
    # The today files are still in range.
    assert "GOPR0002.JPG" in names
    assert "IMG_0003.HEIC" in names


def test_gopro_copy_into_is_idempotent_and_copies_live_photo_pair(tmp_path: Path):
    """Second copy_into run = 0 newly copied; Live Photo MOV is copied too."""
    mount = _seed_gopro_dcim(tmp_path)
    adapter = GoProAdapter(mount)
    items = adapter.list_in_range(
        datetime.now() - timedelta(days=2),
        datetime.now() + timedelta(minutes=1),
    )

    dest = tmp_path / "out"
    res1 = adapter.copy_into(items, dest)
    # MP4 + JPG + HEIC + paired MOV = 4 files actually written.
    assert res1.copied == 4, res1
    assert res1.failed == 0
    assert (dest / "IMG_0003.HEIC").is_file()
    assert (dest / "IMG_0003.MOV").is_file()
    assert (dest / "GOPR0002.JPG").is_file()
    assert (dest / "GX010001.MP4").is_file()

    res2 = adapter.copy_into(items, dest)
    assert res2.copied == 0, "second run must be a no-op (idempotency)"
    assert res2.skipped == 4
    assert res2.failed == 0


def test_copy_files_warns_and_skips_on_size_conflict(tmp_path: Path, capsys):
    """Never overwrite a same-named dest that has a different size."""
    src = tmp_path / "src" / "a.bin"
    src.parent.mkdir()
    src.write_bytes(b"\x00" * 64)
    dest_folder = tmp_path / "dest"
    dest_folder.mkdir()
    # Pre-existing dest file with DIFFERENT size — must be left alone.
    (dest_folder / "a.bin").write_bytes(b"\x00" * 8)

    item = ImportItem(
        src_path=str(src),
        captured_at=0.0,
        kind="still",
        size_bytes=64,
        source_label="test",
    )
    res = copy_files([item], dest_folder)
    assert res.copied == 0
    assert res.skipped == 1
    # The pre-existing file is preserved untouched.
    assert (dest_folder / "a.bin").stat().st_size == 8
    err = capsys.readouterr().err
    assert "different size" in err


# ---------------------------------------------------------------------------
# PhotosLibraryAdapter (no real Photos library on test machines)
# ---------------------------------------------------------------------------


def test_photos_library_available_false_when_osxphotos_missing(monkeypatch, tmp_path):
    """Adapter degrades cleanly when the optional dep isn't importable."""
    from aftermovie.import_sources import photos as photos_mod

    monkeypatch.setattr(photos_mod._OSXPHOTOS, "module", None)
    adapter = PhotosLibraryAdapter(library_path=tmp_path)
    assert adapter.available() is False


def test_photos_library_available_true_when_osxphotos_present(monkeypatch, tmp_path):
    """The other gate: importable AND a library directory exists on disk."""
    from aftermovie.import_sources import photos as photos_mod

    # Pretend the library bundle exists.
    library = tmp_path / "Photos Library.photoslibrary"
    library.mkdir()
    monkeypatch.setattr(photos_mod._OSXPHOTOS, "module", object())
    adapter = PhotosLibraryAdapter(library_path=library)
    assert adapter.available() is True


def test_photos_library_list_in_range_empty_when_dep_missing(monkeypatch, tmp_path):
    """No osxphotos → list_in_range returns [] silently (warns once)."""
    from aftermovie.import_sources import photos as photos_mod

    monkeypatch.setattr(photos_mod._OSXPHOTOS, "module", None)
    adapter = PhotosLibraryAdapter(library_path=tmp_path)
    items = adapter.list_in_range(
        datetime(2026, 1, 1), datetime(2026, 12, 31),
    )
    assert items == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_sources_orders_photos_library_first(monkeypatch, tmp_path):
    """Registry contract: photos_library always first, gopros after."""
    from aftermovie.import_sources import gopro as gopro_mod

    fake_mount_a = tmp_path / "GOPRO_A"
    fake_mount_b = tmp_path / "GOPRO_B"
    (fake_mount_a / "DCIM").mkdir(parents=True)
    (fake_mount_b / "DCIM").mkdir(parents=True)

    monkeypatch.setattr(
        gopro_mod, "detect_gopro_mounts",
        lambda: [fake_mount_a, fake_mount_b],
    )

    sources = all_sources()
    assert len(sources) == 3
    assert sources[0].name == "photos_library"
    assert sources[1].name == "gopro"
    assert sources[2].name == "gopro"
    # Stable per-mount label so the UI can tell two GoPros apart.
    assert sources[1].label != sources[2].label


def test_all_sources_yields_only_photos_when_no_gopros(monkeypatch):
    """No mounts → just the Photos library Adapter is returned."""
    from aftermovie.import_sources import gopro as gopro_mod

    monkeypatch.setattr(gopro_mod, "detect_gopro_mounts", lambda: [])
    sources = all_sources()
    assert len(sources) == 1
    assert sources[0].name == "photos_library"


def test_import_item_dataclass_shape():
    """The contract the GUI + HTTP agents code against."""
    item = ImportItem(
        src_path="/tmp/a.mp4",
        captured_at=1234.0,
        kind="video",
        size_bytes=100,
        source_label="GoPro (X)",
    )
    assert item.extra == {}
    assert item.kind == "video"


def test_copy_result_defaults_to_zero():
    res = CopyResult()
    assert res.copied == 0 and res.skipped == 0
    assert res.failed == 0 and res.bytes_written == 0
    assert res.dest_folder == ""
