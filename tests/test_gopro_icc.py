"""Tests for the GoProICCAdapter — MTP-mode GoPro Import source.

We do NOT exercise PyObjC / a real camera here. The ImageCaptureCore
framework is event-driven and stalls in any unit-test runner without a
GUI run loop; instead we drive the Adapter's public surface with stub
ICCameraFile / ICCameraDevice objects that mimic the ObjC interface
(`.name()`, `.UTI()`, `.modificationDate()`, `.fileSize()`).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aftermovie import optional_dep
from aftermovie.import_sources import all_sources
from aftermovie.import_sources import gopro_icc as icc_mod
from aftermovie.import_sources.gopro_icc import (
    GoProICCAdapter,
    _file_matches_filter,
)


# ---------------------------------------------------------------------------
# Stubs that mimic ICCameraFile / ICCameraDevice without PyObjC.
# ---------------------------------------------------------------------------


class _StubFile:
    """Mimics ICCameraFile's ObjC-style nullary getter selectors."""

    def __init__(self, name: str, uti: str, mtime: datetime, size: int):
        self._name = name
        self._uti = uti
        self._mtime = mtime
        self._size = size

    def name(self) -> str:
        return self._name

    def UTI(self) -> str:
        return self._uti

    def modificationDate(self) -> datetime:
        return self._mtime

    def creationDate(self):
        return None

    def fileSize(self) -> int:
        return self._size


class _StubDevice:
    def __init__(self, files: list[_StubFile]):
        self._files = files

    def mediaFiles(self) -> list[_StubFile]:
        return list(self._files)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_optional_dep_and_browse_cache():
    optional_dep._reset_for_tests()
    icc_mod._reset_browse_cache_for_tests()
    yield
    optional_dep._reset_for_tests()
    icc_mod._reset_browse_cache_for_tests()


def _adapter() -> GoProICCAdapter:
    return GoProICCAdapter(camera_name="HERO9 Black", uuid="UUID-HERO9-AAA")


# ---------------------------------------------------------------------------
# available()
# ---------------------------------------------------------------------------


def test_available_false_when_imagecapturecore_missing(monkeypatch):
    """The Adapter degrades cleanly when the optional dep doesn't import."""
    monkeypatch.setattr(icc_mod._ICC, "module", None)
    assert _adapter().available() is False


def test_available_false_when_no_matching_camera_attached(monkeypatch):
    """Even with ICC loaded, a wrong-uuid browse means we're unavailable."""
    monkeypatch.setattr(icc_mod._ICC, "module", object())  # pretend loaded
    monkeypatch.setattr(icc_mod, "_browse_cameras", lambda force=False: [])
    assert _adapter().available() is False


# ---------------------------------------------------------------------------
# list_in_range filtering — the core "skip junk extensions" invariant.
# ---------------------------------------------------------------------------


def test_list_in_range_skips_lrv_thm_sav_url_even_when_dates_match():
    """A `.lrv`/`.thm`/`.sav`/`.url` whose mtime is in-range must NOT appear.

    The skip happens at filter time, not at copy time — otherwise the GUI
    would show a `100GOPRO/GX010001.LRV` row that the user can't deselect
    independently of the primary MP4.
    """
    now = datetime.now()
    files = [
        _StubFile("GX010001.MP4", "public.movie", now, 1024),
        _StubFile("GX010001.LRV", "", now, 64),  # proxy
        _StubFile("GX010001.THM", "", now, 32),  # thumbnail
        _StubFile("GOPRO.SAV", "", now, 8),       # state file
        _StubFile("GOPRO.URL", "", now, 8),       # web link
        _StubFile("GOPR0002.JPG", "public.jpeg", now, 512),
        _StubFile("IMG_0003.HEIC", "public.heic", now, 256),
    ]
    dev = _StubDevice(files)
    adapter = _adapter()
    since = now - timedelta(hours=1)
    until = now + timedelta(hours=1)

    items = adapter._collect_items(dev, since, until)
    names = sorted(it.src_path for it in items)
    assert names == ["GOPR0002.JPG", "GX010001.MP4", "IMG_0003.HEIC"]

    by_name = {it.src_path: it for it in items}
    assert by_name["GX010001.MP4"].kind == "video"
    assert by_name["GOPR0002.JPG"].kind == "still"
    assert by_name["IMG_0003.HEIC"].kind == "still"
    # Adapter populated its name->ICCameraFile cache for copy_into.
    assert "GX010001.MP4" in adapter._file_cache
    assert "GX010001.LRV" not in adapter._file_cache


def test_list_in_range_filters_out_files_outside_window():
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    files = [
        _StubFile("GX010001.MP4", "public.movie", yesterday, 1024),
        _StubFile("GX010002.MP4", "public.movie", now, 2048),
    ]
    dev = _StubDevice(files)
    adapter = _adapter()
    # Window starts AFTER yesterday → only the second file makes the cut.
    items = adapter._collect_items(dev, now - timedelta(minutes=1), now + timedelta(minutes=1))
    assert [it.src_path for it in items] == ["GX010002.MP4"]


def test_file_matches_filter_excludes_known_junk():
    """Belt-and-braces unit test for the predicate driving the UTI filter."""
    assert _file_matches_filter("GX010001.MP4", "public.movie") is True
    assert _file_matches_filter("GOPR0002.JPG", "public.jpeg") is True
    assert _file_matches_filter("IMG_0003.HEIC", "public.heic") is True
    assert _file_matches_filter("GX010001.LRV", "") is False
    assert _file_matches_filter("GX010001.THM", "") is False
    assert _file_matches_filter("GOPRO.URL", "") is False
    assert _file_matches_filter("GOPRO.SAV", "") is False
    # Unknown UTI with movie extension → still allowed by ext fallback.
    assert _file_matches_filter("X.MP4", "") is True


# ---------------------------------------------------------------------------
# copy_into idempotency — dest exists with matching size → skip, never call
# requestDownloadFile.
# ---------------------------------------------------------------------------


def test_copy_into_skips_when_dest_exists_with_matching_size(monkeypatch, tmp_path: Path):
    """Idempotent-skip invariant matches base.copy_files: dest with same
    size is left alone, requestDownloadFile is NEVER dispatched."""
    # Pretend ICC is loaded; the device-lookup + session-open paths are
    # patched out so the test never hits PyObjC.
    monkeypatch.setattr(icc_mod._ICC, "module", object())
    adapter = _adapter()

    dest = tmp_path / "out"
    dest.mkdir()
    # Seed a same-sized dest file for "GX010001.MP4".
    (dest / "GX010001.MP4").write_bytes(b"\x00" * 1024)

    item_seed = [
        # Manually build the cache the way list_in_range would.
    ]

    icc_file = _StubFile("GX010001.MP4", "public.movie", datetime.now(), 1024)
    adapter._file_cache = {"GX010001.MP4": icc_file}

    fake_dev = object()
    monkeypatch.setattr(adapter, "_find_device", lambda: fake_dev)
    monkeypatch.setattr(icc_mod, "_open_session", lambda d, timeout_s=15.0: None)
    monkeypatch.setattr(icc_mod, "_close_session", lambda d: None)

    download_calls: list[str] = []

    def _fake_download(self, dev, icc, dest_folder, src_name):
        download_calls.append(src_name)
        return True, 1024

    monkeypatch.setattr(GoProICCAdapter, "_download_one", _fake_download)

    from aftermovie.import_sources.base import ImportItem
    items = [ImportItem(
        src_path="GX010001.MP4",
        captured_at=datetime.now().timestamp(),
        kind="video",
        size_bytes=1024,
        source_label=adapter.label,
    )]

    res = adapter.copy_into(items, dest)
    assert res.skipped == 1
    assert res.copied == 0
    assert res.failed == 0
    # Hard contract: no download was actually dispatched.
    assert download_calls == []


def test_copy_into_dispatches_download_when_dest_missing(monkeypatch, tmp_path: Path):
    """Counter to the skip test: a fresh dest path → download IS called."""
    monkeypatch.setattr(icc_mod._ICC, "module", object())
    adapter = _adapter()

    dest = tmp_path / "out"  # not yet created
    icc_file = _StubFile("GX010001.MP4", "public.movie", datetime.now(), 1024)
    adapter._file_cache = {"GX010001.MP4": icc_file}

    monkeypatch.setattr(adapter, "_find_device", lambda: object())
    monkeypatch.setattr(icc_mod, "_open_session", lambda d, timeout_s=15.0: None)
    monkeypatch.setattr(icc_mod, "_close_session", lambda d: None)

    calls: list[str] = []

    def _fake_download(self, dev, icc, dest_folder, src_name):
        calls.append(src_name)
        # Simulate ICC writing the file out so the size check works.
        (Path(dest_folder) / src_name).write_bytes(b"\x00" * 1024)
        return True, 1024

    monkeypatch.setattr(GoProICCAdapter, "_download_one", _fake_download)

    from aftermovie.import_sources.base import ImportItem
    items = [ImportItem(
        src_path="GX010001.MP4",
        captured_at=datetime.now().timestamp(),
        kind="video",
        size_bytes=1024,
        source_label=adapter.label,
    )]

    res = adapter.copy_into(items, dest)
    assert res.copied == 1
    assert res.skipped == 0
    assert res.bytes_written == 1024
    assert calls == ["GX010001.MP4"]


# ---------------------------------------------------------------------------
# Registry hook — ICC branch must not crash when the dep is missing.
# ---------------------------------------------------------------------------


def test_all_sources_does_not_crash_when_imagecapturecore_unavailable(monkeypatch):
    """Module-load + `all_sources()` must succeed when ICC is missing."""
    from aftermovie.import_sources import gopro as gopro_mod
    monkeypatch.setattr(gopro_mod, "detect_gopro_mounts", lambda: [])
    # Force the ICC branch to look unavailable.
    monkeypatch.setattr(icc_mod._ICC, "module", None)

    sources = all_sources()
    # photos_library + zero GoPros + zero ICC adapters.
    assert len(sources) == 1
    assert sources[0].name == "photos_library"


def test_all_sources_adds_one_icc_adapter_per_browsed_gopro(monkeypatch):
    """When ICC IS available and a HERO shows up, we get one extra Adapter."""
    from aftermovie.import_sources import gopro as gopro_mod
    monkeypatch.setattr(gopro_mod, "detect_gopro_mounts", lambda: [])

    fake_adapter = GoProICCAdapter(camera_name="HERO9 Black", uuid="UUID-1")
    monkeypatch.setattr(
        icc_mod, "detect_icc_gopros", lambda: [fake_adapter],
    )

    sources = all_sources()
    assert [s.name for s in sources] == ["photos_library", "gopro_icc_hero9_black"]
    assert sources[1].label == "GoPro (MTP): HERO9 Black"
