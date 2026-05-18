"""Adapter over MTP-mode GoPros via Apple's ImageCaptureCore framework.

The Mass-Storage `GoProAdapter` only sees cameras that mount under
`/Volumes/`. HERO9 / HERO10 default to MTP and do NOT mount as Mass Storage
— Apple's `ptpcamerad` daemon claims the USB interface so `gphoto2` fails
with `Could not claim interface 0`, and `launchctl bootout`-ing the daemon
is blocked by SIP on Sonoma+. The supported escape is to BE a peer of
`ptpcamerad` instead of fighting it: ImageCaptureCore is Apple's
first-party framework for the same protocol.

PyObjC is the bridge — `pyobjc-framework-ImageCaptureCore` ships the
`ImageCaptureCore` Python module that mirrors the ObjC API one-for-one. The
framework is event-driven: an `ICDeviceBrowser` posts to a delegate on the
main run loop, and `ICCameraDevice` does the same for session-open /
download callbacks. We pump `NSRunLoop.currentRunLoop()` in short slices
to turn that into a synchronous Adapter the rest of `aftermovie` can use.

Two run-loop seams matter:

  * Browse: bounded ~5s, cached for 30s at module scope so calls into
    `all_sources()` from the CLI (which can fire many times per render
    flow) don't re-stall the framework.
  * Session open + per-file download: bounded ~15s each. We open one
    session per `list_in_range` call and one per `copy_into` call, and
    close it in a `try/finally` even when the caller bails early.

We can't reuse `copy_files` from `base.py` — that helper `shutil.copy2`s
from a filesystem path, and ICC files have no filesystem path until ICC
materializes them. So `copy_into` re-implements the same idempotent-skip
invariants (skip if dest exists with same size, never overwrite a
different-sized dest) by hand and routes through
`ICCameraDevice.requestDownloadFile_options_...` for the actual transfer.

The `name -> ICCameraFile` map is rebuilt inside each public call from the
camera's `mediaFiles()` because (a) holding ObjC objects across run-loop
quiescent periods is fragile when the device disconnects mid-flow and
(b) it keeps the Adapter stateless across instance reuse.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aftermovie.ffmpeg_cmd import log
from aftermovie.import_sources.base import (
    CopyResult,
    ImportItem,
    ProgressCb,
)
from aftermovie.optional_dep import optional_import


_ICC = optional_import(
    "ImageCaptureCore",
    warning="  ! ImageCaptureCore not available — GoPro MTP import disabled. "
            "Install with: pip install pyobjc-framework-ImageCaptureCore",
)


# ICDeviceTypeCamera is bit 0 on the mask used by ICDeviceBrowser.
ICDeviceTypeMaskCamera = 1

# UTIs we treat as importable. Everything else (`.url`, `.sav`, `.thm`,
# `.lrv` proxy / sidecar files) is filtered out.
_INCLUDED_UTIS = {
    "public.movie",
    "public.mpeg-4",
    "public.jpeg",
    "public.heic",
    "public.heif",
    "public.tiff",
    "public.image",
}
_VIDEO_UTIS = {"public.movie", "public.mpeg-4"}

# Extensions ICC sometimes surfaces with no/blank UTI; we drop them.
_EXCLUDED_EXTS = {".lrv", ".thm", ".url", ".sav"}


def _slug(text: str) -> str:
    """Filesystem / CLI-safe id for the per-camera Adapter name."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "camera"


# ---------------------------------------------------------------------------
# Run-loop helpers. PyObjC's NSRunLoop runs forever by default; we pump it
# in 0.05-0.1s slices, polling a flag set by an ObjC delegate.
# ---------------------------------------------------------------------------


def _pump_until(pred, timeout_s: float, *, slice_s: float = 0.1) -> bool:
    """Pump `NSRunLoop.currentRunLoop()` until `pred()` is true or timeout.

    Returns True iff `pred()` flipped True before the timeout. Used to turn
    ICC's event-driven delegate callbacks (browser-found-device,
    session-opened, download-finished) into synchronous calls.
    """
    from Foundation import NSRunLoop, NSDate
    rl = NSRunLoop.currentRunLoop()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return True
        rl.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(slice_s))
    return pred()


# ---------------------------------------------------------------------------
# Browse — module-level cache with 30s TTL so repeated `all_sources()` calls
# (the CLI hits it multiple times per `import --dry-run`) don't stall.
# ---------------------------------------------------------------------------


@dataclass
class _CachedBrowse:
    devices: list[Any]
    expires_at: float


_BROWSE_CACHE: _CachedBrowse | None = None
_BROWSE_LOCK = threading.Lock()
_BROWSE_TTL_S = 30.0
_BROWSE_TIMEOUT_S = 5.0


def _browse_cameras(force: bool = False) -> list[Any]:
    """Return the currently-attached ICC cameras (cached).

    **Thread invariant**: `ICDeviceBrowser` dispatches callbacks to the
    *main* thread's run loop. Pumping `NSRunLoop.currentRunLoop()` on a
    worker thread (e.g. inside an HTTP request handler) would block forever
    waiting for events that are never delivered there. So a real browse is
    only performed when called from the main thread; off-thread callers
    receive the last cached result (possibly empty if no main-thread call
    has primed it yet). Callers that need fresh data should pre-warm at
    process startup on the main thread.

    The browser is event-driven; we run it for up to ~5s, collect every
    `didAddDevice` callback, then stop. Cached for 30s so repeated calls
    from `all_sources()` (multiple per CLI invocation) don't stall.
    """
    global _BROWSE_CACHE
    if not _ICC.available:
        return []
    on_main = threading.current_thread() is threading.main_thread()
    with _BROWSE_LOCK:
        now = time.time()
        cache_fresh = _BROWSE_CACHE is not None and _BROWSE_CACHE.expires_at > now
        if not force and cache_fresh:
            return list(_BROWSE_CACHE.devices)
        if not on_main:
            # Off-thread: never trigger a real browse — would deadlock on
            # NSRunLoop waiting for main-thread events. Serve stale cache
            # (or empty if no main-thread warmup ever ran).
            return list(_BROWSE_CACHE.devices) if _BROWSE_CACHE else []
        try:
            devices = _browse_cameras_uncached()
        except Exception as e:
            log(f"  ! gopro_icc: ICC browse failed: {e}")
            devices = []
        _BROWSE_CACHE = _CachedBrowse(devices=devices, expires_at=now + _BROWSE_TTL_S)
        return list(devices)


def prewarm_browse_cache() -> None:
    """Force one ICC browse on the main thread to populate the cache.

    Surfaces (`aftermovie select`, `aftermovie import`) should call this at
    startup so subsequent calls from worker threads (HTTP handlers, render
    jobs) have a populated cache to read from.
    """
    _browse_cameras(force=True)


def _browse_cameras_uncached() -> list[Any]:
    """One real ICC browse pass. Caller wraps in the cache."""
    import objc
    from Foundation import NSObject
    ICC = _ICC.require()
    if ICC is None:
        return []

    class _BrowserDelegate(NSObject):
        def init(self):
            self = objc.super(_BrowserDelegate, self).init()
            if self is None:
                return None
            self.devices = []
            self.done = False
            return self

        def deviceBrowser_didAddDevice_moreComing_(self, _b, dev, more):
            self.devices.append(dev)
            if not more:
                self.done = True

        def deviceBrowser_didRemoveDevice_moreGoing_(self, _b, dev, _more):
            try:
                self.devices.remove(dev)
            except ValueError:
                pass

    delegate = _BrowserDelegate.alloc().init()
    browser = ICC.ICDeviceBrowser.alloc().init()
    browser.setDelegate_(delegate)
    browser.setBrowsedDeviceTypeMask_(ICDeviceTypeMaskCamera)
    browser.start()
    try:
        _pump_until(lambda: delegate.done, _BROWSE_TIMEOUT_S)
    finally:
        browser.stop()
    return list(delegate.devices)


def _device_uuid(dev: Any) -> str:
    """Best-effort stable id for an ICDevice across browses."""
    for sel in ("UUIDString", "persistentIDString", "serialNumberString"):
        try:
            val = getattr(dev, sel)()
        except Exception:
            val = None
        if val:
            return str(val)
    try:
        return str(dev.name())
    except Exception:
        return ""


def _device_name(dev: Any) -> str:
    try:
        return str(dev.name() or "")
    except Exception:
        return ""


def _is_gopro(dev: Any) -> bool:
    """Match HERO9, HERO10, GoPro HERO11, etc."""
    n = _device_name(dev)
    return ("HERO" in n) or ("GoPro" in n)


def _reset_browse_cache_for_tests() -> None:
    """Clear the module-level browse cache. Tests only."""
    global _BROWSE_CACHE
    _BROWSE_CACHE = None


# ---------------------------------------------------------------------------
# Session helpers. We wrap "open the session and wait for the catalog" and
# "close the session" so each public Adapter call has a try/finally without
# repeating the boilerplate.
# ---------------------------------------------------------------------------


class _SessionDelegate:
    """Plain-Python facade for the ObjC session delegate.

    We can't make this a top-level class because the ObjC selector names
    (`device:didOpenSessionWithError:`) only exist when PyObjC is loaded.
    Instead this factory returns an `NSObject` subclass at call time.
    """


def _make_session_delegate() -> Any:
    import objc
    from Foundation import NSObject

    class _DeviceDelegate(NSObject):
        def init(self):
            self = objc.super(_DeviceDelegate, self).init()
            if self is None:
                return None
            self.ready = False
            self.opened = False
            self.err: str | None = None
            return self

        def device_didOpenSessionWithError_(self, _d, err):
            if err is not None:
                self.err = str(err)
            self.opened = True

        def device_didCloseSessionWithError_(self, _d, _err):
            self.opened = False

        def device_didEncounterError_(self, _d, err):
            self.err = str(err)

        def deviceDidBecomeReadyWithCompleteContentCatalog_(self, _d):
            self.ready = True

        def didRemoveDevice_(self, _d):
            pass

    return _DeviceDelegate.alloc().init()


def _open_session(dev: Any, timeout_s: float = 15.0) -> Any:
    """Open `dev`, wait for the complete catalog, return the delegate.

    Raises RuntimeError on timeout or framework error. The caller MUST
    `dev.requestCloseSession()` in a `try/finally`.
    """
    delegate = _make_session_delegate()
    dev.setDelegate_(delegate)
    dev.requestOpenSession()
    ok = _pump_until(
        lambda: delegate.ready or delegate.err is not None,
        timeout_s,
    )
    if delegate.err:
        raise RuntimeError(f"ICC session error: {delegate.err}")
    if not ok or not delegate.ready:
        raise RuntimeError("ICC session timeout — catalog not ready")
    return delegate


def _close_session(dev: Any) -> None:
    """Best-effort close. Pumps briefly so the close callback can fire."""
    try:
        dev.requestCloseSession()
    except Exception:
        return
    _pump_until(lambda: False, 0.5)


# ---------------------------------------------------------------------------
# File extraction. We re-derive (name, UTI, mtime, size) into plain Python
# tuples so the Adapter logic is unit-testable with stub objects that
# mimic the same shape.
# ---------------------------------------------------------------------------


def _nsdate_to_ts(d: Any) -> float | None:
    """Convert an NSDate (or a Python datetime, for tests) to POSIX ts."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.timestamp()
    # NSDate: timeIntervalSince1970 returns POSIX seconds as a CDouble.
    try:
        return float(d.timeIntervalSince1970())
    except Exception:
        return None


def _file_kind(uti: str) -> str:
    """Map the file's UTI to the ImportItem.kind vocabulary."""
    if uti in _VIDEO_UTIS:
        return "video"
    return "still"


def _file_matches_filter(name: str, uti: str) -> bool:
    """True iff the file is one we want to import."""
    ext = Path(name).suffix.lower()
    if ext in _EXCLUDED_EXTS:
        return False
    if not uti:
        # No UTI from ICC — fall back to extension whitelist.
        return ext in {".mp4", ".mov", ".m4v", ".jpg", ".jpeg",
                       ".heic", ".heif", ".png", ".tif", ".tiff"}
    return uti in _INCLUDED_UTIS


# ---------------------------------------------------------------------------
# The Adapter itself.
# ---------------------------------------------------------------------------


class GoProICCAdapter:
    """`ImportSource` over one MTP-mode GoPro reached via ImageCaptureCore.

    One Adapter instance per detected ICC camera; the uuid pins the
    Adapter to a specific physical device across browse refreshes so
    `available()` stays honest if a second GoPro is plugged in.

    The Adapter is stateless across `list_in_range` / `copy_into` boundaries
    — each call re-opens a session, rebuilds the `name -> ICCameraFile`
    map from `dev.mediaFiles()`, and closes the session in a `try/finally`.
    Holding ObjC files across calls is fragile if the user disconnects
    the camera between dry-run and real copy, and stateless calls keep the
    Locality of "one CLI subcommand → one ICC transaction".
    """

    name: str
    label: str

    def __init__(self, camera_name: str, uuid: str) -> None:
        self.camera_name = camera_name
        self.uuid = uuid
        self.name = f"gopro_icc_{_slug(camera_name)}"
        self.label = f"GoPro (MTP): {camera_name}"
        # name -> ICCameraFile cache populated by list_in_range so copy_into
        # can match items back to ObjC handles without a second catalog walk.
        # Bounded to the latest list_in_range result; rebuilt on each call.
        self._file_cache: dict[str, Any] = {}

    # -- Adapter Interface -------------------------------------------------

    def available(self) -> bool:
        """True iff ImageCaptureCore loads AND a camera with our uuid
        currently shows up in a fresh (cached, 30s TTL) ICC browse."""
        if not _ICC.available:
            return False
        for dev in _browse_cameras():
            if _device_uuid(dev) == self.uuid:
                return True
        return False

    def _find_device(self) -> Any | None:
        for dev in _browse_cameras(force=True):
            if _device_uuid(dev) == self.uuid:
                return dev
        return None

    def list_in_range(self, since: datetime, until: datetime) -> list[ImportItem]:
        """Open a session, enumerate `mediaFiles()`, filter, return items.

        Side-effect: rebuilds `self._file_cache` (`name -> ICCameraFile`)
        so `copy_into` can match items back without a second catalog walk.
        """
        if not _ICC.available:
            return []
        dev = self._find_device()
        if dev is None:
            return []

        self._file_cache = {}
        try:
            _open_session(dev)
        except RuntimeError as e:
            log(f"  ! gopro_icc: {e}")
            return []

        try:
            return self._collect_items(dev, since, until)
        finally:
            _close_session(dev)

    def _collect_items(
        self, dev: Any, since: datetime, until: datetime,
    ) -> list[ImportItem]:
        """Pure(ish) filter step — extracted so tests can drive it with a
        stub device exposing a `mediaFiles()` returning fake file objects."""
        since_ts = since.timestamp()
        until_ts = until.timestamp()
        items: list[ImportItem] = []
        media_files = dev.mediaFiles() or []
        for f in media_files:
            name = str(f.name() or "")
            uti = str(f.UTI() or "")
            if not name:
                continue
            if not _file_matches_filter(name, uti):
                continue
            mod = f.modificationDate()
            ts = _nsdate_to_ts(mod)
            if ts is None:
                # Fallback: try creationDate, then skip.
                ts = _nsdate_to_ts(f.creationDate())
            if ts is None:
                continue
            if ts < since_ts or ts > until_ts:
                continue
            try:
                size = int(f.fileSize())
            except Exception:
                size = 0
            self._file_cache[name] = f
            items.append(ImportItem(
                src_path=name,  # ICC files have no filesystem path
                captured_at=ts,
                kind=_file_kind(uti),
                size_bytes=size,
                source_label=self.label,
                extra={},
            ))
        items.sort(key=lambda it: (it.captured_at, it.src_path))
        return items

    def copy_into(
        self,
        items: list[ImportItem],
        dest_folder: Path,
        progress_cb: ProgressCb | None = None,
    ) -> CopyResult:
        """Download each item via ICC, honoring the shared idempotency rules.

        Same invariants as `base.copy_files`:
          - skip if dest exists with same size
          - never overwrite a different-sized dest (log + skip)
          - call `progress_cb(done, total, src)` after each file

        Implemented by hand because ICC files have no filesystem path;
        `shutil.copy2`-based `copy_files` can't reach them. Each download
        is async via `requestDownloadFile_options_...`; we pump the run
        loop after each call to keep the Adapter synchronous.
        """
        dest_folder = dest_folder.expanduser().resolve()
        dest_folder.mkdir(parents=True, exist_ok=True)
        res = CopyResult(dest_folder=str(dest_folder))
        if not items:
            return res
        if not _ICC.available:
            log("  ! gopro_icc: ImageCaptureCore unavailable; cannot copy.")
            res.failed = len(items)
            return res

        dev = self._find_device()
        if dev is None:
            log(f"  ! gopro_icc: device {self.camera_name} no longer attached.")
            res.failed = len(items)
            return res

        # Re-open the session for the copy phase. The session opened by a
        # prior list_in_range was already closed (try/finally).
        try:
            _open_session(dev)
        except RuntimeError as e:
            log(f"  ! gopro_icc: {e}")
            res.failed = len(items)
            return res

        try:
            # Rebuild name -> ICCameraFile if list_in_range wasn't called
            # immediately before (or if the device replugged): a fresh
            # mediaFiles() pull is cheap once the session is open.
            if not self._file_cache:
                for f in dev.mediaFiles() or []:
                    try:
                        self._file_cache[str(f.name())] = f
                    except Exception:
                        continue

            total = len(items)
            done = 0
            for item in items:
                src_name = Path(item.src_path).name
                dest = dest_folder / src_name
                # Idempotent-skip invariants — by hand because copy_files
                # operates on filesystem paths and ICC files don't have one.
                if dest.exists():
                    try:
                        dest_size = dest.stat().st_size
                    except OSError:
                        dest_size = -1
                    if dest_size == item.size_bytes and item.size_bytes > 0:
                        res.skipped += 1
                        done += 1
                        if progress_cb:
                            progress_cb(done, total, item.src_path)
                        continue
                    if item.size_bytes > 0:
                        log(
                            f"  ! import: {src_name} already exists with "
                            f"different size ({dest_size} vs "
                            f"{item.size_bytes}); skipping (not overwriting)."
                        )
                        res.skipped += 1
                        done += 1
                        if progress_cb:
                            progress_cb(done, total, item.src_path)
                        continue

                icc_file = self._file_cache.get(src_name)
                if icc_file is None:
                    log(f"  ! gopro_icc: file {src_name} not on camera.")
                    res.failed += 1
                    done += 1
                    if progress_cb:
                        progress_cb(done, total, item.src_path)
                    continue

                ok, bytes_w = self._download_one(dev, icc_file, dest_folder, src_name)
                if ok:
                    res.copied += 1
                    res.bytes_written += bytes_w
                else:
                    res.failed += 1
                done += 1
                if progress_cb:
                    progress_cb(done, total, item.src_path)
        finally:
            _close_session(dev)
        return res

    def _download_one(
        self, dev: Any, icc_file: Any, dest_folder: Path, src_name: str,
    ) -> tuple[bool, int]:
        """Fire one async ICC download and pump the run loop until it
        finishes. Returns (ok, bytes_written)."""
        import objc
        from Foundation import (
            NSObject, NSURL, NSMutableDictionary, NSNumber, NSString,
        )
        ICC = _ICC.require()
        if ICC is None:
            return False, 0

        class _DLDelegate(NSObject):
            def init(self):
                self = objc.super(_DLDelegate, self).init()
                if self is None:
                    return None
                self.done = False
                self.err: str | None = None
                return self

            def didDownloadFile_error_options_contextInfo_(
                self, _file, err, _opts, _ctx,
            ):
                if err is not None:
                    self.err = str(err)
                self.done = True

        dl = _DLDelegate.alloc().init()
        opts = NSMutableDictionary.dictionary()
        try:
            opts.setObject_forKey_(
                NSURL.fileURLWithPath_(str(dest_folder)),
                ICC.ICDownloadsDirectoryURL,
            )
            opts.setObject_forKey_(
                NSString.stringWithString_(src_name),
                ICC.ICSaveAsFilename,
            )
            opts.setObject_forKey_(
                NSNumber.numberWithBool_(False),
                ICC.ICOverwrite,
            )
        except AttributeError:
            # Older PyObjC: keys are plain string constants.
            opts.setObject_forKey_(
                NSURL.fileURLWithPath_(str(dest_folder)),
                "ICDownloadsDirectoryURL",
            )
            opts.setObject_forKey_(
                NSString.stringWithString_(src_name),
                "ICSaveAsFilename",
            )
            opts.setObject_forKey_(
                NSNumber.numberWithBool_(False),
                "ICOverwrite",
            )

        sel = objc.selector(
            None,
            selector=b"didDownloadFile:error:options:contextInfo:",
        )
        try:
            dev.requestDownloadFile_options_downloadDelegate_didDownloadSelector_contextInfo_(
                icc_file, opts, dl, sel, None,
            )
        except Exception as e:
            log(f"  ! gopro_icc: download dispatch failed for {src_name}: {e}")
            return False, 0

        _pump_until(lambda: dl.done, 120.0)
        if dl.err:
            log(f"  ! gopro_icc: download failed for {src_name}: {dl.err}")
            return False, 0
        dest_path = dest_folder / src_name
        try:
            size = dest_path.stat().st_size
        except OSError:
            size = 0
        return True, size


# ---------------------------------------------------------------------------
# Registry helper used by base.all_sources().
# ---------------------------------------------------------------------------


def detect_icc_gopros() -> list[GoProICCAdapter]:
    """Return one `GoProICCAdapter` per HERO/GoPro on the ICC bus.

    Empty list when ImageCaptureCore is missing or no camera is attached.
    Browser results are cached (30s TTL) at module scope so the CLI can
    call `all_sources()` cheaply.
    """
    if not _ICC.available:
        return []
    out: list[GoProICCAdapter] = []
    try:
        devs = _browse_cameras()
    except Exception as e:
        log(f"  ! gopro_icc: browse failed: {e}")
        return []
    for dev in devs:
        if not _is_gopro(dev):
            continue
        name = _device_name(dev) or "GoPro"
        uuid = _device_uuid(dev)
        out.append(GoProICCAdapter(camera_name=name, uuid=uuid))
    # Stable ordering: by camera name (so two HEROs don't shuffle between runs).
    out.sort(key=lambda a: a.camera_name)
    return out
