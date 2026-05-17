"""Stdlib HTTP server backing the `aftermovie select` web GUI.

Endpoints (all JSON unless otherwise noted):

    GET  /                          → index.html (or a placeholder)
    GET  /api/sources               → [{path, name, kind, thumb_url, selected}, ...]
    GET  /thumbs/<sha1>.jpg         → cached 256x256 JPG (or 404)
    POST /api/selection             → save .aftermovie-selection.json
    POST /api/render                → launch run_auto in a worker thread → {job_id}
    GET  /api/status/<job_id>       → {state, output_path, log_tail}

The server is intentionally synchronous and single-threaded except for the
render worker (one per render job). Stdlib HTTPServer is the only
dependency here — we don't want to pull Flask/FastAPI into the runtime.

The CLI side is `cli.py::cmd_select`; this module is import-free of argparse
so it can also be driven directly from tests.
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aftermovie.analyze.capture_time import captured_at_for
from aftermovie.analyze.selection import (
    SELECTION_FILENAME,
    load_excluded,
    save_excluded,
)
from aftermovie.analyze.stills import (
    LIVE_PHOTO_VIDEO_EXTS,
    STILL_EXTS,
    _is_excluded_output,
    _under_skipped_dir,
)
from aftermovie.config import THEMES, VIDEO_EXTS, list_luts
from aftermovie.select.thumbnails import _cache_key as thumb_cache_key
from aftermovie.select.thumbnails import _thumbs_cache_dir, thumb_path_for

# ---- source discovery -------------------------------------------------------

# A source row the GUI sees. The `path` field is the canonical absolute path
# we use to round-trip the user's selection back to disk.
@dataclass
class SourceRow:
    path: str
    name: str
    kind: str  # "video" | "still" | "live_photo"
    thumb_url: str
    selected: bool
    captured_at: float | None
    size_bytes: int | None


def _kind_for(p: Path, live_movs: set[str]) -> str:
    if str(p) in live_movs:
        return "live_photo"
    if p.suffix.lower() in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
        return "still"
    return "video"


def _list_sources(folder: Path) -> list[SourceRow]:
    """Walk `folder` the same way the analyzer does and return GUI rows.

    Live Photo pairs (`IMG_0488.HEIC` + `IMG_0488.MOV`) collapse to a single
    `live_photo` entry pointing at the MOV (the HEIC is implied by the
    shared stem). Standalone stills are `still`. Everything else with a
    VIDEO_EXTS suffix is `video`. Hidden files, prior aftermovie outputs,
    and SKIP_DIR_NAMES subtrees are dropped exactly like `discover_sources`.

    Items are sorted by their capture timestamp (EXIF / ffprobe / mtime).
    """
    folder = folder.resolve()
    if not folder.is_dir():
        return []

    excluded = load_excluded(folder)

    # First pass: group files by stem so we can detect paired Live Photos.
    by_stem: dict[str, list[Path]] = {}
    for p in folder.rglob("*"):
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        if p.name.startswith("."):
            continue
        if _under_skipped_dir(p, folder):
            continue
        by_stem.setdefault(p.stem, []).append(p)

    # Sets so we can identify Live Photo MOVs and stills that have a video pair.
    live_movs: set[str] = set()
    paired_still_paths: set[str] = set()
    for files in by_stem.values():
        movs = [f for f in files if f.suffix in LIVE_PHOTO_VIDEO_EXTS]
        stills = [f for f in files if f.suffix in STILL_EXTS]
        if movs and stills:
            # Treat the MOV as a Live Photo entry; hide the still companions.
            for m in movs:
                live_movs.add(str(m))
            for s in stills:
                paired_still_paths.add(str(s))

    # Allow both the case-sensitive VIDEO_EXTS set the analyzer uses AND a
    # case-insensitive fallback so .lrv / .insv variants captured on other
    # filesystems still show up.
    video_exts_ci = {e.lower() for e in VIDEO_EXTS}
    still_exts_ci = {".heic", ".heif", ".jpg", ".jpeg", ".png"}

    rows: list[SourceRow] = []
    seen_paths: set[str] = set()
    for files in by_stem.values():
        for p in files:
            abs_p = str(p)
            if abs_p in seen_paths:
                continue
            if abs_p in paired_still_paths:
                continue  # hidden — its MOV represents the Live Photo
            ext_l = p.suffix.lower()
            if ext_l in video_exts_ci:
                if _is_excluded_output(p):
                    continue
            elif ext_l in still_exts_ci:
                pass
            else:
                continue
            kind = _kind_for(p, live_movs)
            try:
                size_b = p.stat().st_size
            except OSError:
                size_b = None
            try:
                captured = captured_at_for(p)
            except Exception:
                captured = None
            thumb_url = f"/thumbs/{thumb_cache_key(p)}.jpg"
            rows.append(SourceRow(
                path=abs_p,
                name=p.name,
                kind=kind,
                thumb_url=thumb_url,
                selected=(abs_p not in excluded),
                captured_at=captured,
                size_bytes=size_b,
            ))
            seen_paths.add(abs_p)

    rows.sort(key=lambda r: (r.captured_at if r.captured_at is not None else float("inf"), r.name))
    return rows


# ---- render job tracking ----------------------------------------------------

@dataclass
class RenderJob:
    job_id: str
    state: str = "running"  # "running" | "done" | "error"
    output_path: str | None = None
    error: str | None = None
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class _LogCapture(io.TextIOBase):
    """A tee-style stderr replacement that pushes each line into a job's log_tail."""

    def __init__(self, job: RenderJob, mirror: io.TextIOBase | None) -> None:
        super().__init__()
        self._job = job
        self._mirror = mirror
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        if self._mirror is not None:
            try:
                self._mirror.write(s)
            except Exception:
                pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._job.log_tail.append(line)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass


def _run_render_job(
    job: RenderJob,
    clips: Path,
    song: Path | None,
    output: Path | None,
    overrides: dict[str, Any],
) -> None:
    """Worker thread: drive `pipeline_runner.run_auto` and capture state on the job."""
    # Imports are deferred so just importing `select.server` doesn't drag in
    # librosa/numpy/ffmpeg-heavy modules in test contexts that don't render.
    from aftermovie.effective_config import resolve
    from aftermovie.pipeline_runner import AutoOpts, opts_from_namespace, run_auto

    saved_stderr = sys.stderr
    sys.stderr = _LogCapture(job, saved_stderr if sys.stderr is not None else None)
    try:
        if song is None:
            raise ValueError("`song` is required to render")
        cfg = resolve(cli_overrides=overrides, theme=overrides.get("theme"))
        # Build AutoOpts from the resolved config, then layer the few extras
        # the CLI Namespace would normally carry.
        import argparse
        from dataclasses import asdict as _asdict
        ns = argparse.Namespace(**_asdict(cfg))
        for k, v in overrides.items():
            setattr(ns, k, v)
        opts = opts_from_namespace(ns)
        opts.reveal = False  # never trigger Finder reveal from the web flow
        out_path = output
        if out_path is None:
            out_dir = Path(cfg.output_dir).expanduser()
            out_path = out_dir / f"aftermovie-{clips.resolve().name or 'edit'}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_path = run_auto(clips, song, out_path, opts)
        job.output_path = str(result_path)
        job.state = "done"
    except Exception as e:
        tb = traceback.format_exc()
        job.error = f"{type(e).__name__}: {e}"
        for line in tb.splitlines():
            job.log_tail.append(line)
        job.state = "error"
    finally:
        sys.stderr = saved_stderr
        job.finished_at = time.time()


# ---- HTTP handler -----------------------------------------------------------

# Class-level state. ThreadingHTTPServer instantiates a fresh handler per
# request, so we use class attrs the SelectServer wires up at boot.
class _Handler(BaseHTTPRequestHandler):
    clips_root: Path = None  # type: ignore[assignment]
    song_default: Path | None = None
    static_dir: Path | None = None
    jobs: dict[str, RenderJob] = {}
    jobs_lock: threading.Lock = threading.Lock()

    # Stay quiet by default; the CLI prints its own startup banner.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    # ---- routing helpers ----
    def _send_json(self, body: Any, status: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    # ---- GET ----
    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/" or route == "/index.html":
                self._serve_index()
                return
            if route == "/api/sources":
                self._serve_sources()
                return
            if route == "/api/options":
                self._serve_options()
                return
            m = re.match(r"^/thumbs/([A-Fa-f0-9]+)\.jpg$", route)
            if m:
                self._serve_thumb(m.group(1))
                return
            m = re.match(r"^/api/status/([A-Za-z0-9\-]+)$", route)
            if m:
                self._serve_status(m.group(1))
                return
            # Static sibling files (app.js, style.css, any future assets).
            # Restricted to `static_dir` so we can't escape via ../ traversal.
            if self.static_dir is not None and "/" not in route.lstrip("/")[1:]:
                name = route.lstrip("/")
                if name and "/" not in name and name not in (".", ".."):
                    candidate = (self.static_dir / name).resolve()
                    try:
                        candidate.relative_to(self.static_dir.resolve())
                    except ValueError:
                        candidate = None
                    if candidate and candidate.is_file():
                        self._serve_static(candidate)
                        return
            self._send_json({"error": "not_found", "path": route}, status=404)
        except Exception as e:
            self._send_json({"error": "server_error", "detail": str(e)}, status=500)

    # ---- POST ----
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/api/selection":
                self._handle_selection(self._read_json())
                return
            if route == "/api/render":
                self._handle_render(self._read_json())
                return
            self._send_json({"error": "not_found", "path": route}, status=404)
        except Exception as e:
            self._send_json({"error": "server_error", "detail": str(e)}, status=500)

    # ---- handlers ----
    def _serve_index(self) -> None:
        if self.static_dir is not None:
            idx = self.static_dir / "index.html"
            if idx.is_file():
                try:
                    body = idx.read_bytes()
                    self._send_bytes(body, "text/html; charset=utf-8")
                    return
                except OSError:
                    pass
        body = (
            b"<!doctype html><html><head><meta charset='utf-8'>"
            b"<title>aftermovie select</title></head><body>"
            b"<h1>aftermovie select</h1>"
            b"<p>The frontend hasn't been bundled yet. The JSON API is live at "
            b"<code>/api/sources</code>.</p></body></html>"
        )
        self._send_bytes(body, "text/html; charset=utf-8")

    def _serve_static(self, path: Path) -> None:
        import mimetypes

        try:
            body = path.read_bytes()
        except OSError:
            self._send_json({"error": "not_found", "path": str(path)}, status=404)
            return
        content_type, _ = mimetypes.guess_type(path.name)
        self._send_bytes(body, content_type or "application/octet-stream")

    def _serve_sources(self) -> None:
        rows = _list_sources(self.clips_root)
        payload = [{
            "path": r.path,
            "name": r.name,
            "kind": r.kind,
            "thumb_url": r.thumb_url,
            "selected": r.selected,
            "captured_at": r.captured_at,
            "size_bytes": r.size_bytes,
        } for r in rows]
        self._send_json(payload)

    def _serve_options(self) -> None:
        self._send_json({
            "luts": list_luts(),
            "themes": [
                {"name": name, **meta}
                for name, meta in sorted(THEMES.items())
            ],
            "audio_mix": ["ducked", "music_only", "clip_only"],
            "pace": ["auto", "slow", "medium", "fast"],
            "transitions": ["soft", "auto", "cut"],
            "aspect": ["16:9", "9:16", "1:1"],
            "resolution": ["1920x1080", "1080x1920", "1080x1080", "1280x720"],
        })

    def _serve_thumb(self, key: str) -> None:
        # Look up the SourceRow whose key matches; this also ensures the
        # thumb is generated on demand even when the rows endpoint hasn't
        # been hit first.
        cache_dir = _thumbs_cache_dir()
        cached = cache_dir / f"{key}.jpg"
        if not cached.is_file():
            # Walk the folder to find a matching source.
            for r in _list_sources(self.clips_root):
                if r.thumb_url == f"/thumbs/{key}.jpg":
                    p = thumb_path_for(Path(r.path))
                    if p is not None:
                        cached = p
                    break
        if not cached.is_file():
            self._send_json({"error": "not_found"}, status=404)
            return
        try:
            data = cached.read_bytes()
        except OSError:
            self._send_json({"error": "read_failed"}, status=500)
            return
        self._send_bytes(data, "image/jpeg")

    def _serve_status(self, job_id: str) -> None:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
        if job is None:
            self._send_json({"error": "not_found", "job_id": job_id}, status=404)
            return
        body = {
            "job_id": job.job_id,
            "state": job.state,
            "output_path": job.output_path,
            "error": job.error,
            "log_tail": "\n".join(job.log_tail),
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
        self._send_json(body)

    def _handle_selection(self, body: dict[str, Any]) -> None:
        raw = body.get("excluded")
        if not isinstance(raw, list):
            self._send_json({"error": "bad_request",
                             "detail": "expected JSON body with 'excluded': []"},
                            status=400)
            return
        excluded = [str(p) for p in raw if isinstance(p, str)]
        sidecar = save_excluded(self.clips_root, excluded)
        self._send_json({"ok": True, "sidecar": str(sidecar),
                         "filename": SELECTION_FILENAME, "n_excluded": len(excluded)})

    def _handle_render(self, body: dict[str, Any]) -> None:
        raw_excluded = body.get("excluded")
        if isinstance(raw_excluded, list):
            excluded = [str(p) for p in raw_excluded if isinstance(p, str)]
            save_excluded(self.clips_root, excluded)

        # `song` may come from the request, otherwise the CLI-supplied default.
        song = self.song_default
        if isinstance(body.get("song"), str):
            song = Path(body["song"]).expanduser().resolve()

        output: Path | None = None
        if isinstance(body.get("output"), str) and body["output"]:
            output = Path(body["output"]).expanduser().resolve()

        # Whitelist render overrides — only known knobs flow through.
        ALLOWED_OVERRIDES = {
            "max_length", "pace", "aspect", "res", "fps", "music_db", "clip_db",
            "audio_mix", "transitions", "no_speed_ramp", "no_reframe", "lut",
            "theme", "source_cap", "chronological", "preview", "no_stills",
            "still_duration", "titles", "title_text", "burst_window_s",
        }
        overrides: dict[str, Any] = {}
        for k in ALLOWED_OVERRIDES:
            if k in body and body[k] is not None:
                overrides[k] = body[k]

        job = RenderJob(job_id=str(uuid.uuid4()))
        with self.jobs_lock:
            self.jobs[job.job_id] = job
        worker = threading.Thread(
            target=_run_render_job,
            args=(job, self.clips_root, song, output, overrides),
            name=f"render-{job.job_id[:8]}",
            daemon=True,
        )
        worker.start()
        self._send_json({"job_id": job.job_id})


# ---- server boot ------------------------------------------------------------

class SelectServer:
    """Boot/teardown wrapper around `ThreadingHTTPServer` for the select GUI.

    Tests use this as a context manager; the CLI calls `serve_forever()`
    directly and lets Ctrl-C raise KeyboardInterrupt.
    """

    def __init__(self, clips: Path, *, port: int = 8765,
                 song: Path | None = None, static_dir: Path | None = None) -> None:
        clips_root = Path(clips).expanduser().resolve()
        if not clips_root.is_dir():
            raise NotADirectoryError(f"--clips path is not a directory: {clips_root}")
        if static_dir is None:
            static_dir = Path(__file__).resolve().parent / "static"

        # Fresh per-server handler subclass so multiple SelectServers can
        # coexist (e.g. parallel tests) without sharing class-level state.
        class Handler(_Handler):
            pass
        Handler.clips_root = clips_root
        Handler.song_default = song
        Handler.static_dir = static_dir
        Handler.jobs = {}
        Handler.jobs_lock = threading.Lock()
        self._handler_cls = Handler

        self.clips_root = clips_root
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- public surface ----
    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"

    def __enter__(self) -> "SelectServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler_cls)
        # If port=0 was passed, capture the OS-assigned port.
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name=f"select-http-{self.port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def serve_forever(self) -> None:
        """Blocking variant used by the CLI; calls SelectServer.stop() on Ctrl-C."""
        if self._httpd is None:
            self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler_cls)
            self.port = self._httpd.server_address[1]
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def run(clips: Path, *, port: int = 8765, song: Path | None = None,
        open_browser: bool = True) -> None:
    """CLI entry point — print banner, optionally open browser, block on serve."""
    srv = SelectServer(clips, port=port, song=song)
    srv.start()
    print(f"Server: {srv.url}", flush=True)
    print(f"Clips:  {srv.clips_root}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    if open_browser:
        try:
            import subprocess
            subprocess.run(["open", srv.url], check=False, capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        # Block until interrupted. We use a sleep loop so the daemon HTTP
        # thread keeps running.
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
