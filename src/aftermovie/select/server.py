"""Stdlib HTTP server backing the `aftermovie select` web GUI.

Endpoints (all JSON unless otherwise noted):

    GET  /                          → index.html (or a placeholder)
    GET  /api/sources               → [{path, name, kind, thumb_url, selected}, ...]
    GET  /api/options               → dropdown choices for theme/lut/etc.
    GET  /api/plan                  → latest Plan JSON for clips_root
    GET  /api/preferences           → favorited/banned/pinned dict
    GET  /thumbs/<sha1>.jpg         → cached 256x256 JPG (or 404)
    POST /api/selection             → save .aftermovie-selection.json
    POST /api/preferences           → save .aftermovie-preferences.json
    POST /api/render                → launch run_auto in a worker thread → {job_id}
    GET  /api/status/<job_id>       → {state, output_path, log_tail}
    GET  /api/import-sources        → [{name, label, available}, ...]
    POST /api/import                → launch import worker → {job_id, dest_folder}
    GET  /api/import-status/<job_id> → {state, copied, skipped, failed, total, ...}
    GET  /api/song                  → {path, name, duration_s?, tempo_bpm?} | {path: null}
    POST /api/upload-song           → multipart; copies into the song cache → {path, ...}
    POST /api/set-song-path         → {path} → validates + activates the song
    GET  /api/candidate-songs       → [{path, name, duration_s | null}, ...]
    GET  /api/song-info?path=…      → {duration_s, tempo_bpm, energy_curve_samples}
    GET  /api/recent-songs          → [{path, name, last_used_iso}, ...]

The server is intentionally synchronous and single-threaded except for the
render worker (one per render job). Stdlib HTTPServer is the only
dependency here — we don't want to pull Flask/FastAPI into the runtime.

`_Handler` is a thin Adapter: each route parses the request, calls a
`SelectionService` method, and serializes the response. The domain logic
(source discovery, sidecar I/O, plan lookup, render-job lifecycle) all
lives in `service.SelectionService` so a CLI / TUI / inline test can
exercise the same operations without HTTP.

The CLI side is `cli.py::cmd_select`; this module is import-free of argparse
so it can also be driven directly from tests.
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from aftermovie.select.service import SelectionService


# ---- multipart helper -------------------------------------------------------

def _parse_multipart_one_file(body: bytes, boundary: str) -> tuple[str, bytes]:
    """Parse a single-file multipart/form-data body; return `(filename, payload)`.

    Minimal stdlib-only parser — handles the one shape the song upload uses
    (one file part). Raises `ValueError` on malformed input so the HTTP
    Adapter can map that to a 400. Filename comes from the part's
    `Content-Disposition` header; if missing, we default to "song".
    """
    delim = ("--" + boundary).encode()
    parts = body.split(delim)
    # parts[0] is the preamble (usually empty); parts[-1] is the closing "--\r\n".
    for part in parts[1:-1]:
        part = part.lstrip(b"\r\n")
        if not part:
            continue
        # Split headers from payload at the first blank line.
        header_blob, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue
        # Strip the trailing CRLF that precedes the next boundary delimiter.
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        try:
            header_text = header_blob.decode("utf-8", errors="replace")
        except UnicodeDecodeError as exc:
            raise ValueError("bad multipart headers") from exc
        # Find the filename in Content-Disposition.
        m = re.search(r'filename="([^"]*)"', header_text)
        filename = m.group(1) if m else "song"
        return filename, payload
    raise ValueError("no file part in multipart body")


# ---- HTTP handler -----------------------------------------------------------

# Class-level state. ThreadingHTTPServer instantiates a fresh handler per
# request, so we hang the per-server SelectionService off the class attrs
# the SelectServer wires up at boot.
class _Handler(BaseHTTPRequestHandler):
    service: SelectionService = None  # type: ignore[assignment]
    static_dir: Path | None = None

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
                self._send_json(self.service.available_options())
                return
            if route == "/api/plan":
                self._serve_plan()
                return
            if route == "/api/preferences":
                self._send_json(self.service.get_preferences())
                return
            m = re.match(r"^/thumbs/([A-Fa-f0-9]+)\.jpg$", route)
            if m:
                self._serve_thumb(m.group(1))
                return
            m = re.match(r"^/api/status/([A-Za-z0-9\-]+)$", route)
            if m:
                self._serve_status(m.group(1))
                return
            if route == "/api/import-sources":
                self._serve_import_sources()
                return
            m = re.match(r"^/api/import-status/([A-Za-z0-9\-]+)$", route)
            if m:
                self._serve_import_status(m.group(1))
                return
            if route == "/api/song":
                self._send_json(self.service.current_song())
                return
            if route == "/api/candidate-songs":
                self._send_json(self.service.list_candidate_songs())
                return
            if route == "/api/recent-songs":
                self._send_json(self.service.recent_songs())
                return
            if route == "/api/song-info":
                self._serve_song_info(parsed.query)
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
            if route == "/api/preferences":
                self._handle_preferences(self._read_json())
                return
            if route == "/api/render":
                self._handle_render(self._read_json())
                return
            if route == "/api/import":
                self._handle_import(self._read_json())
                return
            if route == "/api/set-song-path":
                self._handle_set_song_path(self._read_json())
                return
            if route == "/api/upload-song":
                self._handle_upload_song()
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
        rows = self.service.list_sources()
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

    def _serve_plan(self) -> None:
        plan = self.service.latest_plan()
        if plan is None:
            self._send_json({"error": "no_plan"}, status=404)
            return
        self._send_json(plan)

    def _serve_thumb(self, key: str) -> None:
        body = self.service.thumb_for_key(key)
        if body is None:
            self._send_json({"error": "not_found"}, status=404)
            return
        self._send_bytes(body, "image/jpeg")

    def _serve_status(self, job_id: str) -> None:
        body = self.service.status(job_id)
        if body is None:
            self._send_json({"error": "not_found", "job_id": job_id}, status=404)
            return
        self._send_json(body)

    def _handle_selection(self, body: dict[str, Any]) -> None:
        raw = body.get("excluded")
        if not isinstance(raw, list):
            self._send_json({"error": "bad_request",
                             "detail": "expected JSON body with 'excluded': []"},
                            status=400)
            return
        result = self.service.save_selection(raw)
        self._send_json({"ok": True, "sidecar": str(result.sidecar),
                         "filename": result.filename, "n_excluded": result.n_items})

    def _handle_preferences(self, body: dict[str, Any]) -> None:
        if not isinstance(body, dict):
            self._send_json({"error": "bad_request",
                             "detail": "expected JSON object"},
                            status=400)
            return
        self.service.save_preferences(body)
        self._send_json({"ok": True})

    def _handle_render(self, body: dict[str, Any]) -> None:
        job = self.service.start_render(body)
        self._send_json({"job_id": job.job_id})

    def _serve_import_sources(self) -> None:
        try:
            self._send_json(self.service.available_import_sources())
        except ModuleNotFoundError:
            self._send_json({"error": "import_module_missing"}, status=503)

    def _serve_import_status(self, job_id: str) -> None:
        body = self.service.import_status(job_id)
        if body is None:
            self._send_json({"error": "not_found", "job_id": job_id}, status=404)
            return
        self._send_json(body)

    # ---- song picker handlers (D1-D4) ----

    def _handle_set_song_path(self, body: dict[str, Any]) -> None:
        """Validate `path` exists + is audio; activate it on the service."""
        raw = body.get("path")
        if not isinstance(raw, str) or not raw:
            self._send_json({"error": "bad_request",
                             "detail": "expected JSON body with 'path'"},
                            status=400)
            return
        try:
            info = self.service.set_song(Path(raw))
        except ValueError as e:
            self._send_json({"error": "bad_request", "detail": str(e)},
                            status=400)
            return
        self._send_json({"ok": True, **info})

    def _handle_upload_song(self) -> None:
        """Copy a multipart-uploaded song into the server cache + activate it.

        We parse the multipart body ourselves (no Flask/FastAPI dep). The
        cache lives at `~/.skills-data/aftermovie/songs/<sha1>.<ext>` so
        re-uploading the same file is idempotent (the SHA1 of the content
        keys the destination).
        """
        ctype = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "bad_request",
                             "detail": "expected multipart/form-data"},
                            status=400)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json({"error": "bad_request",
                             "detail": "empty upload"},
                            status=400)
            return
        # Extract boundary.
        m = re.search(r"boundary=(.+?)(?:;|$)", ctype)
        if not m:
            self._send_json({"error": "bad_request",
                             "detail": "missing multipart boundary"},
                            status=400)
            return
        boundary = m.group(1).strip().strip('"')
        try:
            filename, payload = _parse_multipart_one_file(
                self.rfile.read(length), boundary,
            )
        except ValueError as e:
            self._send_json({"error": "bad_request", "detail": str(e)},
                            status=400)
            return
        if not payload:
            self._send_json({"error": "bad_request",
                             "detail": "no file content"},
                            status=400)
            return

        # Cache the upload under SHA1(content).<ext>. The cache lives next to
        # the song-analysis cache so both song-related blobs collocate under
        # ~/.skills-data/aftermovie/.
        import hashlib
        from aftermovie.config import data_dir

        sha = hashlib.sha1(payload).hexdigest()
        ext = Path(filename or "song").suffix.lower() or ".mp3"
        songs_dir = data_dir() / "songs"
        songs_dir.mkdir(parents=True, exist_ok=True)
        dest = songs_dir / f"{sha}{ext}"
        if not dest.is_file():
            try:
                dest.write_bytes(payload)
            except OSError as e:
                self._send_json({"error": "server_error", "detail": str(e)},
                                status=500)
                return

        try:
            info = self.service.set_song(dest)
        except ValueError as e:
            self._send_json({"error": "bad_request", "detail": str(e)},
                            status=400)
            return
        self._send_json({"ok": True, **info})

    def _serve_song_info(self, query: str) -> None:
        """Lazy-analyze the song at `?path=…` and return cached JSON."""
        params = parse_qs(query or "")
        raw = (params.get("path") or [""])[0]
        if not raw:
            self._send_json({"error": "bad_request",
                             "detail": "missing 'path' query"},
                            status=400)
            return
        try:
            info = self.service.song_info(Path(raw))
        except ValueError as e:
            self._send_json({"error": "bad_request", "detail": str(e)},
                            status=400)
            return
        self._send_json(info)

    def _handle_import(self, body: dict[str, Any]) -> None:
        try:
            job = self.service.start_import(
                since=str(body.get("since", "")),
                until=str(body.get("until", "")),
                source_names=list(body.get("sources") or []),
                dest_parent=body.get("dest_parent"),
                dry_run=bool(body.get("dry_run", False)),
            )
        except ValueError as e:
            self._send_json({"error": "bad_request", "detail": str(e)}, status=400)
            return
        except ModuleNotFoundError:
            self._send_json({"error": "import_module_missing"}, status=503)
            return
        self._send_json({"job_id": job.job_id, "dest_folder": job.dest_folder})


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

        self.service = SelectionService(clips_root, song_default=song)

        # Fresh per-server handler subclass so multiple SelectServers can
        # coexist (e.g. parallel tests) without sharing class-level state.
        class Handler(_Handler):
            pass
        Handler.service = self.service
        Handler.static_dir = static_dir
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
