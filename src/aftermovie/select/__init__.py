"""Web GUI for picking which clips go into the aftermovie before rendering.

Composition:

* `service.py` — `SelectionService`, the GUI's domain Interface (no HTTP).
* `server.py` — stdlib HTTP server, a thin Adapter over `SelectionService`.
* `thumbnails.py` — cached 256x256 JPG thumbnail generator.

The CLI `aftermovie select --clips PATH` boots the server, opens a browser,
and blocks until Ctrl-C. The browser writes the user's pick set to
`<clips>/.aftermovie-selection.json` and (when the user clicks Render)
asks the server to run `pipeline_runner.run_auto(...)` against the
filtered source pool.
"""
from aftermovie.select.server import SelectServer, run as run_server  # noqa: F401
from aftermovie.select.service import SelectionService  # noqa: F401
from aftermovie.select.thumbnails import thumb_path_for  # noqa: F401

__all__ = ["SelectServer", "SelectionService", "run_server", "thumb_path_for"]
