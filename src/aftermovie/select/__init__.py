"""Web GUI for picking which clips go into the aftermovie before rendering.

Composition:

* `server.py` — stdlib HTTP server with JSON endpoints + thumbnail serving.
* `thumbnails.py` — cached 256x256 JPG thumbnail generator.

The CLI `aftermovie select --clips PATH` boots the server, opens a browser,
and blocks until Ctrl-C. The browser writes the user's pick set to
`<clips>/.aftermovie-selection.json` and (when the user clicks Render)
asks the server to run `pipeline_runner.run_auto(...)` against the
filtered source pool.
"""
from aftermovie.select.server import SelectServer, run as run_server  # noqa: F401
from aftermovie.select.thumbnails import thumb_path_for  # noqa: F401

__all__ = ["SelectServer", "run_server", "thumb_path_for"]
