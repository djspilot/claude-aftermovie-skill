"""Thin wrappers around ffmpeg / ffprobe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def log(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """subprocess.run with our defaults."""
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def ffprobe_json(path: Path) -> dict[str, Any]:
    """Return ffprobe metadata for a file as a dict."""
    res = run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture=True,
    )
    return json.loads(res.stdout)
