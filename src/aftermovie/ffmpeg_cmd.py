"""Thin wrappers around ffmpeg / ffprobe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


def log(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """subprocess.run with our defaults."""
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


# ---- streaming progress wrapper -------------------------------------------

# Keys ffmpeg emits in a `-progress pipe:1` block. Floats / ints get coerced;
# the rest stay strings. The block terminator is `progress=continue` (every
# stats period) or `progress=end` (final emit when ffmpeg flushes the encoder).
_PROGRESS_INT_KEYS = frozenset({
    "frame", "fps", "bitrate", "total_size", "out_time_us",
    "out_time_ms", "dup_frames", "drop_frames",
})
_PROGRESS_FLOAT_KEYS = frozenset({"speed"})


def _coerce_progress_value(key: str, raw: str) -> Any:
    """Map raw ffmpeg `-progress` value to int / float / str.

    ffmpeg emits `speed=N/A` early in the stream and `bitrate=N/A` for muxers
    that don't compute one; both collapse to `None` so callers don't have to
    special-case parse failures.
    """
    if raw in ("N/A", ""):
        return None
    if key in _PROGRESS_INT_KEYS:
        try:
            return int(raw)
        except ValueError:
            try:
                return int(float(raw))
            except ValueError:
                return None
    if key in _PROGRESS_FLOAT_KEYS:
        # `0.998x` style — strip the trailing `x` ffmpeg sticks on speed.
        s = raw.rstrip("x")
        try:
            return float(s)
        except ValueError:
            return None
    return raw


def parse_progress_stream(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    """Yield one dict per ffmpeg `-progress` block.

    ffmpeg writes `key=value\\n` pairs and terminates each block with a
    `progress=continue` (mid-stream) or `progress=end` (final) line. This
    helper accumulates pairs until a `progress=` line lands, then yields
    the merged block. Pure-Python parser so it can be unit-tested without
    spawning a subprocess.
    """
    block: dict[str, Any] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "progress":
            # Terminator. Stamp the marker on the block so callers can detect
            # the final `end` block if they want to.
            block["progress"] = value
            yield block
            block = {}
            continue
        block[key] = _coerce_progress_value(key, value)
    # Trailing block without a terminator — emit so callers see the last data.
    if block:
        yield block


def run_with_progress(
    cmd: list[str],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    *,
    check: bool = True,
    total_frames: int | None = None,
    on_pid: Callable[[int], None] | None = None,
) -> subprocess.CompletedProcess:
    """Run ffmpeg with `-progress pipe:1 -nostats`, streaming progress to a callback.

    The `-progress` and `-nostats` flags are appended to `cmd`; callers pass
    the same `cmd` list they'd hand to `run()`. Stdout is consumed by the
    parser; stderr is captured and returned on the CompletedProcess for logs.

    Each parsed block is forwarded to `on_progress(block)`. The block is a
    dict containing the keys ffmpeg emits — `frame`, `out_time_ms`, `speed`,
    `fps`, plus the terminator key `progress` (`"continue"` or `"end"`).

    `total_frames` is forwarded into the block as `total_frames` so the
    callback can compute `fraction = frame / total_frames` without juggling
    closures.

    `on_pid(pid)` fires once with the subprocess pid the moment ffmpeg starts,
    so the caller can surface the PID into `RenderJob.current_ffmpeg_pid` for
    the GUI (and `kill -INT <pid>` from CLI debugging).

    Raises `subprocess.CalledProcessError` when `check=True` and ffmpeg exits
    non-zero — same contract as `run()`.
    """
    full_cmd = list(cmd) + ["-progress", "pipe:1", "-nostats"]
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if on_pid is not None:
        try:
            on_pid(proc.pid)
        except Exception:
            pass

    assert proc.stdout is not None

    def _lines() -> Iterable[str]:
        # readline() returns "" at EOF; iterating the file object would also
        # work but readline gives us tighter control over partial lines.
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            yield line

    if on_progress is not None:
        for block in parse_progress_stream(_lines()):
            if total_frames is not None:
                block.setdefault("total_frames", total_frames)
            try:
                on_progress(block)
            except Exception:
                # A misbehaving callback must not kill the render — log + move on.
                pass
    else:
        # Drain stdout so the pipe buffer doesn't fill and block ffmpeg.
        for _ in _lines():
            pass

    stderr_data = proc.stderr.read() if proc.stderr is not None else ""
    rc = proc.wait()
    completed = subprocess.CompletedProcess(args=full_cmd, returncode=rc,
                                            stdout="", stderr=stderr_data)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, full_cmd, output="", stderr=stderr_data)
    return completed


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
