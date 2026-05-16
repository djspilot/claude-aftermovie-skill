#!/usr/bin/env python3
"""
Generate deterministic test fixtures.

Run from the repo root: `python tests/fixtures/build.py`.
Produces:
  - clip_a.mp4, clip_b.mp4, clip_c.mp4 (5s testsrc2 with varied motion/color)
  - tone.wav (10s 120 BPM click + sine)
  - hilight_sample.mp4 (1s mp4 with a hand-crafted HMMT atom)

Run only when you need to regenerate them — they are committed binaries.
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def make_clip(name: str, color: str, rate: int = 30, duration: int = 5) -> None:
    out = HERE / name
    # mandelbrot/testsrc2 give us motion the signalstats filter can pick up.
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"mandelbrot=size=320x240:rate={rate}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-vf", f"colorbalance=rs=0.3:gs={'0.3' if 'g' in color else '0'}:bs={'0.3' if 'b' in color else '0'}",
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out),
    ])
    print(f"  built {out.name}")


def make_tone(name: str = "tone.wav", duration: int = 10, bpm: int = 120) -> None:
    out = HERE / name
    period = 60.0 / bpm
    # Sine tone + click every beat.
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
        "-f", "lavfi", "-i", f"aevalsrc='0.6*sin(2*PI*1500*t)*exp(-30*mod(t,{period:.4f}))':duration={duration}",
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest",
        "-ac", "1",
        str(out),
    ])
    print(f"  built {out.name}")


def make_hilight_fixture(name: str = "hilight_sample.mp4") -> None:
    """
    Build a 1-second valid MP4, then inject a minimal HMMT box into the udta
    container so read_hilight_tags() can find it. HiLight at 500 ms.
    """
    out = HERE / name
    # Build a base MP4 with a moov/udta box (writes empty by default in ffmpeg).
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10",
        "-t", "1",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-metadata", "comment=hilight-fixture",
        str(out),
    ])
    # Append a stand-alone HMMT atom outside the moov — read_hilight_tags
    # uses a byte search for "HMMT" so it does not require formal box nesting.
    timestamps_ms = [500]
    payload = struct.pack(">I", len(timestamps_ms))  # count
    payload += b"".join(struct.pack(">I", t) for t in timestamps_ms)
    box = b"HMMT" + payload
    box_with_size = struct.pack(">I", len(box) + 4) + box
    with open(out, "ab") as f:
        f.write(box_with_size)
    print(f"  built {out.name} ({len(timestamps_ms)} hilight tag)")


def main() -> None:
    print(f"Building fixtures in {HERE}")
    make_clip("clip_a.mp4", "r")
    make_clip("clip_b.mp4", "g")
    make_clip("clip_c.mp4", "b")
    make_tone()
    make_hilight_fixture()
    print("done.")


if __name__ == "__main__":
    main()
