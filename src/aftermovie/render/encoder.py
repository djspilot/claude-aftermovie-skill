"""Encoder-profile Module.

Defines the `EncoderProfile` Interface — a small bundle of ffmpeg flags
that the pipeline's cmd-list construction consumes in place of hard-coded
codec arguments. Three concrete profiles ship today:

    X264      libx264 -preset fast -crf 20            (legacy / fallback)
    H264_VT   h264_videotoolbox -b:v 8M               (Apple Silicon HW)
    HEVC_VT   hevc_videotoolbox -b:v 6M -tag:v hvc1   (Apple Silicon HW, default)

`select_default(chip)` is the Seam between hardware detection (see
`aftermovie.render.chip`) and the renderer. The `AFTERMOVIE_VIDEO_CODEC`
env var overrides selection — `auto` defers to `select_default`, any of
{`x264`, `h264_vt`, `hevc_vt`} forces that profile.

Selection deliberately probes ffmpeg's `-encoders` output once per
process and caches the result, so x264 boxes (CI, Linux) fall back
gracefully when an env var asks for a VT profile that isn't compiled in.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

from aftermovie.render.chip import ChipInfo


@dataclass(frozen=True)
class EncoderProfile:
    """ffmpeg encoder bundle consumed by the renderer.

    Attributes:
        name:        Short identifier used in env files, logs, and tests
                     (`x264`, `h264_vt`, `hevc_vt`).
        video_args:  The `-c:v ...` flag list, including bitrate / preset /
                     tag flags. Does NOT include `-pix_fmt` (carried
                     separately so the filter chain can mirror it).
        audio_args:  Audio codec flags; shared across profiles today but
                     kept on the profile so future codecs (e.g. ProRes)
                     can override.
        pix_fmt:     Output pixel format. Doubles as the value used by
                     the pixfmt-guard Seam at the tail of the vfilter
                     chain when the encoder is a VT variant.
        is_hardware: True for VideoToolbox profiles — flips the
                     `-hwaccel videotoolbox` decode path on at the input
                     side of the cmd list.
    """

    name: str
    video_args: list[str] = field(default_factory=list)
    audio_args: list[str] = field(default_factory=list)
    pix_fmt: str = "yuv420p"
    is_hardware: bool = False


# ---- Concrete profiles ------------------------------------------------------

X264 = EncoderProfile(
    name="x264",
    video_args=["-c:v", "libx264", "-preset", "fast", "-crf", "20"],
    audio_args=["-c:a", "aac"],
    pix_fmt="yuv420p",
    is_hardware=False,
)

H264_VT = EncoderProfile(
    name="h264_vt",
    # `-realtime 0` lets the encoder spend time on rate control instead of
    # the low-latency fast path. `-allow_sw 1` keeps the encoder usable on
    # rare configurations where the HW encoder rejects a particular
    # resolution / pixfmt combo (it falls back to a SW path inside
    # VideoToolbox rather than failing the whole render).
    video_args=[
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-realtime", "0",
        "-allow_sw", "1",
    ],
    audio_args=["-c:a", "aac"],
    pix_fmt="yuv420p",
    is_hardware=True,
)

HEVC_VT = EncoderProfile(
    name="hevc_vt",
    # `-tag:v hvc1` is critical — without it, Apple's QuickTime / Finder
    # preview won't recognize the HEVC stream and will refuse to play.
    video_args=[
        "-c:v", "hevc_videotoolbox",
        "-b:v", "6M",
        "-tag:v", "hvc1",
        "-allow_sw", "1",
    ],
    audio_args=["-c:a", "aac"],
    pix_fmt="yuv420p",
    is_hardware=True,
)


_BY_NAME: dict[str, EncoderProfile] = {
    X264.name: X264,
    H264_VT.name: H264_VT,
    HEVC_VT.name: HEVC_VT,
}


# ---- ffmpeg availability probe ----------------------------------------------

@lru_cache(maxsize=1)
def _available_encoders() -> frozenset[str]:
    """Names ffmpeg reports under `-encoders`. Cached for the process."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return frozenset()
    names: set[str] = set()
    for line in res.stdout.splitlines():
        # Lines look like ` V....D libx264              libx264 H.264 ...`.
        # First non-flag token after the flag column is the encoder name.
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


def has_encoder(name: str) -> bool:
    """True iff ffmpeg ships the named encoder."""
    return name in _available_encoders()


def _profile_available(profile: EncoderProfile) -> bool:
    """Whether ffmpeg has the codec this profile asks for."""
    # video_args is `['-c:v', 'libx264', ...]` — the encoder name is index 1.
    if len(profile.video_args) < 2 or profile.video_args[0] != "-c:v":
        return True
    return has_encoder(profile.video_args[1])


# ---- Selection Seam ---------------------------------------------------------

def select_default(chip: ChipInfo) -> EncoderProfile:
    """Pick a default encoder for the host chip.

    Apple Silicon with `hevc_videotoolbox` available → HEVC_VT (best
    quality-per-bit, and Apple-native). Falls back to H264_VT, then x264.
    Non-Apple hardware always gets x264.
    """
    if chip.is_apple_silicon:
        if _profile_available(HEVC_VT):
            return HEVC_VT
        if _profile_available(H264_VT):
            return H264_VT
    return X264


def select_from_env(chip: ChipInfo) -> EncoderProfile:
    """Resolve `AFTERMOVIE_VIDEO_CODEC` against the available encoders.

    `auto` (default, also matches empty/unset) → `select_default(chip)`.
    A profile name forces that profile when ffmpeg has it, else falls
    back to `select_default(chip)` so a stale env file on a Linux box
    doesn't crash the renderer.
    """
    raw = (os.environ.get("AFTERMOVIE_VIDEO_CODEC") or "auto").strip().lower()
    if raw in ("", "auto"):
        return select_default(chip)
    profile = _BY_NAME.get(raw)
    if profile is None:
        # Unknown value — surface via fallback rather than raising mid-render.
        return select_default(chip)
    if not _profile_available(profile):
        return select_default(chip)
    return profile


# ---- Decode + pixfmt-guard helpers (B2 + B5) --------------------------------

def hwaccel_input_flags(profile: EncoderProfile) -> list[str]:
    """Flags to prepend before each `-i <input>` when HW encode is in use.

    Returning `[]` for SW profiles keeps the caller branch-free.
    """
    if not profile.is_hardware:
        return []
    return [
        "-hwaccel", "videotoolbox",
        "-hwaccel_output_format", "videotoolbox",
    ]


def vfilter_input_shim(profile: EncoderProfile) -> str | None:
    """Filter prefix that ensures SW filters can touch decoded frames.

    When `-hwaccel videotoolbox -hwaccel_output_format videotoolbox` is
    active, ffmpeg may either (a) actually decode on VT and hand us a
    hardware surface, or (b) silently fall back to a SW decoder (some
    codecs / resolutions VT can't handle). The next filter in the chain
    is a CPU `scale` / `crop`, which can't read a VT surface.

    A bare `format=nv12` filter handles both cases: ffmpeg's automatic
    `hwdownload` kicks in when the upstream frame is on the GPU, and
    it's a no-op pixfmt conversion when the upstream frame is already
    on the CPU.

    Returns None for SW profiles.
    """
    if not profile.is_hardware:
        return None
    return "format=nv12"


def vfilter_output_guard(profile: EncoderProfile) -> str | None:
    """Trailing format step so VT encoders accept the filter-graph output.

    VideoToolbox encoders reject some intermediate pixel formats produced
    by `setpts` / `tpad` / `lut3d`. Re-asserting `yuv420p` at the tail of
    the chain costs ~nothing and avoids "Filter ... not compatible with
    the encoder" errors.

    Returns None for SW profiles (libx264 handles arbitrary YUV inputs).
    """
    if not profile.is_hardware:
        return None
    return f"format={profile.pix_fmt}"


__all__ = [
    "EncoderProfile",
    "X264",
    "H264_VT",
    "HEVC_VT",
    "has_encoder",
    "hwaccel_input_flags",
    "select_default",
    "select_from_env",
    "vfilter_input_shim",
    "vfilter_output_guard",
]
