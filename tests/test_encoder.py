"""Tests for the encoder-profile Module (B1)."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aftermovie.render.chip import ChipInfo
from aftermovie.render.encoder import (
    H264_VT,
    HEVC_VT,
    X264,
    hwaccel_input_flags,
    select_default,
    select_from_env,
    vfilter_input_shim,
    vfilter_output_guard,
)


def _apple_chip() -> ChipInfo:
    return ChipInfo(
        brand="Apple M5 Pro", arch="arm64",
        perf_cores=10, eff_cores=5, media_engines=1,
    )


def _intel_chip() -> ChipInfo:
    return ChipInfo(
        brand="Intel Core i9", arch="x86_64",
        perf_cores=8, eff_cores=0, media_engines=0,
    )


def _generic_chip() -> ChipInfo:
    return ChipInfo(
        brand="Generic", arch="x86_64",
        perf_cores=4, eff_cores=0, media_engines=0,
    )


def test_x264_profile_matches_legacy_cmd():
    """The X264 profile must reproduce the pre-refactor flag list."""
    assert X264.video_args == [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
    ]
    assert X264.pix_fmt == "yuv420p"
    assert X264.is_hardware is False


def test_vt_profiles_request_videotoolbox_codecs():
    assert H264_VT.video_args[:2] == ["-c:v", "h264_videotoolbox"]
    assert HEVC_VT.video_args[:2] == ["-c:v", "hevc_videotoolbox"]
    # hvc1 tag is critical for Apple QuickTime compatibility.
    assert "-tag:v" in HEVC_VT.video_args
    assert "hvc1" in HEVC_VT.video_args
    assert H264_VT.is_hardware is True
    assert HEVC_VT.is_hardware is True


def test_select_default_apple_picks_vt_when_available():
    """On Apple Silicon with VT present, default is HEVC_VT."""
    with patch("aftermovie.render.encoder._available_encoders",
               return_value=frozenset({
                   "libx264", "h264_videotoolbox", "hevc_videotoolbox",
               })):
        profile = select_default(_apple_chip())
    assert profile is HEVC_VT


def test_select_default_apple_falls_through_when_vt_missing():
    """No VT encoders → x264 even on Apple Silicon."""
    with patch("aftermovie.render.encoder._available_encoders",
               return_value=frozenset({"libx264"})):
        profile = select_default(_apple_chip())
    assert profile is X264


def test_select_default_non_apple_always_x264():
    """Intel / generic hosts get x264 regardless of what ffmpeg reports."""
    fake_encoders = frozenset({
        "libx264", "h264_videotoolbox", "hevc_videotoolbox",
    })
    with patch("aftermovie.render.encoder._available_encoders",
               return_value=fake_encoders):
        assert select_default(_intel_chip()) is X264
        assert select_default(_generic_chip()) is X264


def test_select_from_env_x264_override_wins_on_apple():
    """AFTERMOVIE_VIDEO_CODEC=x264 forces x264 even on a VT-capable host."""
    fake_encoders = frozenset({
        "libx264", "h264_videotoolbox", "hevc_videotoolbox",
    })
    with patch.dict(os.environ, {"AFTERMOVIE_VIDEO_CODEC": "x264"}), \
         patch("aftermovie.render.encoder._available_encoders",
               return_value=fake_encoders):
        profile = select_from_env(_apple_chip())
    assert profile is X264


def test_select_from_env_h264_vt_override():
    fake_encoders = frozenset({
        "libx264", "h264_videotoolbox", "hevc_videotoolbox",
    })
    with patch.dict(os.environ, {"AFTERMOVIE_VIDEO_CODEC": "h264_vt"}), \
         patch("aftermovie.render.encoder._available_encoders",
               return_value=fake_encoders):
        profile = select_from_env(_apple_chip())
    assert profile is H264_VT


def test_select_from_env_auto_defers_to_select_default():
    fake_encoders = frozenset({
        "libx264", "h264_videotoolbox", "hevc_videotoolbox",
    })
    with patch.dict(os.environ, {"AFTERMOVIE_VIDEO_CODEC": "auto"}), \
         patch("aftermovie.render.encoder._available_encoders",
               return_value=fake_encoders):
        profile = select_from_env(_apple_chip())
    assert profile is HEVC_VT


def test_select_from_env_falls_back_when_requested_codec_missing():
    """Stale env asking for VT on a Linux box → fall through to x264."""
    with patch.dict(os.environ, {"AFTERMOVIE_VIDEO_CODEC": "hevc_vt"}), \
         patch("aftermovie.render.encoder._available_encoders",
               return_value=frozenset({"libx264"})):
        profile = select_from_env(_intel_chip())
    assert profile is X264


def test_select_from_env_unset_is_treated_as_auto():
    fake_encoders = frozenset({"libx264", "hevc_videotoolbox"})
    env = {k: v for k, v in os.environ.items() if k != "AFTERMOVIE_VIDEO_CODEC"}
    with patch.dict(os.environ, env, clear=True), \
         patch("aftermovie.render.encoder._available_encoders",
               return_value=fake_encoders):
        profile = select_from_env(_apple_chip())
    assert profile is HEVC_VT


def test_hwaccel_input_flags_only_for_hw_profiles():
    assert hwaccel_input_flags(X264) == []
    assert "videotoolbox" in hwaccel_input_flags(H264_VT)
    assert "-hwaccel" in hwaccel_input_flags(HEVC_VT)


def test_vfilter_shims_only_for_hw_profiles():
    assert vfilter_input_shim(X264) is None
    assert vfilter_output_guard(X264) is None
    shim = vfilter_input_shim(HEVC_VT)
    assert shim is not None
    assert "nv12" in shim
    assert vfilter_output_guard(HEVC_VT) == "format=yuv420p"
