"""Speed-ramp render tests.

Covers two layers:
- `_ramp_speeds` + `_prerender_clip` cmd construction (no ffmpeg required).
- Smoke render of a tiny ramp plan against the bundled fixture (skipped
  when ffmpeg is unavailable).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aftermovie.ffmpeg_cmd import ffprobe_json
from aftermovie.render import pipeline as render_pipeline


def test_ramp_speeds_collapses_when_endpoints_close():
    """Differences ≤ 0.05 collapse to constant speed — too subtle to bother."""
    entry = {"speed": 1.0, "speed_start": 1.0, "speed_end": 1.03}
    s0, s1 = render_pipeline._ramp_speeds(entry)
    assert s0 == s1


def test_ramp_speeds_passes_through_real_ramps():
    entry = {"speed": 0.4, "speed_start": 0.4, "speed_end": 1.0}
    assert render_pipeline._ramp_speeds(entry) == (0.4, 1.0)


def test_ramp_speeds_defaults_to_speed_when_ramp_missing():
    entry = {"speed": 1.5}
    assert render_pipeline._ramp_speeds(entry) == (1.5, 1.5)


def _capture_run(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, check=True, capture=False):
        captured.append(list(cmd))
        # Touch the output file so the caller's existence checks pass.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(render_pipeline, "run", fake_run)
    # `_source_has_audio` shells out to ffprobe; stub it so unit tests
    # don't need ffmpeg installed.
    monkeypatch.setattr(render_pipeline, "_source_has_audio", lambda _src: True)
    return captured


def test_prerender_emits_time_varying_setpts(tmp_path, monkeypatch):
    captured = _capture_run(monkeypatch)
    entry = {
        "source": "/dev/null/fake.mp4",
        "start_s": 0.0,
        "end_s": 2.0,
        "out_duration_s": 2.0,
        "speed": 0.6,           # legacy field — ignored when ramp diverges
        "speed_start": 0.4,
        "speed_end": 1.0,
        "audio_interest": 1.0,
    }
    ok = render_pipeline._prerender_clip(
        entry, tmp_path / "out.mp4",
        aspect="16:9", target_res="320x240", target_fps=24,
        lut=None, keep_audio=False,
    )
    assert ok
    cmd = captured[-1]
    vf = cmd[cmd.index("-vf") + 1]
    # Two-segment ramp must include a time-varying expression — `T/` is the
    # tell, since a constant-speed setpts never references T.
    assert "setpts=" in vf
    assert "T/" in vf, f"expected time-varying setpts, got vf={vf!r}"
    # f0=1/0.4=2.5 and the delta f1-f0=1.0-2.5=-1.5 must both be in the expr.
    assert "2.5000" in vf, f"missing start factor in {vf!r}"
    assert "-1.5000" in vf, f"missing factor delta in {vf!r}"
    # Source duration is end-start=2.0 so the normalisation is /2.0000.
    assert "T/2.0000" in vf, f"missing per-frame normalisation in {vf!r}"


def test_prerender_no_ramp_uses_constant_setpts(tmp_path, monkeypatch):
    captured = _capture_run(monkeypatch)
    entry = {
        "source": "/dev/null/fake.mp4",
        "start_s": 0.0,
        "end_s": 2.0,
        "out_duration_s": 2.0,
        "speed": 1.5,
        # No speed_start / speed_end — legacy behaviour.
        "audio_interest": 1.0,
    }
    ok = render_pipeline._prerender_clip(
        entry, tmp_path / "out.mp4",
        aspect="16:9", target_res="320x240", target_fps=24,
        lut=None, keep_audio=False,
    )
    assert ok
    vf = captured[-1][captured[-1].index("-vf") + 1]
    assert "setpts=" in vf
    # Constant setpts has no T reference.
    assert "T/" not in vf, f"expected constant setpts, got vf={vf!r}"


def test_prerender_subtle_ramp_collapses_to_constant(tmp_path, monkeypatch):
    captured = _capture_run(monkeypatch)
    entry = {
        "source": "/dev/null/fake.mp4",
        "start_s": 0.0,
        "end_s": 2.0,
        "out_duration_s": 2.0,
        "speed": 1.0,
        "speed_start": 1.0,
        "speed_end": 1.02,
        "audio_interest": 1.0,
    }
    ok = render_pipeline._prerender_clip(
        entry, tmp_path / "out.mp4",
        aspect="16:9", target_res="320x240", target_fps=24,
        lut=None, keep_audio=False,
    )
    assert ok
    vf = captured[-1][captured[-1].index("-vf") + 1]
    assert "T/" not in vf, f"≤0.05 delta should not emit ramp, got vf={vf!r}"


def test_prerender_ramp_pad_dur_uses_ramp_native_duration(tmp_path, monkeypatch):
    """slot=2.0s, src=1.0s, ramp 0.5→1.0 → native_out=1.0*(2+1)/2=1.5s, pad≈0.5s."""
    captured = _capture_run(monkeypatch)
    entry = {
        "source": "/dev/null/fake.mp4",
        "start_s": 0.0,
        "end_s": 1.0,
        "out_duration_s": 2.0,
        "speed": 0.75,
        "speed_start": 0.5,
        "speed_end": 1.0,
        "audio_interest": 1.0,
    }
    ok = render_pipeline._prerender_clip(
        entry, tmp_path / "out.mp4",
        aspect="16:9", target_res="320x240", target_fps=24,
        lut=None, keep_audio=False,
    )
    assert ok
    vf = captured[-1][captured[-1].index("-vf") + 1]
    assert "tpad=stop_mode=clone:stop_duration=0.500" in vf, \
        f"expected pad_dur≈0.5s, got vf={vf!r}"


# ---------------------------------------------------------------------------
# Smoke: a real ffmpeg render of a tiny ramp plan must produce expected length.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_ramp_plan_renders_to_expected_duration(tmp_path: Path, fixtures_dir: Path,
                                                  tone: Path):
    """A one-Entry plan with a speed ramp renders close to its slot duration."""
    clip = fixtures_dir / "clip_a.mp4"
    out = tmp_path / "ramp.mp4"
    plan = {
        "song": str(tone),
        "song_start_s": 0.0,
        "resolution": "320x240",
        "fps": 24,
        "aspect": "16:9",
        "audio_mix": "music_only",
        "music_db": -8,
        "lut": None,
        "theme": "cinematic",
        "song_meta": {"intro_end_s": 0.0},
        "entries": [
            {
                "source": str(clip),
                "start_s": 0.0,
                "end_s": 4.0,
                "out_duration_s": 3.0,
                "speed": 1.0,
                "speed_start": 1.5,
                "speed_end": 0.8,
                "beat_time_s": 0.0,
                "score": 1.0,
                "reasons": ["test_ramp"],
                "audio_interest": 0.0,
                "source_width": 320,
                "source_height": 240,
            },
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    from aftermovie.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "render", "--plan", str(plan_path), "--output", str(out),
    ])
    args.func(args)

    assert out.exists(), "ramp render did not produce an mp4"
    info = ffprobe_json(out)
    dur = float(info["format"]["duration"])
    # slot is 3.0s and the renderer hard-caps with `-t 3.000`; allow ±0.1s
    # for container/frame quantisation.
    assert 2.9 <= dur <= 3.1, f"expected ~3.0s output, got {dur}"
    # Output must be h264 video.
    codecs = {s.get("codec_name") for s in info.get("streams", [])}
    assert "h264" in codecs, f"expected h264, got {codecs}"
