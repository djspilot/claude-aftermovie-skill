"""Final-mux audio chain: loudness normalization contract."""
from __future__ import annotations

from pathlib import Path

from aftermovie.render import pipeline as pl


def test_final_mux_appends_loudnorm(tmp_path: Path, monkeypatch):
    """The mux filtergraph must end in a -14 LUFS loudnorm (then 48k
    resample) feeding [a_out], for every audio_mix mode."""
    captured: dict = {}
    monkeypatch.setattr(
        pl, "_run_assemble_with_progress",
        lambda cmd, *a, **k: captured.__setitem__("cmd", cmd),
    )
    for mode in ("music_only", "ducked", "clip_only"):
        plan = {"song": str(tmp_path / "song.mp3"), "music_db": -8}
        pl._final_mux(tmp_path / "missing.mp4", plan, tmp_path / "out.mp4",
                      audio_mix=mode, keep_audio=mode != "music_only")
        cmd = captured["cmd"]
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "loudnorm=I=-14" in fc, (mode, fc)
        # loudnorm is the LAST stage: it produces the mapped [a_out] label.
        assert fc.split("loudnorm")[-1].endswith("[a_out]"), (mode, fc)
        assert "aresample=48000" in fc
