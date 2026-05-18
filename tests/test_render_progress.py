"""Tests for Phase A progress UX — `-progress pipe:1` parser, ProgressEvent
plumbing, and `RenderJob` field updates.

Three Seams under test:
  1. `ffmpeg_cmd.parse_progress_stream` — pure-Python parser of ffmpeg's
     key=value progress format. No subprocess; fed synthetic lines.
  2. `render.pipeline.ProgressEvent` dataclass shape — the Interface every
     stage emits into the callback chain.
  3. `select.service._run_render_job` — the closure that maps a stream of
     `ProgressEvent`s onto `RenderJob.progress_percent` + `.stage`.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from PIL import Image

from aftermovie.ffmpeg_cmd import parse_progress_stream
from aftermovie.render.pipeline import ProgressEvent
from aftermovie.select.service import (
    RENDER_STAGE_WEIGHTS,
    RenderJob,
    SelectionService,
    _run_render_job,
)


# ---- ffmpeg -progress parser ----------------------------------------------

def test_parse_progress_stream_yields_block_per_progress_marker() -> None:
    """One `progress=continue` line == one yielded block with coerced types."""
    lines = [
        "frame=42\n",
        "fps=30\n",
        "bitrate=1024kbits/s\n",
        "out_time_ms=1500000\n",
        "speed=0.998x\n",
        "progress=continue\n",
        "frame=84\n",
        "out_time_ms=3000000\n",
        "speed=1.000x\n",
        "progress=end\n",
    ]
    blocks = list(parse_progress_stream(lines))
    assert len(blocks) == 2

    first = blocks[0]
    assert first["frame"] == 42
    assert first["fps"] == 30
    # speed has its trailing `x` stripped and coerces to float.
    assert isinstance(first["speed"], float)
    assert abs(first["speed"] - 0.998) < 1e-6
    assert first["out_time_ms"] == 1_500_000
    assert first["progress"] == "continue"

    last = blocks[1]
    assert last["frame"] == 84
    assert last["progress"] == "end"


def test_parse_progress_stream_handles_na_values() -> None:
    """`speed=N/A` and `bitrate=N/A` early in the stream collapse to None
    instead of leaking the literal string into callbacks."""
    lines = [
        "frame=1\n",
        "bitrate=N/A\n",
        "speed=N/A\n",
        "progress=continue\n",
    ]
    blocks = list(parse_progress_stream(lines))
    assert len(blocks) == 1
    assert blocks[0]["frame"] == 1
    assert blocks[0]["bitrate"] is None
    assert blocks[0]["speed"] is None


def test_parse_progress_stream_ignores_blank_and_keyless_lines() -> None:
    """Blank lines and lines without `=` are dropped (ffmpeg occasionally
    emits a header line during pipe handshake)."""
    lines = [
        "",
        "ffmpeg version 6.0\n",
        "frame=7\n",
        "progress=continue\n",
    ]
    blocks = list(parse_progress_stream(lines))
    assert blocks == [{"frame": 7, "progress": "continue"}]


def test_parse_progress_stream_emits_trailing_partial_block() -> None:
    """If the stream ends without a `progress=` terminator (e.g. ffmpeg got
    SIGKILLed) the partial block is still yielded — better stale data than
    silent loss."""
    lines = ["frame=11\n", "fps=24\n"]
    blocks = list(parse_progress_stream(lines))
    assert blocks == [{"frame": 11, "fps": 24}]


# ---- ProgressEvent dataclass shape ----------------------------------------

def test_progress_event_is_frozen_dataclass() -> None:
    """`ProgressEvent` is the Interface the callback chain uses; making it
    frozen ensures no caller can mutate one event and confuse a downstream
    listener."""
    e = ProgressEvent(stage="prerender", stage_index=3, stage_total=10,
                      fraction_in_stage=0.25, current_pid=12345)
    assert e.stage == "prerender"
    assert e.stage_index == 3
    assert e.stage_total == 10
    assert e.fraction_in_stage == 0.25
    assert e.current_pid == 12345

    with pytest.raises(Exception):
        e.stage = "mux"  # type: ignore[misc]


# ---- end-to-end: stubbed run_auto → RenderJob progression ------------------

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def _seed_folder(tmp_path: Path, fixtures_dir: Path) -> Path:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    shutil.copy(fixtures_dir / "clip_a.mp4", clips_dir / "clip_a.mp4")
    Image.new("RGB", (200, 150), (180, 30, 30)).save(clips_dir / "still.jpg")
    return clips_dir


def test_progress_event_sequence_walks_render_job_to_done(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """Stub `cmd_render` to emit a ProgressEvent sequence; verify
    `RenderJob.progress_percent` advances 0 → 100 and `.stage` transitions
    through prerender → assemble → mux → done.
    """
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    song = fixtures_dir / "tone.wav"

    # The stub stands in for the entire `run_auto` pipeline. It fires three
    # prerender events (each emitting per-clip 0%/100% to exercise the
    # multi-step weighting), then one assemble event, then one mux event.
    def fake_run_auto(clips, song_arg, output, opts, *, progress_cb=None):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"")
        if progress_cb is None:
            return Path(output)
        n_clips = 3
        for i in range(1, n_clips + 1):
            progress_cb(ProgressEvent(
                stage="prerender", stage_index=i, stage_total=n_clips,
                fraction_in_stage=0.0, current_pid=1000 + i,
            ))
            progress_cb(ProgressEvent(
                stage="prerender", stage_index=i, stage_total=n_clips,
                fraction_in_stage=1.0, current_pid=1000 + i,
            ))
        progress_cb(ProgressEvent(
            stage="assemble", stage_index=1, stage_total=1,
            fraction_in_stage=1.0, current_pid=2000,
        ))
        progress_cb(ProgressEvent(
            stage="mux", stage_index=1, stage_total=1,
            fraction_in_stage=1.0, current_pid=3000,
        ))
        return Path(output)

    # `_run_render_job` resolves the EffectiveConfig + builds an AutoOpts
    # before calling `run_auto`; swap that out so we don't pay analyze/score
    # cost in this test.
    monkeypatch.setattr("aftermovie.pipeline_runner.run_auto", fake_run_auto,
                        raising=True)

    job = RenderJob(job_id="test-job-1")
    overrides = {"preview": True, "theme": "cinematic"}
    _run_render_job(job, clips_dir, song, tmp_path / "out.mp4", overrides)

    assert job.state == "done", f"expected done, got {job.state} (error={job.error})"
    assert job.progress_percent == 100.0, job.progress_percent
    assert job.stage == "done"
    # The percentage walked through prerender's 75% budget then assemble + mux.
    # We can't assert on intermediate snapshots from the outside (the worker
    # ran inline in this test thread), but we can sanity-check the closure
    # math by re-driving it manually below.


def test_progress_closure_advances_monotonically_through_stages(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """Drive the progress closure directly and assert each event monotonically
    bumps `progress_percent` and `stage` transitions land in the right order.
    """
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    song = fixtures_dir / "tone.wav"
    snapshots: list[dict] = []

    def fake_run_auto(clips, song_arg, output, opts, *, progress_cb=None):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"")
        # Snapshot the job after each event so we can assert monotonicity.
        events = [
            ProgressEvent("prerender", 1, 4, 0.0, 100),
            ProgressEvent("prerender", 1, 4, 1.0, 100),
            ProgressEvent("prerender", 2, 4, 0.5, 101),
            ProgressEvent("prerender", 4, 4, 1.0, 103),
            ProgressEvent("assemble", 1, 1, 0.5, 200),
            ProgressEvent("assemble", 1, 1, 1.0, 200),
            ProgressEvent("mux", 1, 1, 1.0, 300),
        ]
        for e in events:
            if progress_cb is not None:
                progress_cb(e)
            snapshots.append({
                "stage": job.stage,
                "stage_index": job.stage_index,
                "stage_total": job.stage_total,
                "progress_percent": job.progress_percent,
                "pid": job.current_ffmpeg_pid,
            })
        return Path(output)

    import aftermovie.pipeline_runner as pr
    orig = pr.run_auto
    pr.run_auto = fake_run_auto
    try:
        job = RenderJob(job_id="walk-1")
        _run_render_job(job, clips_dir, song, tmp_path / "out.mp4",
                        {"preview": True})
    finally:
        pr.run_auto = orig

    # Each new snapshot must be >= previous (monotonic).
    pcts = [snap["progress_percent"] for snap in snapshots]
    for prev, cur in zip(pcts, pcts[1:]):
        assert cur >= prev, f"progress regressed: {prev} → {cur}"

    # Final stage in the walked sequence is `mux` (the closure's
    # post-`run_auto` _set_stage("done") happens after these snapshots).
    assert snapshots[-1]["stage"] == "mux"

    # Last in-stream pct is the cumulative weights of analyze+score+prerender
    # +assemble+mux ≈ 100 (we hit fraction=1.0 on each terminal stage).
    assert pcts[-1] >= sum(RENDER_STAGE_WEIGHTS[s]
                           for s in ("prerender", "assemble", "mux")) - 0.01

    # PID surfaced from the last event of each stage. The post-run "done"
    # transition clears the field (no live ffmpeg subprocess) so we assert
    # the in-stream snapshot, not the final RenderJob state.
    assert snapshots[-1]["pid"] == 300


def test_render_job_status_includes_progress_fields(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """`/api/status/<id>` (via `SelectionService.status`) carries the new
    fields with sensible defaults when the job hasn't started yet."""
    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)

    # Inject a job directly so we don't need to spawn a worker.
    job = RenderJob(job_id="status-1")
    with svc._jobs_lock:
        svc._jobs[job.job_id] = job

    snap = svc.status(job.job_id)
    assert snap is not None
    # Defaults: empty stage, zero counters, no ETA / pid yet.
    assert snap["stage"] == ""
    assert snap["stage_index"] == 0
    assert snap["stage_total"] == 0
    assert snap["progress_percent"] == 0.0
    assert snap["eta_s"] is None
    assert snap["current_ffmpeg_pid"] is None
    assert snap["cpu_seconds_used"] == 0.0


def test_render_stage_weights_sum_to_100() -> None:
    """The per-stage weight table is the math behind `progress_percent`.
    If it doesn't sum to 100 the bar will either undershoot or overshoot."""
    total = sum(RENDER_STAGE_WEIGHTS.values())
    assert abs(total - 100.0) < 1e-6, f"weights sum to {total}, not 100"
